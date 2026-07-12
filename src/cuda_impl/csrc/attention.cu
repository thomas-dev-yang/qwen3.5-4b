#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define QWEN35_QUERY_HEADS 16
#define QWEN35_KV_HEADS 4 // same as num_kv_groups
#define QWEN35_HEAD_DIM 256
#define QWEN35_KV_GROUPS (QWEN35_QUERY_HEADS / QWEN35_KV_HEADS)
#define QWEN35_ATTENTION_SCALE 0.0625F

// Typed CUDA entrypoint. The torch::Tensor function below is responsible for
// validating tensors, allocating output, extracting these pointers and sizes,
// and launching this kernel on PyTorch's current CUDA stream.
__global__ void qwen35_attention_kernel(const __nv_bfloat16 *__restrict__ query,
                                        const __nv_bfloat16 *__restrict__ key,
                                        const __nv_bfloat16 *__restrict__ value,
                                        __nv_bfloat16 *__restrict__ output, int batch_size,
                                        int query_tokens, int key_tokens) {
  // One thread owns one complete output vector:
  //   output[batch][query_token][query_head][0..255]
  // The intentionally simple launch is:
  //   grid  = (query_tokens, 16 query heads, batch)
  //   block = (1, 1, 1)
  const int query_token = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;

  if (batch >= batch_size || query_head >= QWEN35_QUERY_HEADS || query_token >= query_tokens) {
    return;
  }

  const int kv_head = query_head / QWEN35_KV_GROUPS;

  const int query_offset =
      ((batch * QWEN35_QUERY_HEADS + query_head) * query_tokens + query_token) * QWEN35_HEAD_DIM;

  // Queries are the newest query_tokens positions in the K/V sequence. This
  // gives query_token 0 access to the cached prefix and itself, but not later
  // tokens in the same prefill chunk.
  const int cached_tokens = key_tokens - query_tokens;
  const int visible_key_tokens = cached_tokens + query_token + 1;

  float output_accumulator[QWEN35_HEAD_DIM] = {0.0F};
  float running_max = -FLT_MAX;
  float running_sum = 0.0F;

  for (int key_token = 0; key_token < visible_key_tokens; ++key_token) {
    const int kv_offset =
        ((batch * QWEN35_KV_HEADS + kv_head) * key_tokens + key_token) * QWEN35_HEAD_DIM;

    float score = 0.0F;
    for (int feature = 0; feature < QWEN35_HEAD_DIM; ++feature) {
      score += __bfloat162float(query[query_offset + feature]) *
               __bfloat162float(key[kv_offset + feature]);
    }
    score *= QWEN35_ATTENTION_SCALE;

    // Online softmax merge. The previous numerator and denominator are in the
    // exp(score - running_max) coordinate system, so both must be rescaled if
    // this score establishes a new maximum.
    const float next_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - next_max);
    const float current_weight = expf(score - next_max);
    running_sum = running_sum * previous_scale + current_weight;

    for (int feature = 0; feature < QWEN35_HEAD_DIM; ++feature) {
      output_accumulator[feature] = output_accumulator[feature] * previous_scale +
                                    current_weight * __bfloat162float(value[kv_offset + feature]);
    }
    running_max = next_max;
  }

  const int output_offset =
      ((batch * query_tokens + query_token) * QWEN35_QUERY_HEADS + query_head) * QWEN35_HEAD_DIM;
  for (int feature = 0; feature < QWEN35_HEAD_DIM; ++feature) {
    output[output_offset + feature] = __float2bfloat16(output_accumulator[feature] / running_sum);
  }
}

// PyTorch boundary for Qwen3.5 full attention. This is not the whole attention
// module: QKV projection, Q/K RMSNorm, RoPE, and KV-cache updates happen before
// this function. Output gating and the output projection happen afterward.
//
// Expected tensors:
//   query: [batch, 16, query_tokens, 256], BF16, CUDA
//          Projected, Q-normalized, and RoPE-applied.
//   key:   [batch, 4, key_tokens, 256], BF16, CUDA
//          Projected, K-normalized, RoPE-applied, and already combined with
//          cached keys when decoding.
//   value: [batch, 4, key_tokens, 256], BF16, CUDA
//          Projected and already combined with cached values when decoding.
//   attention_mask: additive and broadcastable over
//          [batch, 16, query_tokens, key_tokens]. Zero means visible; a large
//          negative value means masked. An empty tensor means no explicit mask.
//
// For this checkpoint, scale is 1 / sqrt(256) = 0.0625 and num_kv_groups is
// 16 query heads / 4 KV heads = 4. Query head h reads KV head
// h / num_kv_groups.
//
// Return a contiguous BF16 CUDA tensor shaped
// [batch, query_tokens, 16, 256].
//
// Keep torch::Tensor handling in this entrypoint: validate shape/dtype/device,
// allocate the output with query.options(), obtain the current CUDA stream, and
// populate a typed AttentionParams descriptor. The actual launcher and
// __global__ kernels should consume raw __nv_bfloat16 pointers, explicit
// dimensions/strides, scale, and cudaStream_t. This kernel does not own the KV
// cache; key and value already represent the complete visible history.
torch::Tensor attention_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value,
                             torch::Tensor attention_mask, double scale, int64_t num_kv_groups) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");

  TORCH_CHECK(query.scalar_type() == at::kBFloat16, "query must be bfloat16");

  TORCH_CHECK(key.scalar_type() == at::kBFloat16, "key must be bfloat16");

  TORCH_CHECK(value.scalar_type() == at::kBFloat16, "value must be bfloat16");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "query, key, and value must be rank-4 tensors");
  TORCH_CHECK(query.size(1) == QWEN35_QUERY_HEADS && query.size(3) == QWEN35_HEAD_DIM,
              "query must have shape [batch, 16, query_tokens, 256]");
  TORCH_CHECK(key.size(1) == QWEN35_KV_HEADS && key.size(3) == QWEN35_HEAD_DIM,
              "key must have shape [batch, 4, key_tokens, 256]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must have the same shape as key");
  TORCH_CHECK(query.size(0) == key.size(0), "query and key batch sizes must match");
  TORCH_CHECK(key.size(2) >= query.size(2), "key_tokens must be at least query_tokens");
  TORCH_CHECK(num_kv_groups == QWEN35_KV_GROUPS, "num_kv_groups must be 4");
  TORCH_CHECK(scale == QWEN35_ATTENTION_SCALE, "scale must be 0.0625");

  const auto *query_ptr = reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>());

  const auto *key_ptr = reinterpret_cast<const __nv_bfloat16 *>(key.data_ptr<at::BFloat16>());

  const auto *value_ptr = reinterpret_cast<const __nv_bfloat16 *>(value.data_ptr<at::BFloat16>());

  // ignore attention mask and assume autoregressive

  const int batch_size = static_cast<int>(query.size(0));
  const int query_tokens = static_cast<int>(query.size(2));
  const int key_tokens = static_cast<int>(key.size(2));
  auto output = torch::empty({batch_size, query_tokens, QWEN35_QUERY_HEADS, QWEN35_HEAD_DIM},
                             query.options());
  auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>());

  const dim3 grid(query_tokens, QWEN35_QUERY_HEADS, batch_size);
  const dim3 block(1, 1, 1);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  qwen35_attention_kernel<<<grid, block, 0, stream>>>(query_ptr, key_ptr, value_ptr, output_ptr,
                                                      batch_size, query_tokens, key_tokens);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Qwen3.5 attention kernel launch failed");

  return output;
}
