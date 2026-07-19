#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <tuple>

#define QWEN35_LINEAR_HEADS 32
#define QWEN35_LINEAR_KEY_DIM 128
#define QWEN35_LINEAR_VALUE_DIM 128
#define QWEN35_LINEAR_QK_SCALE 0.08838834764831845F

// One cached Qwen3.5 linear-attention decode step. All tensors are contiguous:
//
//   query: [batch, 1, 32, 128], BF16
//   key:   [batch, 1, 32, 128], BF16
//   value: [batch, 1, 32, 128], BF16
//   decay: [batch, 1, 32],      FP32; this is g, so use expf(g)
//   beta:  [batch, 1, 32],      BF16
//   state: [batch, 32, 128, 128], BF16, updated in place
//   output:[batch, 1, 32, 128], BF16
//
// Q and K must first be independently L2-normalized over their 128 features.
// For each batch item and head, the reference computation is:
//
//   q = l2norm(query) / sqrt(128)
//   k = l2norm(key)
//   S = exp(decay) * state
//   remembered[v] = sum_k S[k, v] * k[k]
//   delta[v] = beta * (value[v] - remembered[v])
//   S[k, v] += k[k] * delta[v]
//   output[v] = sum_k S[k, v] * q[k]
//
// Compute the updated state and output from FP32 intermediates, then round both
// to BF16. The output must use the unrounded updated state. Blocks cannot
// communicate through shared memory, so choose ownership such that every state
// element is written exactly once and both reductions are well-defined.
__global__ void qwen35_gated_delta_decode_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ key,
    const __nv_bfloat16 *__restrict__ value, const float *__restrict__ decay,
    const __nv_bfloat16 *__restrict__ beta, __nv_bfloat16 *__restrict__ recurrent_state,
    __nv_bfloat16 *__restrict__ output, int batch_size) {
    // TODO: implement the single-token recurrent gated-delta update.
}

std::tuple<torch::Tensor, torch::Tensor>
gated_delta_decode_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value,
                        torch::Tensor decay, torch::Tensor beta,
                        torch::Tensor recurrent_state) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
                "query, key, and value must be CUDA tensors");
    TORCH_CHECK(decay.is_cuda() && beta.is_cuda() && recurrent_state.is_cuda(),
                "decay, beta, and recurrent_state must be CUDA tensors");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16,
                "query and key must be bfloat16");
    TORCH_CHECK(value.scalar_type() == at::kBFloat16 && beta.scalar_type() == at::kBFloat16,
                "value and beta must be bfloat16");
    TORCH_CHECK(decay.scalar_type() == at::kFloat, "decay must be float32");
    TORCH_CHECK(recurrent_state.scalar_type() == at::kBFloat16,
                "recurrent_state must be bfloat16");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
                "query, key, and value must be contiguous");
    TORCH_CHECK(decay.is_contiguous() && beta.is_contiguous() && recurrent_state.is_contiguous(),
                "decay, beta, and recurrent_state must be contiguous");

    const int64_t batch = query.size(0);
    TORCH_CHECK(query.sizes() == torch::IntArrayRef({batch, 1, QWEN35_LINEAR_HEADS,
                                                     QWEN35_LINEAR_KEY_DIM}),
                "query must have shape [batch, 1, 32, 128]");
    TORCH_CHECK(key.sizes() == query.sizes(), "key must have the same shape as query");
    TORCH_CHECK(value.sizes() == torch::IntArrayRef({batch, 1, QWEN35_LINEAR_HEADS,
                                                     QWEN35_LINEAR_VALUE_DIM}),
                "value must have shape [batch, 1, 32, 128]");
    TORCH_CHECK(decay.sizes() == torch::IntArrayRef({batch, 1, QWEN35_LINEAR_HEADS}),
                "decay must have shape [batch, 1, 32]");
    TORCH_CHECK(beta.sizes() == decay.sizes(), "beta must have the same shape as decay");
    TORCH_CHECK(recurrent_state.sizes() ==
                    torch::IntArrayRef({batch, QWEN35_LINEAR_HEADS, QWEN35_LINEAR_KEY_DIM,
                                        QWEN35_LINEAR_VALUE_DIM}),
                "recurrent_state must have shape [batch, 32, 128, 128]");

    auto output = torch::empty({batch, 1, QWEN35_LINEAR_HEADS, QWEN35_LINEAR_VALUE_DIM},
                               query.options());
    TORCH_CHECK(false,
                "qwen35_gated_delta_decode_kernel is not implemented; implement and launch it in "
                "src/cuda_impl/csrc/linear_attention.cu");
    return {output, recurrent_state};
}
