#include "attention_v2.cuh"

#include <cfloat>

#define QWEN35_QUERY_HEADS 16
#define QWEN35_KV_HEADS 4
#define QWEN35_HEAD_DIM 256
#define NUM_THREADS_WARP 32
#define QWEN35_KV_GROUPS (QWEN35_QUERY_HEADS / QWEN35_KV_HEADS)
#define QWEN35_ATTENTION_SCALE 0.0625F

// Version 2 workspace. Keep this symbol distinct so NCU can select it exactly.
//
// Suggested initial mapping:
//   grid  = (query_tokens, 16 query heads, batch)
//   block = (256 features)
//
// One block owns one output vector. The threads must cooperate to reduce each
// Q dot K score, update the online-softmax state, and accumulate V features.

__device__ int lane() {
  return threadIdx.x & 31;
}

__device__ int warp() {
  return threadIdx.x >> 5;
}

__global__ void qwen35_attention_v2_kernel(Qwen35AttentionParams params) {
  const int query_token = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;
  const int feature = threadIdx.x;

  if (batch >= params.batch_size || query_head >= QWEN35_QUERY_HEADS ||
      query_token >= params.query_tokens || feature >= QWEN35_HEAD_DIM) {
    return;
  }

  const int kv_head = query_head / QWEN35_KV_GROUPS;
  const int cached_tokens = params.key_tokens - params.query_tokens;
  const int visible_key_tokens = cached_tokens + query_token + 1;

  const int query_offset =
      ((batch * QWEN35_QUERY_HEADS + query_head) * params.query_tokens + query_token) *
      QWEN35_HEAD_DIM;

  float running_max = -FLT_MAX;
  float running_sum = 0.0F;

  float output_accumulator = 0.0F;

  for (int key_token = 0; key_token < visible_key_tokens; ++key_token) {
    const int kv_offset =
        ((batch * QWEN35_KV_HEADS + kv_head) * params.key_tokens + key_token) * QWEN35_HEAD_DIM;

    float partial = __bfloat162float(params.query[query_offset + feature]) *
                    __bfloat162float(params.key[kv_offset + feature]);

    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    __shared__ float warp_sums[QWEN35_HEAD_DIM / NUM_THREADS_WARP];
    __shared__ float shared_score;

    if (lane() == 0) {
      warp_sums[warp()] = partial;
    }
    __syncthreads();

    if (warp() == 0) {
      float sum = lane() < (QWEN35_HEAD_DIM / NUM_THREADS_WARP) ? warp_sums[lane()] : 0.0f;

      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }

      if (lane() == 0) {
        shared_score = sum * QWEN35_ATTENTION_SCALE;
      }
    }
    __syncthreads();

    float score = shared_score;

    // Online softmax merge. The previous numerator and denominator are in the
    // exp(score - running_max) coordinate system, so both must be rescaled if
    // this score establishes a new maximum.
    const float next_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - next_max);
    const float current_weight = expf(score - next_max);
    running_sum = running_sum * previous_scale + current_weight;

    output_accumulator = output_accumulator * previous_scale +
                         current_weight * __bfloat162float(params.value[kv_offset + feature]);
    running_max = next_max;
  }

  const int output_offset =
      ((batch * params.query_tokens + query_token) * QWEN35_QUERY_HEADS + query_head) *
      QWEN35_HEAD_DIM;
  params.output[output_offset + feature] = __float2bfloat16(output_accumulator / running_sum);
}

void launch_qwen35_attention_v2(const Qwen35AttentionParams &params, cudaStream_t stream) {
  const dim3 grid(params.query_tokens, QWEN35_QUERY_HEADS, params.batch_size);
  const dim3 block(QWEN35_HEAD_DIM);
  qwen35_attention_v2_kernel<<<grid, block, 0, stream>>>(params);
}
