#include "attention_v3.cuh"

#include <cfloat>

#define QWEN35_QUERY_HEADS 16
#define QWEN35_KV_HEADS 4
#define QWEN35_HEAD_DIM 256
#define QWEN35_V3_KEY_TILE 128
#define QWEN35_KV_GROUPS 4
#define QWEN35_V3_PARTIAL_VALUES (QWEN35_HEAD_DIM + 2)
#define NUM_THREADS_WARP 32
#define QWEN35_ATTENTION_SCALE 0.0625F

namespace {

__device__ __forceinline__ int lane() {
    return threadIdx.x & 31;
}

__device__ __forceinline__ int warp() {
    return threadIdx.x >> 5;
}

} // namespace

// One block owns one (batch, query head, query token, key tile) partial.
// Each partial stores:
//   [0]       tile maximum
//   [1]       tile softmax denominator
//   [2..257]  unnormalized output numerator
__global__ void qwen35_attention_v3_partial_kernel(Qwen35AttentionV3Params params) {
    const int key_tile = blockIdx.x;
    const int query_token = blockIdx.y;
    const int query_head = blockIdx.z % QWEN35_QUERY_HEADS;
    const int batch = blockIdx.z / QWEN35_QUERY_HEADS;
    const int feature = threadIdx.x;

    if (batch >= params.attention.batch_size || query_token >= params.attention.query_tokens ||
        key_tile >= params.key_tiles || feature >= QWEN35_HEAD_DIM) {
        return;
    }

    // TODO: compute one online-softmax partial over this key tile and write it
    // to params.partials.
    const int cached_tokens = params.attention.key_tokens - params.attention.query_tokens;
    const int visible_key_tokens = cached_tokens + query_token + 1;
    const int key_start = key_tile * QWEN35_V3_KEY_TILE;
    const int key_end = min(key_start + QWEN35_V3_KEY_TILE, visible_key_tokens);
    const int kv_head = query_head / QWEN35_KV_GROUPS;
    const int query_offset =
        ((batch * QWEN35_QUERY_HEADS + query_head) * params.attention.query_tokens + query_token) *
        QWEN35_HEAD_DIM;
    float running_max = -FLT_MAX;
    float running_sum = 0.0F;
    float output_accumulator = 0.0F;
    const int partial_offset =
        ((((batch * params.attention.query_tokens + query_token) * QWEN35_QUERY_HEADS +
           query_head) *
              params.key_tiles +
          key_tile) *
         QWEN35_V3_PARTIAL_VALUES);

    for (int key_token = key_start; key_token < key_end; ++key_token) {
        const int kv_offset =
            ((batch * QWEN35_KV_HEADS + kv_head) * params.attention.key_tokens + key_token) *
            QWEN35_HEAD_DIM;

        float partial = __bfloat162float(params.attention.query[query_offset + feature]) *
                        __bfloat162float(params.attention.key[kv_offset + feature]);

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

        output_accumulator =
            output_accumulator * previous_scale +
            current_weight * __bfloat162float(params.attention.value[kv_offset + feature]);
        running_max = next_max;
    }

    params.partials[partial_offset + 2 + feature] = output_accumulator;

    if (feature == 0) {
        params.partials[partial_offset] = running_max;
        params.partials[partial_offset + 1] = running_sum;
    }
}

// One block owns one final (batch, query head, query token) output vector and
// merges all key-tile partials with the online-softmax merge operation.
__global__ void qwen35_attention_v3_merge_kernel(Qwen35AttentionV3Params params) {
    const int query_token = blockIdx.x;
    const int query_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int feature = threadIdx.x;

    if (batch >= params.attention.batch_size || query_token >= params.attention.query_tokens ||
        query_head >= QWEN35_QUERY_HEADS || feature >= QWEN35_HEAD_DIM) {
        return;
    }

    float merged_max = -FLT_MAX;
    float merged_sum = 0.0F;
    float merged_numerator = 0.0F;

    for (int key_tile = 0; key_tile < params.key_tiles; ++key_tile) {
        const int partial_offset =
            ((((batch * params.attention.query_tokens + query_token) * QWEN35_QUERY_HEADS +
               query_head) *
                  params.key_tiles +
              key_tile) *
             QWEN35_V3_PARTIAL_VALUES);
        const float partial_max = params.partials[partial_offset];
        const float partial_sum = params.partials[partial_offset + 1];
        const float partial_numerator = params.partials[partial_offset + 2 + feature];

        const float next_max = fmaxf(merged_max, partial_max);
        const float merged_scale = expf(merged_max - next_max);
        const float partial_scale = expf(partial_max - next_max);

        merged_sum = merged_sum * merged_scale + partial_sum * partial_scale;
        merged_numerator = merged_numerator * merged_scale + partial_numerator * partial_scale;
        merged_max = next_max;
    }

    const int output_offset =
        ((batch * params.attention.query_tokens + query_token) * QWEN35_QUERY_HEADS + query_head) *
        QWEN35_HEAD_DIM;
    params.attention.output[output_offset + feature] =
        __float2bfloat16(merged_numerator / merged_sum);
}

void launch_qwen35_attention_v3(const Qwen35AttentionV3Params &params, cudaStream_t stream) {
    const dim3 partial_grid(params.key_tiles, params.attention.query_tokens,
                            params.attention.batch_size * QWEN35_QUERY_HEADS);
    const dim3 merge_grid(params.attention.query_tokens, QWEN35_QUERY_HEADS,
                          params.attention.batch_size);
    const dim3 block(QWEN35_HEAD_DIM);

    qwen35_attention_v3_partial_kernel<<<partial_grid, block, 0, stream>>>(params);
    qwen35_attention_v3_merge_kernel<<<merge_grid, block, 0, stream>>>(params);
}
