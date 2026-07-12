#include "attention_v4.cuh"

#include <cfloat>

#define QWEN35_QUERY_HEADS 16
#define QWEN35_KV_HEADS 4
#define QWEN35_HEAD_DIM 256
#define QWEN35_KV_GROUPS (QWEN35_QUERY_HEADS / QWEN35_KV_HEADS)
#define QWEN35_ATTENTION_SCALE 0.0625F
#define QWEN35_V4_KEY_TILE 128
#define QWEN35_V4_KEYS_PER_GROUP 8
#define QWEN35_V4_PARTIAL_VALUES (QWEN35_HEAD_DIM + 2)

namespace {

__device__ __forceinline__ int lane() {
    return threadIdx.x & 31;
}

__device__ __forceinline__ int warp() {
    return threadIdx.x >> 5;
}

__device__ __forceinline__ int partial_offset(const Qwen35AttentionV3Params &params, int batch,
                                              int query_token, int query_head, int key_tile) {
    return (
        (((batch * params.attention.query_tokens + query_token) * QWEN35_QUERY_HEADS + query_head) *
             params.key_tiles +
         key_tile) *
        QWEN35_V4_PARTIAL_VALUES);
}

} // namespace

// One block owns one Q row and one contiguous key tile. During each group:
//   warp 0 computes Q dot K[group + 0]
//   ...
//   warp 7 computes Q dot K[group + 7]
// The block then forms eight softmax weights, and thread d updates output
// numerator feature d from the same eight V rows.
__global__ void qwen35_attention_v4_partial_kernel(Qwen35AttentionV3Params params) {
    const int key_tile = blockIdx.x;
    const int query_token = blockIdx.y;
    const int query_head = blockIdx.z % QWEN35_QUERY_HEADS;
    const int batch = blockIdx.z / QWEN35_QUERY_HEADS;
    const int feature = threadIdx.x;

    if (batch >= params.attention.batch_size || query_token >= params.attention.query_tokens ||
        key_tile >= params.key_tiles || feature >= QWEN35_HEAD_DIM) {
        return;
    }

    const int cached_tokens = params.attention.key_tokens - params.attention.query_tokens;
    const int visible_key_tokens = cached_tokens + query_token + 1;
    const int key_start = key_tile * QWEN35_V4_KEY_TILE;
    const int key_end = min(key_start + QWEN35_V4_KEY_TILE, visible_key_tokens);
    const int kv_head = query_head / QWEN35_KV_GROUPS;
    const int query_offset =
        ((batch * QWEN35_QUERY_HEADS + query_head) * params.attention.query_tokens + query_token) *
        QWEN35_HEAD_DIM;

    __shared__ float scores[QWEN35_V4_KEYS_PER_GROUP];
    __shared__ float weights[QWEN35_V4_KEYS_PER_GROUP];
    __shared__ float group_max;
    __shared__ float group_sum;

    float running_max = -FLT_MAX;
    float running_sum = 0.0F;
    float output_accumulator = 0.0F;

    for (int group_start = key_start; group_start < key_end;
         group_start += QWEN35_V4_KEYS_PER_GROUP) {
        const int warp_key = group_start + warp();
        const bool warp_key_is_visible = warp_key < key_end;

        // TODO phase 1: each warp computes one complete Q dot K score. Each
        // lane should process features lane, lane + 32, ..., lane + 224.
        // Invalid tail-group warps must publish a score of -infinity.
        (void)warp_key;
        (void)warp_key_is_visible;
        (void)kv_head;
        (void)query_offset;

        // TODO phase 2: use scores[0..7] to compute group_max, group_sum, and
        // weights[0..7]. All threads need the resulting scalar/vector state.

        // TODO phase 3: thread `feature` computes its weighted sum across the
        // eight V rows, then merges that group state into running_max,
        // running_sum, and output_accumulator.
    }

    const int workspace_offset = partial_offset(params, batch, query_token, query_head, key_tile);
    params.partials[workspace_offset + 2 + feature] = output_accumulator;
    if (feature == 0) {
        params.partials[workspace_offset] = running_max;
        params.partials[workspace_offset + 1] = running_sum;
    }

    (void)scores;
    (void)weights;
    (void)group_max;
    (void)group_sum;
}

__global__ void qwen35_attention_v4_merge_kernel(Qwen35AttentionV3Params params) {
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
        const int workspace_offset =
            partial_offset(params, batch, query_token, query_head, key_tile);
        const float tile_max = params.partials[workspace_offset];
        const float tile_sum = params.partials[workspace_offset + 1];
        const float tile_numerator = params.partials[workspace_offset + 2 + feature];
        const float next_max = fmaxf(merged_max, tile_max);
        const float merged_scale = expf(merged_max - next_max);
        const float tile_scale = expf(tile_max - next_max);

        merged_sum = merged_sum * merged_scale + tile_sum * tile_scale;
        merged_numerator = merged_numerator * merged_scale + tile_numerator * tile_scale;
        merged_max = next_max;
    }

    const int output_offset =
        ((batch * params.attention.query_tokens + query_token) * QWEN35_QUERY_HEADS + query_head) *
        QWEN35_HEAD_DIM;
    params.attention.output[output_offset + feature] =
        __float2bfloat16(merged_numerator / merged_sum);
}

void launch_qwen35_attention_v4(const Qwen35AttentionV3Params &params, cudaStream_t stream) {
    const dim3 partial_grid(params.key_tiles, params.attention.query_tokens,
                            params.attention.batch_size * QWEN35_QUERY_HEADS);
    const dim3 merge_grid(params.attention.query_tokens, QWEN35_QUERY_HEADS,
                          params.attention.batch_size);
    const dim3 block(QWEN35_HEAD_DIM);

    qwen35_attention_v4_partial_kernel<<<partial_grid, block, 0, stream>>>(params);
    qwen35_attention_v4_merge_kernel<<<merge_grid, block, 0, stream>>>(params);
}
