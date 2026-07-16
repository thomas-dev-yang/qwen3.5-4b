#include "attention_v5.cuh"
#include "kittens.cuh"

#include <cstddef>

namespace qwen35_attention_v5 {

using namespace kittens;

// This kernel is specialized to Qwen3.5-4B rather than being a general MHA
// implementation. A block is exactly one Hopper warpgroup: four warps / 128
// threads. The warpgroup collaboratively computes 64 query rows for one Q head.
constexpr int query_heads = 16;
constexpr int kv_heads = 4;
constexpr int kv_groups = query_heads / kv_heads;
constexpr int head_dim = 256;
constexpr int query_tile_rows = 64;
constexpr int key_tile_rows = 128;
constexpr int threads = WARPGROUP_WARPS * WARP_THREADS;

// st_* means a block-scoped shared-memory tile. TK owns the physical swizzled
// layout required by WGMMA; its logical dimensions are still ordinary matrices:
//
//   query_tile:  64 Q rows x 256 features
//   key_tile:   128 K rows x 256 features
//   value_tile: 128 V rows x 256 features
//
// output_tile has the same storage size as query_tile, so the kernel reuses the
// Q allocation for output after Q is no longer needed.
using query_tile = st_bf<query_tile_rows, head_dim>;
using key_tile = st_bf<key_tile_rows, head_dim>;
using value_tile = st_bf<key_tile_rows, head_dim>;
using output_tile = st_bf<query_tile_rows, head_dim>;

// gl is a lightweight description of a contiguous global-memory tensor. Its
// dimensions are [batch, depth/head, row/token, column/feature]. A -1 is dynamic
// and is supplied by the launcher; positive dimensions are compile-time facts.
// These types do not allocate or copy memory.
using query_global = gl<bf16, -1, query_heads, -1, head_dim>;
using key_global = gl<bf16, -1, kv_heads, -1, head_dim>;
using value_global = gl<bf16, -1, kv_heads, -1, head_dim>;
using output_global = gl<bf16, -1, query_heads, -1, head_dim>;

// Passed by value as the kernel argument. Besides raw pointers and dimensions,
// each gl carries the strides TK needs for tile loads/stores.
struct globals {
    query_global query;
    key_global key;
    value_global value;
    output_global output;
    int query_tokens;
    int key_tokens;
};

// Dynamic shared memory contains one Q, K, and V tile. The extra 1024 bytes let
// tma_swizzle_allocator align the first tile even if the dynamic-memory base is
// not already at TK's required boundary. Hopper has enough shared memory for one
// such block, but this allocation prevents two blocks from residing on one SM.
constexpr std::size_t shared_bytes =
    sizeof(query_tile) + sizeof(key_tile) + sizeof(value_tile) + 1024;

static_assert(query_tile::rows == query_tile_rows && query_tile::cols == head_dim);
static_assert(key_tile::rows == key_tile_rows && key_tile::cols == head_dim);
static_assert(shared_bytes < MAX_SHARED_MEMORY);

__global__ __launch_bounds__(threads,
                             1) void qwen35_attention_v5_kernel(const __grid_constant__ globals g) {
    // exp2(x * log2(e)) equals exp(x). Folding the attention scale into this
    // constant computes exp(score / sqrt(256)) using TK's exp2 operation.
    constexpr float softmax_temperature = 0.0625F * 1.4426950408889634F;

    // "shared" is the untyped dynamic shared-memory region requested at launch.
    // The allocator places typed, correctly aligned/swizzled TK tiles inside it.
    extern __shared__ int shared[];
    tma_swizzle_allocator allocator(shared);
    query_tile &query_shared = allocator.allocate<query_tile>();
    key_tile &key_shared = allocator.allocate<key_tile>();
    value_tile &value_shared = allocator.allocate<value_tile>();
    output_tile &output_shared = reinterpret_cast<output_tile &>(query_shared);

    // Grid ownership:
    //   blockIdx.x: one group of 64 query tokens
    //   blockIdx.y: one of the 16 query heads
    //   blockIdx.z: one batch item
    const int query_tile_index = blockIdx.x;
    const int query_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_head = query_head / kv_groups;
    const int query_start = query_tile_index * query_tile_rows;

    // A 64-row WGMMA result is distributed across the four warps. Each warp's
    // register tile owns 16 consecutive query rows, hence this per-warp base.
    const int warp_query_start = query_start + warpgroup::warpid() * 16;

    // Q is the newest suffix of K/V. During decode this is key_tokens - 1;
    // during same-length prefill it is zero.
    const int cached_tokens = g.key_tokens - g.query_tokens;

    // All 128 threads cooperate on this global -> shared copy. TK writes the
    // swizzled representation consumed directly by WGMMA. Out-of-range rows in
    // the final partial Q tile are zero-filled and never stored back.
    warpgroup::load(query_shared, g.query, {batch, query_head, query_tile_index, 0});
    __syncthreads();

    // rt_* means a warp-scoped register tile. Every warp has its own instances:
    //
    //   scores:             its 16 Q rows x 128 current K rows
    //   probabilities:      BF16 copy of those softmax weights for tensor cores
    //   output_accumulator: its 16 Q rows x 256 output features, accumulated FP32
    //
    // The col_vec types hold one value per score row: max and denominator for
    // each of the warp's 16 independent softmaxes.
    rt_fl<16, key_tile_rows> scores;
    rt_bf<16, key_tile_rows> probabilities;
    rt_fl<16, head_dim> output_accumulator;
    col_vec<rt_fl<16, key_tile_rows>> running_max;
    col_vec<rt_fl<16, key_tile_rows>> running_sum;
    col_vec<rt_fl<16, key_tile_rows>> previous_max_scaled;
    col_vec<rt_fl<16, key_tile_rows>> next_max_scaled;

    warp::zero(output_accumulator);
    warp::neg_infty(running_max);
    warp::zero(running_sum);

    // K/V are streamed through the same shared-memory allocations 128 rows at a
    // time. output_accumulator, running_max, and running_sum survive the loop and
    // therefore represent the online-softmax state across all previous K tiles.
    const int key_tiles = (g.key_tokens + key_tile_rows - 1) / key_tile_rows;
    for (int key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
        const int key_start = key_tile_index * key_tile_rows;

        warpgroup::load(key_shared, g.key, {batch, kv_head, key_tile_index, 0});
        warpgroup::load(value_shared, g.value, {batch, kv_head, key_tile_index, 0});
        __syncthreads();

        // Warpgroup scope: all four warps issue this together. Logically it is
        // [64, 256] @ [128, 256]^T -> [64, 128]. TK emits a sequence of Hopper
        // WGMMA instructions and distributes 16 result rows to each warp.
        warpgroup::mm_ABt(scores, query_shared, key_shared);

        // Save the old maximum before incorporating the current score tile. It
        // will become the rescaling factor for the old denominator and output.
        warp::copy(previous_max_scaled, running_max);
        warp::mul(previous_max_scaled, previous_max_scaled, softmax_temperature);
        warpgroup::mma_async_wait();

        // Warp scope: each warp masks its own 16x128 score tile. In global
        // coordinates, causal visibility is:
        //
        //   key_position <= cached_tokens + query_position
        //
        // Converting both sides to this tile's local row/column coordinates gives
        // the diagonal below. This is why one formula handles prefill and decode.
        const int causal_diagonal = cached_tokens + warp_query_start - key_start;
        warp::tril(scores, scores, causal_diagonal, base_types::constants<float>::neg_infty());

        // Mask zero-filled K rows in the final partial tile. Without this, their
        // zero dot products would receive nonzero softmax probability.
        warp::right_fill(scores, scores, g.key_tokens - key_start,
                         base_types::constants<float>::neg_infty());

        // Warp-local online softmax, independently for each of the 16 rows.
        // If m_old/l_old/o_old describe previous K tiles and m_new is the max
        // after adding this tile, previous_max_scaled becomes:
        //
        //   exp((m_old - m_new) / sqrt(256))
        //
        // That factor moves the old denominator and numerator into the new max's
        // coordinate system before adding the current tile. Scores and maxima are
        // kept unscaled until immediately before exp2.
        warp::row_max(running_max, scores, running_max);
        warp::mul(scores, scores, softmax_temperature);
        warp::mul(next_max_scaled, running_max, softmax_temperature);
        warp::sub_row(scores, scores, next_max_scaled);
        warp::exp2(scores, scores);
        warp::sub(previous_max_scaled, previous_max_scaled, next_max_scaled);
        warp::exp2(previous_max_scaled, previous_max_scaled);
        warp::mul(running_sum, running_sum, previous_max_scaled);
        warp::row_sum(running_sum, scores, running_sum);
        warp::mul_row(output_accumulator, output_accumulator, previous_max_scaled);

        // Warpgroup scope again. BF16 probabilities are required as the tensor-
        // core input, while output_accumulator remains FP32:
        // [64, 128] @ [128, 256] -> [64, 256].
        warp::copy(probabilities, scores);
        warpgroup::mma_AB(output_accumulator, probabilities, value_shared);
        warpgroup::mma_async_wait();
        __syncthreads();
    }

    // Convert the online numerator into the final attention output. The four
    // warps then cooperatively materialize their register fragments as one
    // swizzled 64x256 shared tile and copy it back to global memory.
    warp::div_row(output_accumulator, output_accumulator, running_sum);
    warpgroup::store(output_shared, output_accumulator);
    __syncthreads();
    warpgroup::store(g.output, output_shared, {batch, query_head, query_tile_index, 0});
}

} // namespace qwen35_attention_v5

void launch_qwen35_attention_v5(const Qwen35AttentionParams &params, cudaStream_t stream) {
    using namespace qwen35_attention_v5;

    // TK's global-layout type is mutable because it also supports stores. Q/K/V
    // are still only read by this kernel; the const_cast adapts the pointer type
    // and does not authorize an actual write in our code.
    auto *query = reinterpret_cast<bf16 *>(const_cast<__nv_bfloat16 *>(params.query));
    auto *key = reinterpret_cast<bf16 *>(const_cast<__nv_bfloat16 *>(params.key));
    auto *value = reinterpret_cast<bf16 *>(const_cast<__nv_bfloat16 *>(params.value));
    auto *output = reinterpret_cast<bf16 *>(params.output);

    // Bind runtime pointers and dynamic batch/token dimensions to the compile-
    // time gl layouts. params.output is the internal head-major temporary
    // described in attention_v5.cuh, not yet the public output layout.
    const globals kernel_globals{
        query_global(query, params.batch_size, nullptr, params.query_tokens, nullptr),
        key_global(key, params.batch_size, nullptr, params.key_tokens, nullptr),
        value_global(value, params.batch_size, nullptr, params.key_tokens, nullptr),
        output_global(output, params.batch_size, nullptr, params.query_tokens, nullptr),
        params.query_tokens,
        params.key_tokens,
    };

    // Blocks using more than CUDA's default shared-memory allowance must opt in.
    // The launch creates one block for every (64-query tile, Q head, batch) tuple
    // on PyTorch's current CUDA stream.
    cudaFuncSetAttribute(qwen35_attention_v5_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));
    const dim3 grid((params.query_tokens + query_tile_rows - 1) / query_tile_rows, query_heads,
                    params.batch_size);
    qwen35_attention_v5_kernel<<<grid, threads, shared_bytes, stream>>>(kernel_globals);
}
