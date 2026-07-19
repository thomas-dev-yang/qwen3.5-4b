#include "attention_v6.cuh"
#include "kittens.cuh"
#include "prototype.cuh"

namespace qwen35_attention_v6 {

using namespace kittens;
using namespace kittens::prototype::lcf;

constexpr int query_heads = 16;
constexpr int kv_heads = 4;
constexpr int kv_groups = query_heads / kv_heads;
constexpr int head_dim = 256;
constexpr int query_tile_rows = 64;
constexpr int key_tile_rows = 64;
constexpr int consumer_warpgroups = 2;
constexpr int consumer_warps = consumer_warpgroups * WARPGROUP_WARPS;
constexpr int producer_warps = WARPGROUP_WARPS;
constexpr int input_stages = 2;

// The two consumer warpgroups own different Q heads at the same token
// positions. Pair boundaries stay within Qwen's groups of four Q heads, so both
// consumers read the same K/V head and reuse every staged K/V tile.
using query_tile = st_bf<query_tile_rows, head_dim>;
using key_tile = st_bf<key_tile_rows, head_dim>;
using value_tile = st_bf<key_tile_rows, head_dim>;
using output_tile = st_bf<query_tile_rows, head_dim>;

using query_global = gl<bf16, -1, query_heads, -1, head_dim, query_tile>;
using key_global = gl<bf16, -1, kv_heads, -1, head_dim, key_tile>;
using value_global = gl<bf16, -1, kv_heads, -1, head_dim, value_tile>;
using output_global = gl<bf16, -1, query_heads, -1, head_dim, output_tile>;

struct attention_v6_layout {
    struct globals {
        query_global query;
        key_global key;
        value_global value;
        output_global output;
        int query_tokens;
        int key_tokens;
    };

    // The LCF template allocates two copies of this block as its input ring.
    struct input_block {
        key_tile key;
        value_tile value;
    };

    // Q is invariant across the K/V loop and remains resident in shared memory.
    struct scratch_block {
        query_tile query[consumer_warpgroups];
    };

    // Input-ring storage is dead before finish(), so the template overlays this
    // output staging area onto the tail of the same dynamic shared-memory region.
    struct finish_block {
        output_tile output[consumer_warpgroups];
    };

    struct common_state {
        int batch;
        int query_tile_index;
        int query_head_base;
        int kv_head;
        int query_start;
    };

    struct consumer_state {
        rt_fl<16, key_tile_rows> scores;
        rt_bf<16, key_tile_rows> probabilities;
        rt_fl<16, head_dim> output_accumulator;
        col_vec<rt_fl<16, key_tile_rows>> running_max;
        col_vec<rt_fl<16, key_tile_rows>> running_sum;
        col_vec<rt_fl<16, key_tile_rows>> previous_max_scaled;
        col_vec<rt_fl<16, key_tile_rows>> next_max_scaled;
    };
};

static_assert(sizeof(attention_v6_layout::scratch_block) +
                  input_stages * sizeof(attention_v6_layout::input_block) <
              MAX_SHARED_MEMORY);

struct qwen35_attention_v6_kernel {
    static constexpr int NUM_CONSUMER_WARPS = consumer_warps;
    static constexpr int NUM_PRODUCER_WARPS = producer_warps;
    static constexpr int INPUT_PIPE_STAGES = input_stages;
    static constexpr int NUM_BLOCKS = 1;

    using layout = attention_v6_layout;

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        // This version uses one fixed task per CTA. The LCF task loop exits after
        // that task; persistence can be added independently once this pipeline
        // has correctness and profile data.
        if (args.task_iter != 0) {
            args.num_iters = -1;
            return;
        }

        args.common.batch = blockIdx.z;
        args.common.query_tile_index = blockIdx.x;
        args.common.query_head_base = blockIdx.y * consumer_warpgroups;
        args.common.kv_head = args.common.query_head_base / kv_groups;
        args.common.query_start = args.common.query_tile_index * query_tile_rows;

        // The highest valid query row determines the final causally visible key.
        // Avoid loading K/V tiles that are entirely in the future during prefill.
        const int query_end_unclamped = args.common.query_start + query_tile_rows;
        const int query_end = query_end_unclamped < args.globals.query_tokens
                                  ? query_end_unclamped
                                  : args.globals.query_tokens;
        const int cached_tokens = args.globals.key_tokens - args.globals.query_tokens;
        const int visible_keys = cached_tokens + query_end;
        args.num_iters = (visible_keys + key_tile_rows - 1) / key_tile_rows;
    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout>) {
            // Hopper can transfer the producer warpgroup's register allocation
            // to the two compute warpgroups, which carry long-lived FP32 state.
            warpgroup::producer_registers();
        }

        __device__ static inline void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == 0) {
                warp::tma::expect(args.inputs_arrived, args.input);
                warp::tma::load_async(args.input.key, args.globals.key,
                                      {args.common.batch, args.common.kv_head, args.iter, 0},
                                      args.inputs_arrived);
                warp::tma::load_async(args.input.value, args.globals.value,
                                      {args.common.batch, args.common.kv_head, args.iter, 0},
                                      args.inputs_arrived);
            } else if (laneid() == 0) {
                // The input semaphore expects one arrival from every producer
                // warp. Warp zero contributes the asynchronous TMA completion.
                arrive(args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<consumer_warpgroups>();

            const int consumer = warpgroup::groupid();
            const int query_head = args.common.query_head_base + consumer;
            warpgroup::load(args.scratch.query[consumer], args.globals.query,
                            {args.common.batch, query_head, args.common.query_tile_index, 0});

            warp::zero(args.state.output_accumulator);
            warp::neg_infty(args.state.running_max);
            warp::zero(args.state.running_sum);
            warpgroup::sync(consumer);
        }

        __device__ static inline void compute(consumer_compute_args<layout> args) {
            constexpr float softmax_temperature = 0.0625F * 1.4426950408889634F;

            const int consumer = warpgroup::groupid();
            const int key_start = args.iter * key_tile_rows;
            const int warp_query_start = args.common.query_start + warpgroup::warpid() * 16;
            const int cached_tokens = args.globals.key_tokens - args.globals.query_tokens;

            // [64, 256] @ [64, 256]^T -> [64, 64]. The producer may already be
            // loading the next ring stage while both consumers execute this MMA.
            warpgroup::mm_ABt(args.state.scores, args.scratch.query[consumer], args.input.key);
            warp::copy(args.state.previous_max_scaled, args.state.running_max);
            warp::mul(args.state.previous_max_scaled, args.state.previous_max_scaled,
                      softmax_temperature);
            warpgroup::mma_async_wait();

            const int causal_diagonal = cached_tokens + warp_query_start - key_start;
            warp::tril(args.state.scores, args.state.scores, causal_diagonal,
                       base_types::constants<float>::neg_infty());
            warp::right_fill(args.state.scores, args.state.scores,
                             args.globals.key_tokens - key_start,
                             base_types::constants<float>::neg_infty());

            // Online softmax state remains private to each consumer warpgroup
            // across every stage in the K/V stream.
            warp::row_max(args.state.running_max, args.state.scores, args.state.running_max);
            warp::mul(args.state.scores, args.state.scores, softmax_temperature);
            warp::mul(args.state.next_max_scaled, args.state.running_max, softmax_temperature);
            warp::sub_row(args.state.scores, args.state.scores, args.state.next_max_scaled);
            warp::exp2(args.state.scores, args.state.scores);
            warp::sub(args.state.previous_max_scaled, args.state.previous_max_scaled,
                      args.state.next_max_scaled);
            warp::exp2(args.state.previous_max_scaled, args.state.previous_max_scaled);
            warp::mul(args.state.running_sum, args.state.running_sum,
                      args.state.previous_max_scaled);
            warp::row_sum(args.state.running_sum, args.state.scores, args.state.running_sum);
            warp::mul_row(args.state.output_accumulator, args.state.output_accumulator,
                          args.state.previous_max_scaled);

            warp::copy(args.state.probabilities, args.state.scores);
            warpgroup::mma_AB(args.state.output_accumulator, args.state.probabilities,
                              args.input.value);
            warpgroup::mma_async_wait();

            // The producer cannot overwrite this ring stage until every warp in
            // both consumer warpgroups has finished reading K and V.
            if (laneid() == 0) {
                arrive(args.inputs_finished);
            }
        }

        __device__ static inline void finish(consumer_finish_args<layout> args) {
            const int consumer = warpgroup::groupid();
            const int query_head = args.common.query_head_base + consumer;

            warp::div_row(args.state.output_accumulator, args.state.output_accumulator,
                          args.state.running_sum);
            warpgroup::store(args.finish.output[consumer], args.state.output_accumulator);
            warpgroup::sync(consumer);

            if (warpgroup::warpid() == 0) {
                warp::tma::store_async(
                    args.globals.output, args.finish.output[consumer],
                    {args.common.batch, query_head, args.common.query_tile_index, 0});
            }
            warp::tma::store_async_read_wait();
            __syncwarp();
            if (laneid() == 0) {
                arrive(args.finish_finished);
            }
        }
    };
};

} // namespace qwen35_attention_v6

void launch_qwen35_attention_v6(const Qwen35AttentionParams &params, cudaStream_t stream) {
    using namespace kittens;
    using namespace qwen35_attention_v6;

    auto *query = reinterpret_cast<bf16 *>(const_cast<__nv_bfloat16 *>(params.query));
    auto *key = reinterpret_cast<bf16 *>(const_cast<__nv_bfloat16 *>(params.key));
    auto *value = reinterpret_cast<bf16 *>(const_cast<__nv_bfloat16 *>(params.value));
    auto *output = reinterpret_cast<bf16 *>(params.output);

    const attention_v6_layout::globals kernel_globals{
        query_global(query, params.batch_size, nullptr, params.query_tokens, nullptr),
        key_global(key, params.batch_size, nullptr, params.key_tokens, nullptr),
        value_global(value, params.batch_size, nullptr, params.key_tokens, nullptr),
        output_global(output, params.batch_size, nullptr, params.query_tokens, nullptr),
        params.query_tokens,
        params.key_tokens,
    };

    using kernel_template = qwen35_attention_v6_kernel;
    // Leave 2 KiB for the template's statically allocated semaphores and
    // alignment reservation. The finish tiles end exactly at this boundary.
    constexpr int shared_bytes = MAX_SHARED_MEMORY - 2048;
    constexpr int threads = kittens::prototype::detail::NUM_THREADS_v<kernel_template>;
    static_assert(threads == (consumer_warps + producer_warps) * WARP_THREADS);

    cudaFuncSetAttribute(kittens::prototype::lcf::kernel<kernel_template>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    const dim3 grid((params.query_tokens + query_tile_rows - 1) / query_tile_rows,
                    query_heads / consumer_warpgroups, params.batch_size);
    kittens::prototype::lcf::kernel<kernel_template>
        <<<grid, threads, shared_bytes, stream>>>(kernel_globals);
}
