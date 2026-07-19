#pragma once

#include "attention_v2.cuh"

// V6 keeps V5's head-major boundary but executes attention through a
// ThunderKittens load-compute-finish pipeline. A producer warpgroup streams
// double-buffered K/V tiles while two consumer warpgroups process Q heads that
// share the same KV head.
void launch_qwen35_attention_v6(const Qwen35AttentionParams &params, cudaStream_t stream);
