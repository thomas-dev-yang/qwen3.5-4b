#pragma once

#include "attention_v2.cuh"

// V5 is the ThunderKittens implementation. The launcher consumes the usual
// head-major Q/K/V tensors. Its output pointer is an internal head-major
// [batch, 16, query_tokens, 256] temporary which attention.cu transposes into
// the public [batch, query_tokens, 16, 256] layout.
void launch_qwen35_attention_v5(const Qwen35AttentionParams &params, cudaStream_t stream);
