#pragma once

#include "attention_v3.cuh"

void launch_qwen35_attention_v4(const Qwen35AttentionV3Params &params, cudaStream_t stream);
