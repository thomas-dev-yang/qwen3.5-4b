#pragma once

#include "attention_v2.cuh"

struct Qwen35AttentionV3Params {
  Qwen35AttentionParams attention;
  float *partials;
  int key_tiles;
};

void launch_qwen35_attention_v3(const Qwen35AttentionV3Params &params, cudaStream_t stream);
