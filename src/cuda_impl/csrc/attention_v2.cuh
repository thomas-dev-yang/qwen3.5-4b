#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

struct Qwen35AttentionParams {
  const __nv_bfloat16 *query;
  const __nv_bfloat16 *key;
  const __nv_bfloat16 *value;
  __nv_bfloat16 *output;
  int batch_size;
  int query_tokens;
  int key_tokens;
};

void launch_qwen35_attention_v2(const Qwen35AttentionParams &params, cudaStream_t stream);
