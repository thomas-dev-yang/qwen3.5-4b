#include <torch/extension.h>

torch::Tensor attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor attention_mask,
    double scale,
    int64_t num_kv_groups) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(false, "Implement the Qwen3.5 attention kernel in attention.cu");
  return query;
}
