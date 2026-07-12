#include <torch/extension.h>

torch::Tensor attention_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value,
                             torch::Tensor attention_mask, double scale, int64_t num_kv_groups,
                             int64_t version);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &attention_cuda, "Qwen3.5 full attention (CUDA)");
}
