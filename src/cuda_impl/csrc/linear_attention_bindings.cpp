#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor>
gated_delta_decode_cuda(torch::Tensor query, torch::Tensor key, torch::Tensor value,
                        torch::Tensor decay, torch::Tensor beta,
                        torch::Tensor recurrent_state);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &gated_delta_decode_cuda,
               "Qwen3.5 single-token gated-delta update (CUDA)");
}
