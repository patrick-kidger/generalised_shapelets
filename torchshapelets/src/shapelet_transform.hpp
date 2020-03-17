#ifndef TORCHSHAPELETS_SHAPELETTRANSFORM_HPP
#define TORCHSHAPELETS_SHAPELETTRANSFORM_HPP

#include <torch/extension.h>
#include <cstdint>    // int64_t


namespace torchshapelets {
    torch::Tensor shapelet_transform(torch::Tensor times, torch::Tensor path, torch::Tensor lengths,
                                     torch::Tensor shapelets, torch::Tensor max_length, int64_t num_samples,
                                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)>& discrepancy_fn,
                                     torch::Tensor discrepancy_arg);
}  // namespace torchshapelets

#endif //TORCHSHAPELETS_SHAPELETTRANSFORM_HPP