#ifndef TORCHSHAPELETS_DISCREPANCIES_HPP
#define TORCHSHAPELETS_DISCREPANCIES_HPP

#include <torch/extension.h>
#include <cstdint>    // int64_t


namespace torchshapelets {
    torch::Tensor l2_discrepancy(torch::Tensor times, torch::Tensor path1, torch::Tensor path2, torch::Tensor linear);
}  // namespace torchshapelets

#endif //TORCHSHAPELETS_DISCREPANCIES_HPP