#ifndef TORCHSHAPELETS_DISCREPANCIES_HPP
#define TORCHSHAPELETS_DISCREPANCIES_HPP

#include <torch/extension.h>
#include <cstdint>    // int64_t


namespace torchshapelets {
    // Computes the L2 discrepancy between two paths.
    //
    // That is, let f be the unique continuous piecewise linear function such that
    // ```
    // f(times[i]) == path1[i] for all i
    // ```
    // and let g be the unique continuous piecewise linear function such that
    // ```
    // g(times[i]) == path2[i] for all i
    // ```
    // where `times` is a strictly increasing 1D tensor of shape (length,), and `path1` is a tensor of shape
    // (..., length, channels) and `path2` is a tensor of shape (length, channels). [So yes, `path2` has no batch
    // dimensions.]
    // Finally let `linear` either be a tensor of shape () or a tensor of shape (channels, channels). If the former then
    // let A = 1 (the scalar), if the latter let A_ij = linear[i, j] for all i, j (so A is now a matrix).
    //
    // Then this function computes
    // ```
    // sqrt( \int_{times[0]}^{times[-1]} || Af(t) - Ag(t) ||_2^2 dt )
    // ```
    // i.e. the L2 norm of Af - Ag.
    // Which will be a tensor of shape (...).
    torch::Tensor l2_discrepancy(torch::Tensor times, torch::Tensor path1, torch::Tensor path2, torch::Tensor linear);

    torch::Tensor l2_discrepancy_squared(torch::Tensor times, torch::Tensor path1, torch::Tensor path2,
                                         torch::Tensor linear);
}  // namespace torchshapelets

#endif //TORCHSHAPELETS_DISCREPANCIES_HPP