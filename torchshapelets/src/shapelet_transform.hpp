#ifndef TORCHSHAPELETS_SHAPELETTRANSFORM_HPP
#define TORCHSHAPELETS_SHAPELETTRANSFORM_HPP

#include <torch/extension.h>
#include <cstdint>    // int64_t


namespace torchshapelets {
    // Adds knots to a piecewise linear path, such that it doesn't change the function it represents.
    //
    // Arguments:
    //     times, path: As `restriction`.
    //     additional_times: Additional times to add knots to by linearly interpolating between the values already
    //         given.
    //     keep_original_times: A boolean, specifying whether to return just the knots associated with
    //         additional_times, or the knots associated with both times and additional_times.
    //
    // Returns:
    //     If keep_original_times is true then let r_length = times.size(0) + additional_times.size(0). If
    //     keep_original_times is false then let r_lenth = additional_times.size(0).
    //     What is then returned is a 2-tuple of tensors. The first corresponds to times, and will be of shape
    //     (r_length,). The second corresponds to the path, and will be of shape
    //     (..., r_length, path.size(-1)).
    //     Note that if keep_original_times is true and additional_times and times have nontrivial intersection then
    //     these extra points will be removed, and r_length will actually be slightly smaller as a result.
    //
    // Warning:
    //     This function does no input validation. (This is deliberate, as it's used in a moderately-hot loop elsewhere)
    std::tuple<torch::Tensor, torch::Tensor>
    unsafe_add_knots(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> times,
                     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> path,
                     torch::Tensor additional_times, bool keep_original_times);

    // Checks that times, path, lengths, max_length are a valid combination of inputs.
    void check_inputs(torch::Tensor times, torch::Tensor path, torch::Tensor lengths, torch::Tensor max_length);

    // Differentiably computes the generalised shapelet transform. See
    // shapelet_transform.py::GeneralisedShapeletTransform for documentation.
    torch::Tensor shapelet_transform(torch::Tensor times, torch::Tensor path, torch::Tensor lengths,
                                     torch::Tensor shapelets, torch::Tensor max_length, const int64_t num_samples,
                                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)>& discrepancy_fn,
                                     torch::Tensor discrepancy_arg);
}  // namespace torchshapelets

#endif //TORCHSHAPELETS_SHAPELETTRANSFORM_HPP