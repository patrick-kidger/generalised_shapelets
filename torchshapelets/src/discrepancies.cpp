#include <torch/extension.h>
#include <pybind11/functional.h>  // py::cpp_function
#include <cstdint>    // int64_t
#include <functional>  // std::function

#include "discrepancies.hpp"


namespace torchshapelets {
    torch::Tensor l2_discrepancy(torch::Tensor times, torch::Tensor path1, torch::Tensor path2, torch::Tensor linear) {
        // times has shape (length,)
        // path1 and path2 have shape (..., length, channels)
        // linear has shape (channels, channels), or should just be passed as an empty Tensor().

        auto path = path1 - path2;

        if (linear.ndimension() == 2) {
            path = torch::matmul(path, linear);
        }

        auto length = path.size(-2) - 1;

        auto next_path = path.narrow(/*dim=*/-2, /*start=*/1, /*length=*/length);
        auto prev_path = path.narrow(/*dim=*/-2, /*start=*/0, /*length=*/length);
        auto next_times = times.narrow(/*dim=*/0, /*start=*/1, /*length=*/length);
        auto prev_times = times.narrow(/*dim=*/0, /*start=*/0, /*length=*/length);

        auto first_term = (next_path - prev_path).pow(2).sum(/*dim=*/-1) / 3;  // sum down channel dim
        auto second_term = (next_path * prev_path).sum(/*dim=*/-1);  // sum down channel dim
        auto combine = (first_term + second_term) * (next_times - prev_times);
        auto out = combine.sum(/*dim=*/-1);  // sum down length dim
        out = out.relu();  // In principle floating point errors could result in something slightly negative here?
        return out.sqrt();
    }
}  // namespace torchshapelets