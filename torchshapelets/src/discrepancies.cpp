#include <torch/extension.h>
#include <pybind11/functional.h>  // py::cpp_function
#include <cstdint>    // int64_t
#include <functional>  // std::function

#include "discrepancies.hpp"


namespace torchshapelets {
    namespace detail {
        torch::Tensor diff(torch::Tensor tensor) {
            return tensor.narrow(/*dim=*/0, /*start=*/1, /*length=*/tensor.size(0) - 1) -
                   tensor.narrow(/*dim=*/0, /*start=*/0, /*length=*/tensor.size(0) - 1);
        }
    }

    torch::Tensor l2_discrepancy(torch::Tensor times, torch::Tensor path1, torch::Tensor path2, torch::Tensor linear) {
        // times has shape (length,)
        // path1 and path2 have shape (..., length, channels)
        // linear has shape (channels, channels), or should just be passed as an empty Tensor().

        auto path = path1 - path2;

        if (linear.ndimension() == 2) {
            path = torch::matmul(path, linear);
        }

        auto times_diffs = detail::diff(times);
        auto times_squared = times * times;
        auto times_cubed = times_squared * times;
        auto times_squared_diffs = detail::diff(times_squared);
        auto times_cubed_diffs = detail::diff(times_cubed);
        auto path_diffs = path.narrow(/*dim=*/-2, /*start=*/1, /*length=*/path.size(-2) - 1) -
             path.narrow(/*dim=*/-2, /*start=*/0, /*length=*/path.size(-2) - 1);
        auto m = path_diffs / times_diffs.unsqueeze(-1);
        auto c = path.narrow(/*dim=*/-2, /*start=*/0, /*length=*/path.size(-2) - 1) -
                 m * times.narrow(/*dim=*/0, /*start=*/0, /*length=*/times.size(0) - 1).unsqueeze(-1);

        auto m_norm = m.norm(/*p=*/2, /*dims=*/{-1});
        auto m_dot_c = (m * c).sum(/*dims=*/{-1});
        auto c_norm = c.norm(/*p=*/2, /*dims=*/{-1});

        auto first_term = (m_norm * m_norm * times_cubed_diffs).sum(/*dims=*/{-1}) / 3;
        auto second_term = (m_dot_c * times_squared_diffs).sum(/*dims=*/{-1});
        auto third_term = (c_norm * c_norm * times_diffs).sum(/*dims=*/{-1});

        return first_term + second_term + third_term;
    }
}  // namespace torchshapelets