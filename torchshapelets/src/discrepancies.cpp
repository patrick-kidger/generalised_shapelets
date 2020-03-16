#include <torch/extension.h>
#include <pybind11/functional.h>  // py::cpp_function
#include <cstdint>    // int64_t
#include <functional>  // std::function

// TODO: remove pycapsule
#include "pycapsule.hpp"

namespace torchshapelets {
    namespace detail {
        torch::Tensor diff(torch::Tensor tensor) {
            return tensor.narrow(/*dim=*/0, /*start=*/1, /*length=*/tensor.size(0) - 1) -
                   tensor.narrow(/*dim=*/0, /*start=*/0, /*length=*/tensor.size(0) - 1);
        }

// TODO: remove
//        struct Partial {
//            Partial(py::cpp_function fn, py::object args) : fn{fn}, args{args} {};
//            py::cpp_function fn;
//            py::object args;
//            constexpr static auto capsule_name = "torchshapelets.Partial"
//        };
    }

//    py::object partial(py::cpp_function fn, py::object args) {
//        return wrap_capsule<detail::Partial>(fn, args);
//    }
//
//    std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor)> unpartial(py::object partial_capsule) {
//        detail::Partial* partial = unwrap_capsule<detail::Partial*>(partial_capsule);
//        return [] (torch::Tensor times, torch::Tensor path1, torch::Tensor path2) {
//            return fn(times, path1, path2, *args);
//        }
//    }

    torch::Tensor l2_discrepancy(torch::Tensor times, torch::Tensor path1, torch::Tensor path2) {
        // times has shape (length,)
        // path1 and path2 have shape (..., length, channels)
        auto path = path1 - path2;

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