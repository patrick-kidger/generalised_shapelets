#include <torch/extension.h>
#include <cstdint>    // int64_t
#include <functional>  // std::function
#include <omp.h>
#include <stdexcept>  // std::invalid_argument
#include <tuple>      // std::get, std::tie, std::tuple

#define TORCHSHAPELETS_HAVE_DETAIL_ASSERTS true


namespace torchshapelets {
    namespace detail {
        struct omp_nested {
            omp_nested() :
            was_omp_max_active_levels(omp_get_max_active_levels()), was_omp_in_parallel(omp_in_parallel())
            {
                if (!was_omp_in_parallel) {
                    omp_set_max_active_levels(2);
                }
            }
            ~omp_nested() {
                if (!was_omp_in_parallel) {
                    omp_set_max_active_levels(was_omp_max_active_levels);
                }
            }
        private:
            int was_omp_max_active_levels;
            int was_omp_in_parallel;
        };

        void assert_increasing(torch::Tensor sequence) {
            auto prev_time = sequence[0];
            for (int64_t i = 1; i < sequence.size(0); ++i) {
                if ((sequence[i] < prev_time).item().to<bool>()) {
                    throw std::invalid_argument("Not an increasing sequence.");
                }
                prev_time = sequence[i];
            }
        }

        torch::Tensor len_index(torch::Tensor input, int64_t index) {
            return input.narrow(/*dim=*/-2, /*start=*/index, /*length=*/1).squeeze(/*dim=*/-2);
        }

        // Differentiably restricts a piecewise linear path.
        //
        // The arguments `times` and `path` between them define a piecewise linear function f, with
        // f(times[i]) = path[..., i, :] for each i, and affine on the pieces in between.
        //
        // The return value is this piecewise linear function restricted to the interval [start, end], represented in
        // the same way by a pair of (times, path).
        //
        // Arguments:
        //     times: A 1D tensor of shape (length,), describing the times of each knot.
        //     path: A tensor of shape (..., length, channels), describing the values of each knot.
        //     start: The start time to restrict to. A scalar.
        //     end: The end time to restrict to. A scalar.
        //
        // Returns:
        //     A 2-tuple of ((start_time, middle_time, end_time), (start_path, middle_path, end_path)), where
        //     torch.cat([start_time.unsqueeze(0), middle_time, end_time.unsqueeze(0)], dim=0) is a tensor of shape
        //     (r_length,) and torch.cat([start_path.unsqueeze(-2), middle_path, end_path.unsqueeze(-2)], dim=-2) is a
        //     tensor of shape (..., r_length, channels).
        //     Furthermore start_time is of shape (), end_time is of shape (), start_path is of shape (..., channels)
        //     and end_path is of shape (..., channels).
        //     Really the thing we want is actually those concatenations, but what we do later on with this function,
        //     downstream, is just to iterate over it - which it's possible to do (with slightly faffier code) without
        //     the overhead of the torch.cat. Hence we leave it split apart like this for efficiency's sake.
        std::tuple<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>,
                   std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
        restriction(torch::Tensor times, torch::Tensor path, torch::Tensor start, torch::Tensor end) {
            if (TORCHSHAPELETS_HAVE_DETAIL_ASSERTS) {
                if (times.ndimension() != 1) {
                throw std::invalid_argument("times must be a 1D tensor of shape (length,).");
                }
                if (path.ndimension() < 2) {
                    throw std::invalid_argument("path must be a tensor of shape(..., length, channels).");
                }
                if (start.ndimension() != 0) {
                    throw std::invalid_argument("start must be a scalar.");
                }
                if (end.ndimension() != 0) {
                    throw std::invalid_argument("end must be a scalar.");
                }
                if (path.size(-2) != times.size(0)) {
                    throw std::invalid_argument("times and path must have the same size length dimension.");
                }
                if (times.size(0) < 2) {
                    throw std::invalid_argument("Length dimension must be of size at least 2 to define a path.");
                }
                assert_increasing(times);
                if ((start < times[0]).item().to<bool>()) {
                    throw std::invalid_argument("start must be within the interval specified by times.");
                }
                if ((end > times[-1]).item().to<bool>()) {
                    throw std::invalid_argument("end must be within the interval specified by times.");
                }
                if ((start >= end).item().to<bool>()) {
                    throw std::invalid_argument("start must be less than end.");
                }
            }

            // >= for start and > for end is deliberate
            // this correctly handles the cases that start == times[i] or end == times[i] for some i.
            // TODO: Switch this from an iterative search to binary search
            auto start_index = ((start >= times).sum() - 1).item().to<int64_t>();
            auto end_index = ((end > times).sum() - 1).item().to<int64_t>();

            auto before_start_time = times[start_index];
            auto after_start_time = times[start_index + 1];
            auto start_ratio = (start - before_start_time) / (after_start_time - before_start_time);
            auto start_diff = len_index(path, start_index + 1) - len_index(path, start_index);
            auto start_restriction = len_index(path, start_index) + start_ratio * start_diff;

            auto middle_restriction = path.narrow(/*dim=*/-2,
                                                  /*start=*/start_index + 1,
                                                  /*length=*/end_index - start_index);

            auto before_end_time = times[end_index];
            auto after_end_time = times[end_index + 1];
            auto end_ratio = (end - before_end_time) / (after_end_time - before_end_time);
            auto end_diff = len_index(path, end_index + 1) - len_index(path, end_index);
            auto end_restriction = len_index(path, end_index) + end_ratio * end_diff;

            auto middle_times = times.narrow(/*dim=*/0, /*start=*/start_index + 1, /*end=*/end_index - start_index);

            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> restricted_times = {start,
                                                                                        middle_times,
                                                                                        end};
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> restricted_path = {start_restriction,
                                                                                       middle_restriction,
                                                                                       end_restriction};
            // I think all this explicit type specification is needed to work around a bug in MSVC? Something like that.
            // I might be wrong. It might all be completely unnecessary.
            return std::tuple<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>,
                              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> {restricted_times,
                                                                                        restricted_path};
        }

        // Differentiably calculates
        // ```
        // g(start, end) = \min_{s \in [start, end]} fn(s)
        // ```
        // In particular this is differentiable with respect to the endpoints.

        // In practice this is done by sampling many points in the region [start, end] and calculating the minimum over
        // all of them. (And handling 'start' and 'end' specially so that they may be differentiated though.) The
        // maximum tolerance for the gap between points is controlled by `max_sampling_gap`.

        // Arguments:
        // start: A scalar.
        // end: A scalar.
        // fn: A function which consumes a scalar and returns an any-dimensional tensor.
        // num_samples: We compute a minimum over s in {start, start + 1/num_samples, start + 2/num_samples, ...,
        //     end - 1/num_samples, end}.

        // Returns:
        //      A PyTorch tensor of the same shape as `fn` returns.
        //
        //
        // templates are the best way to get a zero-cost abstraction (at least according to:)
        // https://vittorioromeo.info/index/blog/passing_functions_to_functions.html
        template <typename fn_type>
        torch::Tensor continuous_min(torch::Tensor start, torch::Tensor end, fn_type fn, torch::TensorOptions device,
                                     int64_t num_samples) {
            if (TORCHSHAPELETS_HAVE_DETAIL_ASSERTS) {
                if (start.ndimension() != 0) {
                throw std::invalid_argument("start must be a scalar.");
                }
                if (end.ndimension() != 0) {
                    throw std::invalid_argument("end must be a scalar.");
                }
                if ((start >= end).item().to<bool>()) {
                    throw std::invalid_argument("start must be less than end.");
                }
                if (num_samples < 1) {
                    throw std::invalid_argument("num_samples must be at least 2.");
                }
            }

            auto start_detached = start.detach().item();
            auto end_detached = end.detach().item();

            auto points = torch::linspace(start_detached, end_detached, num_samples, device);

            // We compute the minimum over the middle region separately, as there's good odds that this bit won't
            // require gradients. If I understand PyTorch's autograd well enough (and this really might be wrong on my
            // part), then this saves the creation of a zero-initialised tensor for each one, during backpropagation.
            std::vector<torch::Tensor> middle_results (num_samples - 2);
            #pragma omp parallel for default(none) shared(middle_results, points, fn, num_samples)
            for (int64_t point_index = 1; point_index < num_samples - 1; ++point_index) {
                middle_results[point_index - 1] = fn(points[point_index]);
            }
            // std::get<0> to get the minimum values, not their indices.
            auto middle_min = std::get<0>(torch::stack(middle_results, /*dim=*/0).min(/*dim=*/0));

            // This allows us to differentiate through the endpoints
            // If we wanted we could also construct every point in points[1:,-1] differentiably, but we're aiming for a
            // continuous minimum here, and that's just an implementation detail. (Basically it depends on whether you
            // use a discretise-then-optimise or optimise-then-discretise approach.)

            return std::get<0>(torch::stack({fn(start), middle_min, fn(end)}, /*dim=*/0).min(/*dim=*/0));
        }

        // Adds knots to a piecewise linear path, such that it doesn't change the function it represents.
        //
        // Arguments:
        //     times, path: As `restriction`.
        //     additional_times: Additional times to add knots to by linearly interpolating between the values already
        //         given.
        //
        // Returns:
        //     A 2-tuple of tensors. The first corresponds to times, and will be of shape
        //     (times.size(0) + additional_times.size(0),). The second corresponds to the path, and will be of shape
        //     (..., times.size(0) + additional_times.size(0), path.size(-1)).
        //     Note that if additional_times and times have nontrivial intersection then this extra points will be
        //     removed, and the returned tensors will actually be slightly smaller as a result.
        std::tuple<torch::Tensor, torch::Tensor>
        add_knots(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> times,
                  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> path,
                  torch::Tensor additional_times) {
            torch::Tensor start_times, middle_times, end_times;  // written during the COVID-19 outbreak
            torch::Tensor start_path, middle_path, end_path;
            std::tie(start_times, middle_times, end_times) = times;
            std::tie(start_path, middle_path, end_path) = path;

            if (TORCHSHAPELETS_HAVE_DETAIL_ASSERTS) {
                if (start_times.ndimension() != 0) {
                throw std::invalid_argument("times[0] must be zero-dimensional.");
                }
                if (middle_times.ndimension() != 1) {
                    throw std::invalid_argument("times[1] must be one-dimensional.");
                }
                if (end_times.ndimension() != 0) {
                    throw std::invalid_argument("times[2] must be zero-dimensional.");
                }
                if (start_path.ndimension() < 1) {
                    throw std::invalid_argument("path[0] must be at least one-dimensional.");
                }
                if (middle_path.ndimension() < 2) {
                    throw std::invalid_argument("path[1] must be at least two-dimensional.");
                }
                if (end_path.ndimension() < 1) {
                    throw std::invalid_argument("path[2] must be at least one-dimensional.");
                }
                if (start_path.ndimension() != end_path.ndimension()) {
                    throw std::invalid_argument("path[0] and path[2] must have the same number of dimensions.");
                }
                if (middle_path.ndimension() - 1 != end_path.ndimension()) {
                    throw std::invalid_argument("path[0] and path[2] must have precisely one fewer dimension than "
                                                "path[1].");
                }
                // TODO: This still doesn't test that the dimensions of start_path, middle_path and end_path are
                //       compatible. (Although that should probably be caught by the torch::cat-s later though.)
                assert_increasing(middle_times);
                if ((end_times <= middle_times[-1]).item().to<bool>()) {
                    throw std::invalid_argument("Not an increasing sequence.");
                }
                if ((start_times >= middle_times[0]).item().to<bool>()) {
                    throw std::invalid_argument("Not an increasing sequence.");
                }
                if (additional_times.ndimension() != 1) {
                    throw std::invalid_argument("additional_times must be one-dimensional.");
                }
                if ((additional_times >= end_times).any().item().to<bool>()) {
                    throw std::invalid_argument("additional times must be strictly within the interval specified by "
                                                "times.");
                }
                if ((additional_times <= start_times).any().item().to<bool>()) {
                    throw std::invalid_argument("additional times must be strictly within the interval specified by "
                                                "times.");
                }
                assert_increasing(additional_times);
            }

            int64_t additional_time_index = 0;
            int64_t next_time_index = 0;
            auto prev_time = start_times;
            auto prev_path = start_path;
            auto next_time = middle_times[0];
            auto next_path = len_index(middle_path, 0);

            std::vector<torch::Tensor> new_times;
            std::vector<torch::Tensor> new_path;
            // # + 2 because of start_times and end_times
            new_times.reserve(2 + middle_times.size(0) + additional_times.size(0));
            new_path.reserve(2 + middle_times.size(0) + additional_times.size(0));
            new_times.push_back(start_times);
            new_path.push_back(start_path);

            for (; additional_time_index < additional_times.size(0); ++additional_time_index) {
                auto additional_time = additional_times[additional_time_index];
                if ((additional_time > next_time).item().to<bool>()) {
                    break;
                }
                if ((additional_time == next_time).item().to<bool>()) {
                    // Note that this operation technically subtly breaks differentiability wrt time, because taking
                    // uniques isn't a differentiable operation. Still, it doesn't do it too badly so this should be
                    // fine.
                    continue;
                }

                auto ratio = (additional_time - prev_time) / (next_time - prev_time);
                auto path_at_additional_time = prev_path + ratio * (next_path - prev_path);
                new_times.push_back(additional_time);
                new_path.push_back(path_at_additional_time);
            }

            for (; additional_time_index < additional_times.size(0); ++additional_time_index) {
                auto additional_time = additional_times[additional_time_index];
                while ((additional_time > next_time).item().to<bool>()) {
                    new_times.push_back(next_time);
                    new_path.push_back(next_path);
                    prev_time = next_time;
                    prev_path = next_path;
                    ++next_time_index;
                    if (next_time_index == middle_times.size(0)) {
                        next_time = end_times;
                        next_path = end_path;
                    }
                    else{
                        next_time = middle_times[next_time_index];
                        next_path = len_index(middle_path, next_time_index);
                    }
                }
                if ((additional_time == next_time).item().to<bool>()) {
                    // Note that this operation technically subtly breaks differentiability wrt time, because taking
                    // uniques isn't a differentiable operation. Still, it doesn't do it too badly so this should be
                    // fine.
                    continue;
                }
                auto ratio = (additional_time - prev_time) / (next_time - prev_time);
                auto path_at_additional_time = prev_path + ratio * (next_path - prev_path);
                new_times.push_back(additional_time);
                new_path.push_back(path_at_additional_time);
            }

            for (; next_time_index < middle_times.size(0); ++next_time_index) {
                new_times.push_back(middle_times[next_time_index]);
                new_path.push_back(len_index(middle_path, next_time_index));
            }
            new_times.push_back(end_times);
            new_path.push_back(end_path);

            return std::tuple<torch::Tensor, torch::Tensor> {torch::stack(new_times, /*dim=*/0),
                                                             torch::stack(new_path, /*dim=*/-2)};
        }
    }  // namespace torchshapelets::detail

    torch::Tensor shapelet_transform(torch::Tensor times, torch::Tensor path, torch::Tensor lengths,
                                     torch::Tensor shapelets, int64_t max_length, int64_t num_samples,
                                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor)>& discrepancy_fn) {
        if ((lengths.max() > max_length).item().to<bool>()) {
            throw std::invalid_argument("Shapelets have exceeded maximum length; please remember to call the "
                                        "`clip_length` method after each backward pass. (After optimiser.step())");
        }
        if (times.ndimension() != 1) {
            throw std::invalid_argument("times must be a 1D tensor of shape (length,).");
        }
        if (path.ndimension() < 2) {
            throw std::invalid_argument("path must be a tensor of shape(..., length, channels).");
        }
        if (lengths.ndimension() != 1) {
            throw std::invalid_argument("lengths must be a 1D tensor of shape (num_shapelets,).");
        }
        if (shapelets.ndimension() != 3) {
            throw std::invalid_argument("shapelets must be a 3D tensor of shape (num_shapelets, num_shapelet_samples, "
                                        "channels).");
        }
        if (times.size(0) < 2) {
            throw std::invalid_argument("times must be of size at least 2 to define a path.");
        }
        detail::assert_increasing(times);
        if ((max_length > times[-1] - times[0]).item().to<bool>()) {
            throw std::invalid_argument("Time series is too short for shapelet.");
        }
        if (path.size(-1) != shapelets.size(2)) {
            throw std::invalid_argument("Path and shapelets have different numbers of channels.");
        }
        if (path.size(-2) != times.size(0)) {
            throw std::invalid_argument("Path and times have different lengths.");
        }
        if (lengths.size(0) != shapelets.size(0)) {
            throw std::invalid_argument("lengths and shapelets have different numbers of shapelets.");
        }

        py::gil_scoped_release release;  // Needed to make Python-based discrepancy functions work
        detail::omp_nested omp_nested_instance;

        auto device = torch::device(times.device());  // Actually an instance of TensorOptions

        const auto num_shapelets = shapelets.size(0);
        const auto num_shapelet_samples = shapelets.size(1);
        std::vector<torch::Tensor> discrepancies (num_shapelets);

        #pragma omp parallel for default(none) shared(times, path, lengths, shapelets, num_samples, \
                                                      device, discrepancies, discrepancy_fn, python_discrepancy_fn)
        for (int64_t shapelet_index = 0; shapelet_index < num_shapelets; ++shapelet_index) {
            auto length = lengths[shapelet_index];
            auto shapelet = shapelets[shapelet_index];
            auto shapelet_times = torch::linspace(0, length.detach().item(), num_shapelet_samples, device);
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                shapelet_times_tuple {shapelet_times[0],
                                      shapelet_times.narrow(/*dim=*/0, /*start=*/1, /*length=*/shapelet_times.size(0) - 2),
                                      shapelet_times[-1]};
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                shapelet_tuple {detail::len_index(shapelet, 0),
                                shapelet.narrow(/*dim=*/-2, /*start=*/1, /*length=*/shapelet.size(0) - 2),
                                detail::len_index(shapelet, -1)};

            auto min_fn = [times, path, length, shapelet_times_tuple, shapelet_tuple, python_discrepancy_fn,
                           &discrepancy_fn] (torch::Tensor point) {
                // restricted path is of shape (..., restricted_length, in_channels)
                std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> restricted_times_tuple, restricted_path_tuple;
                std::tie(restricted_times_tuple, restricted_path_tuple) = detail::restriction(times, path, point,
                                                                                              point + length);
                std::get<0>(restricted_times_tuple) = std::get<0>(restricted_times_tuple) - point;
                std::get<1>(restricted_times_tuple) = std::get<1>(restricted_times_tuple) - point;
                std::get<2>(restricted_times_tuple) = std::get<2>(restricted_times_tuple) - point;
                // normalise both the path and the shapelet to have knots at the same points as each other. Slice with
                // [1:-1] because otherwise floating point inaccuracies mean that spurious errors can get thrown, and it
                // doesn't matter anyway as they have the same start and endpoints.
                torch::Tensor mutual_times, knot_restricted_path, knot_shapelet;
                std::tie(mutual_times, knot_restricted_path) = detail::add_knots(restricted_times_tuple,
                                                                                 restricted_path_tuple,
                                                                                 std::get<1>(shapelet_times_tuple));



                std::tie(mutual_times, knot_shapelet) = detail::add_knots(shapelet_times_tuple,
                                                                          shapelet_tuple,
                                                                          std::get<1>(restricted_times_tuple));

                return discrepancy_fn(mutual_times, knot_restricted_path, knot_shapelet);
            };
            auto discrepancy = detail::continuous_min(times[0], times[-1] - length, min_fn, device,
                                                      num_samples);
            discrepancies[shapelet_index] = discrepancy;
        }

        // returns a tensor of shape (..., num_shapelets)
        return torch::stack(discrepancies, /*dim=*/-1);
    }
}  // namespace torchshapelets