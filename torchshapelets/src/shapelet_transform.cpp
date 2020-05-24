#include <torch/extension.h>
#include <cstdint>    // int64_t
#include <functional>  // std::function
#include <omp.h>
#include <stdexcept>  // std::invalid_argument
#include <tuple>      // std::get, std::tie, std::tuple

#include "shapelet_transform.hpp"


namespace torchshapelets {
    namespace detail {
        struct omp_nested {
            omp_nested() :
            was_omp_max_active_levels(omp_get_max_active_levels()), was_omp_in_parallel(omp_in_parallel())
            {
                if (!was_omp_in_parallel) {
                    // batching over shapelets + batching over the continuous min
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
            if (sequence.size(0) > 0) {
                auto prev_time = sequence[0];
                for (int64_t i = 1; i < sequence.size(0); ++i) {
                    if ((sequence[i] < prev_time).item().to<bool>()) {
                        throw std::invalid_argument("Not an increasing sequence.");
                    }
                    prev_time = sequence[i];
                }
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

            auto middle_times = times.narrow(/*dim=*/0, /*start=*/start_index + 1, /*length=*/end_index - start_index);

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
        std::tuple<torch::Tensor, torch::Tensor> continuous_min(torch::Tensor start, torch::Tensor end, fn_type fn, int64_t num_samples) {
            auto start_detached = start.detach().item();
            auto end_detached = end.detach().item();

            auto points = torch::linspace(start_detached, end_detached, num_samples, start.options());

            // We compute the minimum over the middle region separately, as there's good odds that this bit won't
            // require gradients. If I understand PyTorch's autograd well enough (and this really might be wrong on my
            // part), then this saves the creation of a zero-initialised tensor for each one, during backpropagation.
            std::vector<torch::Tensor> results (num_samples);
            results[0] = fn(start);
            #pragma omp parallel for default(none) shared(results, points, fn, num_samples)
            for (int64_t point_index = 1; point_index < num_samples - 1; ++point_index) {
                results[point_index] = fn(points[point_index]);
            }
            results[num_samples - 1] = fn(end);
            auto out = torch::stack(results, /*dim=*/0).min(/*dim=*/0);
            return std::tuple<torch::Tensor, torch::Tensor> {std::get<0>(out), std::get<1>(out)};
        }
    }  // namespace torchshapelets::detail

    std::tuple<torch::Tensor, torch::Tensor>
    unsafe_add_knots(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> times,
                     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> path,
                     torch::Tensor additional_times, bool keep_original_times) {
        torch::Tensor start_times, middle_times, end_times;  // written during the COVID-19 outbreak
        torch::Tensor start_path, middle_path, end_path;
        std::tie(start_times, middle_times, end_times) = times;
        std::tie(start_path, middle_path, end_path) = path;

        int64_t additional_time_index = 0;
        int64_t next_time_index = 0;
        auto prev_time = start_times;
        auto prev_path = start_path;
        auto next_time = (middle_times.size(0) > 0) ? middle_times[0] : end_times;
        auto next_path = (middle_times.size(0) > 0) ? detail::len_index(middle_path, 0) : end_path;

        std::vector<torch::Tensor> new_times;
        std::vector<torch::Tensor> new_path;
        // # + 2 because of start_times and end_times
        std::vector<torch::Tensor>::size_type r_length;
        if (keep_original_times) {
            r_length = 2 + middle_times.size(0) + additional_times.size(0);
        }
        else {
            r_length = additional_times.size(0);
        }
        new_times.reserve(r_length);
        new_path.reserve(r_length);
        if (keep_original_times) {
            new_times.push_back(start_times);
            new_path.push_back(start_path);
        }

        for (; additional_time_index < additional_times.size(0); ++additional_time_index) {
            auto additional_time = additional_times[additional_time_index];
            if ((additional_time > next_time).item().to<bool>()) {
                break;
            }

            auto ratio = (additional_time - prev_time) / (next_time - prev_time);
            auto path_at_additional_time = prev_path + ratio * (next_path - prev_path);
            new_times.push_back(additional_time);
            new_path.push_back(path_at_additional_time);
        }

        for (; additional_time_index < additional_times.size(0); ++additional_time_index) {
            auto additional_time = additional_times[additional_time_index];
            while ((additional_time > next_time).item().to<bool>()) {
                if (keep_original_times) {
                    new_times.push_back(next_time);
                    new_path.push_back(next_path);
                }
                prev_time = next_time;
                prev_path = next_path;
                ++next_time_index;
                if (next_time_index == middle_times.size(0)) {
                    next_time = end_times;
                    next_path = end_path;
                }
                else{
                    next_time = middle_times[next_time_index];
                    next_path = detail::len_index(middle_path, next_time_index);
                }
            }
            auto ratio = (additional_time - prev_time) / (next_time - prev_time);
            auto path_at_additional_time = prev_path + ratio * (next_path - prev_path);
            new_times.push_back(additional_time);
            new_path.push_back(path_at_additional_time);
        }

        if (keep_original_times) {
            for (; next_time_index < middle_times.size(0); ++next_time_index) {
                new_times.push_back(middle_times[next_time_index]);
                new_path.push_back(detail::len_index(middle_path, next_time_index));
            }
            new_times.push_back(end_times);
            new_path.push_back(end_path);
        }

        return std::tuple<torch::Tensor, torch::Tensor> {torch::stack(new_times, /*dim=*/0),
                                                         torch::stack(new_path, /*dim=*/-2)};
    }


    void check_inputs(torch::Tensor times, torch::Tensor path, torch::Tensor lengths, torch::Tensor max_length) {
        if (!times.is_floating_point()) {
            throw std::invalid_argument("times must be a floating point tensor.");
        }
        if (!path.is_floating_point()) {
            throw std::invalid_argument("path must be a floating point tensor.");
        }
        if (!lengths.is_floating_point()) {
            throw std::invalid_argument("lengths must be a floating point tensor.");
        }
        if (!max_length.is_floating_point()) {
            throw std::invalid_argument("max_length must be a floating point tensor.");
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
        if (times.size(0) < 2) {
            throw std::invalid_argument("times must be of size at least 2 to define a path.");
        }
        if (path.size(-2) != times.size(0)) {
            throw std::invalid_argument("Path and times have different lengths.");
        }
        if ((lengths.max() > max_length).item().to<bool>()) {
            throw std::invalid_argument("Shapelets have exceeded maximum length; please remember to call the "
                                        "`clip_length` method after each backward pass. (After optimiser.step())");
        }
        detail::assert_increasing(times);
        if ((max_length > times[-1] - times[0]).item().to<bool>()) {
            throw std::invalid_argument("Time series is too short for shapelet.");
        }
    }

    std::tuple<torch::Tensor, torch::Tensor>
    shapelet_transform(torch::Tensor times, torch::Tensor path, torch::Tensor lengths, torch::Tensor shapelets,
                       torch::Tensor max_length, const int64_t num_samples,
                       const std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)>& discrepancy_fn,
                       torch::Tensor discrepancy_arg) {
        check_inputs(times, path, lengths, max_length);
        if (!shapelets.is_floating_point()) {
            throw std::invalid_argument("shapelets must be a floating point tensor.");
        }
        if (shapelets.ndimension() != 3) {
            throw std::invalid_argument("shapelets must be a 3D tensor of shape (num_shapelets, num_shapelet_samples, "
                                        "channels).");
        }
        if (path.size(-1) != shapelets.size(2)) {
            throw std::invalid_argument("Path and shapelets have different numbers of channels.");
        }
        if (lengths.size(0) != shapelets.size(0)) {
            throw std::invalid_argument("lengths and shapelets have different numbers of shapelets.");
        }
        if (num_samples < 3) {  // 3 is the smallest that avoids indexing errors
            throw std::invalid_argument("num_samples must be at least 3.");
        }
        if (shapelets.size(0) < 3) {  // 3 is the smallest that avoids indexing errors
            throw std::invalid_argument("shapelets must be sampled at least 3 times.");
        }

        py::gil_scoped_release release;  // Needed to make Python-based discrepancy functions work
        detail::omp_nested omp_nested_instance;

        const auto num_shapelets = shapelets.size(0);
        const auto num_shapelet_samples = shapelets.size(1);
        std::vector<torch::Tensor> discrepancies (num_shapelets);
        std::vector<torch::Tensor> indices (num_shapelets);

        #pragma omp parallel for default(none) \
                                 shared(times, path, lengths, shapelets, discrepancies, discrepancy_arg, discrepancy_fn,\
                                        indices)
        for (int64_t shapelet_index = 0; shapelet_index < num_shapelets; ++shapelet_index) {
            auto length = lengths[shapelet_index];
            auto shapelet = shapelets[shapelet_index];
            auto shapelet_times = torch::linspace(0, length.detach().item(), num_shapelet_samples, length.options());
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                shapelet_times_tuple {shapelet_times[0],
                                      shapelet_times.narrow(/*dim=*/0, /*start=*/1, /*length=*/shapelet_times.size(0) - 2),
                                      shapelet_times[-1]};
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                shapelet_tuple {detail::len_index(shapelet, 0),
                                shapelet.narrow(/*dim=*/-2, /*start=*/1, /*length=*/shapelet.size(0) - 2),
                                detail::len_index(shapelet, -1)};

            // Capturing discrepancy_fn by reference is very important.
            // If you don't, _and_ it corresponds to a Python function that has been passed in, then you get
            // nondeterministic segfaults.
            // The reason (I think), is that discrepancy_fn is actually a pybind11-generated object which misbehaves
            // when being copied. Both copies will claim ownership of some underlying bit of Python, so when one of them
            // is done, the underlying resource gets freed too early.
            // This explanation doesn't perfectly fit with this function sometimes running without issues, but this fix
            // seems to work.
            auto min_fn = [times, path, length, shapelet_times_tuple, shapelet_tuple, &discrepancy_fn, discrepancy_arg]
              (torch::Tensor point) {
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
                std::tie(mutual_times, knot_restricted_path) = unsafe_add_knots(restricted_times_tuple,
                                                                                restricted_path_tuple,
                                                                                std::get<1>(shapelet_times_tuple),
                                                                                /*keep_original_times=*/true);
                std::tie(mutual_times, knot_shapelet) = unsafe_add_knots(shapelet_times_tuple,
                                                                         shapelet_tuple,
                                                                         std::get<1>(restricted_times_tuple),
                                                                         /*keep_original_times=*/true);

                return discrepancy_fn(mutual_times, knot_restricted_path, knot_shapelet, discrepancy_arg);
            };
            torch::Tensor discrepancy, index;
            std::tie(discrepancy, index) = detail::continuous_min(times[0], times[-1] - length, min_fn, num_samples);
            discrepancies[shapelet_index] = discrepancy;
            indices[shapelet_index] = index;
        }

        // each tensor is of shape (..., num_shapelets)
        return std::tuple<torch::Tensor, torch::Tensor> {torch::stack(discrepancies, /*dim=*/-1),
                                                         torch::stack(indices, /*dim=*/-1)};
    }
}  // namespace torchshapelets