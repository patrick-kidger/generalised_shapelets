import torch


def _restriction(times, path, start, end):
    """Differentiably restricts a piecewise linear path.
    
    The arguments `times` and `path` between them define a piecewise linear function f, with
    f(times[i]) = path[..., i, :] for each i, and affine on the pieces in between.
    
    The return value is this piecewise linear function restricted to the interval [start, end], represented in the same
    way by a pair of (times, path).
    
    Arguments:
        times: A 1D tensor of shape (length,), describing the times of each knot.
        path: A tensor of shape (..., length, channels), describing the values of each knot.
        start: The start time to restrict to.
        end: The end time to restrict to.

    Returns:
        A 2-tuple of (restricted_times, restricted_path), where restricted_times is a 1D tensor of shape (r_length,) and
        restricted_path is a tensor of shape(..., r_length, channels).
    """
    
    times = torch.as_tensor(times)
    path = torch.as_tensor(path)
    start = torch.as_tensor(start)
    end = torch.as_tensor(end)
    
    assert len(times.shape) == 1, "times must be a 1D tensor of shape (length,)."
    assert len(path.shape) >= 2, "path must be a tensor of shape(..., length, channels)."
    assert len(start.shape) == 0, "start and end must be scalars."
    assert len(end.shape) == 0, "start and end must be scalars."
    assert path.size(-2) == times.size(0), "times and path must have the same size length dimension."
    assert times.size(0) >= 2, "Length dimension must be of size at least 2 to define a path."
    assert start >= times[0], "start and end must be within the interval specified by times."
    assert end <= times[-1], "start and end must be within the interval specified by times."
    assert start < end, "start must be less than end."
    prev_time = times[0]
    for time in times[1:]:
        assert time > prev_time, "times must be an increasing sequence."
        prev_time = time
    
    # >= for start and > for end is deliberate
    # this correctly handles the cases that start == times[i] or end == times[i] for some i.
    start_index = (start >= times).sum() - 1
    end_index = (end > times).sum() - 1
    
    before_start_time = times[start_index]
    after_start_time = times[start_index + 1]
    start_ratio = (start - before_start_time) / (after_start_time - before_start_time)
    start_restriction = path[..., start_index, :] + start_ratio * (path[..., start_index + 1, :] - path[..., start_index, :])
    
    middle_restriction = path[..., start_index + 1:end_index, :]
    
    before_end_time = times[end_index]
    after_end_time = times[end_index + 1]
    end_ratio = (end - before_end_time) / (after_end_time - before_end_time)
    end_restriction = path[..., end_index, :] + end_ratio * (path[..., end_index + 1, :] - path[..., end_index, :])
    
    restricted_times = torch.cat([start.unsqueeze(0), times[start_index + 1:end_index], end.unsqueeze(0)], dim=0)
    restricted_path = torch.cat([start_restriction.unsqueeze(-2), middle_restriction, end_restriction.unsqueeze(-2)],
                                dim=-2)
    return restricted_times, restricted_path
   
    
def _continuous_min(start, end, fn, max_sampling_gap=None, num_samples=1000):
    """Differentiably calculates
    ```
    g(start, end) = \min_{s \in [start, end]} fn(s)
    ```

    In practice this is done by sampling many points in the region [start, end] and calculating the minimum over all of
    them. (And handling 'start' and 'end' specially so that they may be differentiated though.) The maximum tolerance
    for the gap between points is controlled by `max_sampling_gap`.

    Arguments:
        start: A scalar.
        end: A scalar.
        fn: A function which consumes a scalar and returns an any-dimensional tensor.
        max_sampling_gap, num_samples: Precisely one of these must be non-None. If max_sampling_gap is not None then we
            compute a minimum over s in {start, start + epsilon, start + 2 * epsilon, ..., end - epsilon, end} where
            epsilon is the largest epsilon allowing this exact splitting, such that epsilon < max_sampling_gap. If
            num_samples is not None then it computes a minimum over s in {start, start + 1/num_samples,
            start + 2/num_samples, ..., end - 1/num_samples, end}.

    Returns:
        A PyTorch tensor of the same shape as `fn` returns.
    """
    start = torch.as_tensor(start)
    end = torch.as_tensor(end)
    start_detached = start.detach()
    end_detached = end.detach()
    
    assert start < end, "start must be less than end."
    assert max_sampling_gap > 0, "max_sampling_gap must be positive."
    assert not (max_sampling_gap is None and num_samples is None), "Cannot have both max_sampling_gap and " \
                                                                   "num_samples set to None"
    assert not (max_sampling_gap is not None and num_samples is not None), "Cannot pass both max_sampling_gap and " \
                                                                           "num_samples."
    
    if max_sampling_gap is not None:
        max_sampling_gap = torch.as_tensor(max_sampling_gap)
        assert not max_sampling_gap.requires_grad, "Cannot differentiate with respect to the number of sample points."
        num_samples = 1 + torch.ceil((end_detached - start_detached) / max_sampling_gap)
        num_samples = num_samples.to(torch.long)
    points = torch.linspace(start_detached, end_detached, num_samples)
    
    min_result_middle = torch.stack([fn(point) for point in points[1:-1]], dim=0).min(dim=0).values
    # This allows us to differentiate through the endpoints
    # If we wanted we could also construct every point in points[1:,-1] differentiably, but we're aiming for a
    # continuous minimum here, and that's just an implementation detail. (Basically it depends on whether you use a
    # discretise-then-optimise or optimise-then-discretise approach.)
    min_result = torch.stack([fn(start), min_result_middle, fn(end)], dim=0).min(dim=0).values
    return min_result


def _add_knots(times, path, additional_times):
    """Adds knots to a piecewise linear path, such that it doesn't change the function it represents.

    Arguments:
        times, path: As `_restriction`.
        additional_times: Additional times to add knots to by linearly interpolating between the values already given.

    Returns:
        A tensor of shape (..., times.size(0) + additional_times.size(0), path.size(-1)).
    """
    times = torch.as_tensor(times)
    path = torch.as_tensor(path)
    # Adding knots doesn't change the underlying function so detaching is the correct thing to do
    additional_times = torch.as_tensor(additional_times).detach()

    if len(additional_times) == 0:
        return times, path

    prev_time = times[0]
    for time in times[1:]:
        assert time > prev_time, "times must be increasing"
        prev_time = time
    prev_additional_time = additional_times[0]
    for additional_time in additional_times[1:]:
        assert additional_time > prev_additional_time, "additional_times must be increasing"
        prev_additional_time = additional_time
    assert additional_times[0] >= times[0], "additional_times cannot go outside the interval [times[0], times[-1]]"
    assert additional_times[-1] <= times[-1], "additional_times cannot go outside the interval [times[0], times[-1]]"

    prev_time_index = 0
    iter_times = iter(times)
    prev_time = next(iter_times)
    next_time = next(iter_times)
    new_path = [path[..., 0, :]]
    new_times = [times[0]]
    _skips = 0
    if additional_times[0] == times[0]:
        # Note that this operation technically subtly breaks differentiability wrt time, because taking uniques
        # isn't a differentiable operation. Still, it doesn't do it too badly so this should be fine.
        additional_times = additional_times[1:]
    for additional_time in additional_times:
        while additional_time > next_time:
            prev_time_index += 1
            new_times.append(times[prev_time_index])
            new_path.append(path[..., prev_time_index, :])
            prev_time = next_time
            next_time = next(iter_times)
        if additional_time == next_time:
            # Similarly, this operation subtly breaks differentiability.
            continue
        ratio = (additional_time - prev_time) / (next_time - prev_time)
        path_at_additional_time = path[..., prev_time_index, :] + ratio * (path[..., prev_time_index + 1, :] - path[..., prev_time_index, :])
        new_times.append(additional_time)
        new_path.append(path_at_additional_time)
    for _ in iter_times:
        prev_time_index += 1
        new_times.append(times[prev_time_index])
        new_path.append(path[..., prev_time_index, :])
    new_times.append(times[-1])
    new_path.append(path[..., -1, :])

    return torch.stack(new_times, dim=0), torch.stack(new_path, dim=-2)
  
    
class GeneralisedShapeletTransform(torch.nn.Module):
    """Applies a generalised shapelet transform.

    Each shapelet that it is compared against will be a piecewise linear function.
    """
    def __init__(self, in_channels, num_shapelets, num_shapelet_samples, discrepancy_fn,  max_shapelet_length,
                 continuous_sampling_gap, num_continuous_samples, scale_length_gradients='auto'):
        """
        Arguments:
            in_channels: An integer specifying the number of input channels in the path it will be called with.
            num_shapelets: How many shapelets to compare the path against.
            num_shapelet_samples: How finely each shapelet will be discretised, i.e. how many knots it has.
            discrepancy_fn: The function measuring the similarity or discrepancy between two paths. This function should
                take three arguments, call them `times`, `path1`, `path2`. Then `times` and `path1` between them
                describe the unique continuous piecwise affine path `f` such that f(times[i]) = path1[i] for all i,
                whilst `times` and `path2` similarly define the unique continuous piecewise affine path `g` such that
                g(times[i]) = path2[i]. `times` will be a 1D tensor of shape (length,), whilst `path1` and `path2`
                should be tensors of shape (..., length, channels) and (length, channels) respectively, where '...'
                '...' represents some non-negative number of batch dimensions. It should return a tensor of shape (...,)
                describing the similarity between `path1` and `path2`.
            max_shapelet_length: The maximum length for a shapelet. (As if it grows too long then it cannot be compared
                against.)
            continuous_sampling_gap, num_continuous_samples: Precisely one of these must be non-None. If
                continuous_sampling_gap is not None then we compute a minimum over s in {start, start + epsilon,
                start + 2 * epsilon, ..., end - epsilon, end} where epsilon is the largest epsilon allowing this exact
                splitting, such that epsilon < continuous_sampling_gap. If num_continuous_samples is not None then it
                computes a minimum over s in {start, start + 1/num_samples, start + 2/num_samples, ...,
                end - 1/num_samples, end}.
            scale_length_gradients: Shapelet lengths are often much larger than other parameters in models, so their
                learning rates should be larger as well. This can either be done by setting parameter-specific learning
                rates in the optimiser, but by default we apply it automatically. This may be disabled by setting this
                parameter to 1 (i.e. no scaling), or set to a specific scaling value by passing that as a value. By
                default a suitable scaling is inferred from the max_shapelet_length argument.
        """
        
        super(GeneralisedShapeletTransform, self).__init__()
        
        self.in_channels = in_channels
        self.num_shapelets = num_shapelets
        self.num_shapelet_samples = num_shapelet_samples
        self.discrepancy_fn = discrepancy_fn
        self.continuous_sampling_gap = continuous_sampling_gap
        self.num_continuous_samples = num_continuous_samples
        self.max_shapelet_length = max_shapelet_length
        self.scale_length_gradients = scale_length_gradients

        self.lengths = torch.nn.Parameter(torch.empty(num_shapelets))
        self.shapelets = torch.nn.Parameter(torch.empty(num_shapelets, num_shapelet_samples, in_channels))

        self.reset_parameters()

        if scale_length_gradients == 'auto':
            scale = max_shapelet_length
        else:
            scale = scale_length_gradients
        self.lengths.register_hook(lambda grad: scale * grad)

    def reset_parameters(self):
        with torch.no_grad():
            self.lengths.uniform_(self.max_shapelet_length / 2, self.max_shapelet_length)
            self.shapelets.uniform_(-1, 1)

    def clip_length(self):
        """Clips the length of the shapelets to valid values. Should be called after every backward pass. (i.e. after
        optimiser.step())"""
        with torch.no_grad():
            self.lengths.clamp_(0.01, self.max_shapelet_length)
        
    def forward(self, times, path):
        # times is of shape (length,)
        # path is of shape (..., length, in_channels)

        assert self.lengths.max() <= self.max_shapelet_length, ("Shapelets have exceeded maximum length; please "
                                                                "remember to call the `clip_length` method after each "
                                                                "backward pass. (After optimiser.step())")
        assert self.max_shapelet_length < times[-1] - times[0], "Time series is too short for shapelet."
        assert path.size(-1) == self.in_channels, ("Wrong number of input channels. Expected {}, got {}"
                                                   "".format(self.in_channels, path.size(-1)))
                                                   
        discrepancies = []
        for length, shapelet in zip(self.lengths, self.shapelets):
            def min_fn(point):
                # restricted path is of shape (..., restricted_length, in_channels)
                restricted_times, restricted_path = _restriction(times, path, point, point + length)
                restricted_times = restricted_times - point
                # detach is fine because adding knots doesn't actually change the underlying function; there's no
                # gradients that need backpropagating.
                shapelet_times = torch.linspace(0, length.detach(), self.num_shapelet_samples)
                # normalise both the path and the shapelet to have knots at the same points as each other. Slice with
                # [1:-1] because otherwise floating point inaccuracies mean that spurious errors can get thrown, and it
                # doesn't matter anyway as they have the same start and endpoints.
                mutual_times, restricted_path = _add_knots(restricted_times, restricted_path, shapelet_times[1:-1])
                _, shapelet_ = _add_knots(shapelet_times, shapelet, restricted_times[1:-1])
                # this will then return a tensor of shape (...,)
                return self.discrepancy_fn(mutual_times, restricted_path, shapelet_)
            discrepancy = _continuous_min(times[0], times[-1] - length, min_fn, self.continuous_sampling_gap,
                                          self.num_continuous_samples)
            discrepancies.append(discrepancy)
        # returns a tensor of shape (..., num_shapelets)
        return torch.stack(discrepancies, dim=-1)
