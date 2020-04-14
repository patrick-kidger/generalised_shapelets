import torch

from . import discrepancies
from . import _impl


class GeneralisedShapeletTransform(torch.nn.Module):
    """Applies a generalised shapelet transform.

    Each shapelet that it is compared against will be a piecewise linear function.
    """
    def __init__(self, in_channels, num_shapelets, num_shapelet_samples, discrepancy_fn,  max_shapelet_length,
                 lengths_per_shapelet=1, num_continuous_samples=None, scale_length_gradients='auto'):
        """
        Arguments:
            in_channels: An integer specifying the number of input channels in the path it will be called with.
            num_shapelets: How many shapelets to compare the path against.
            num_shapelet_samples: How finely each shapelet will be discretised, i.e. how many knots it has.
            discrepancy_fn: The function measuring the similarity or discrepancy between a shapelet and a path. This
                function should take three arguments, call them `times`, `path`, `shapelet`. `times` will be a 1D tensor
                of shape (length,), whilst `path` will be a tensor of shape (..., length, channels) and `shapelet` will
                be a tensor of shape (length, channels), where '...' represents some non-negative number of batch
                dimensions. Then `times` and `path` between them describe the unique continuous piecwise affine path `f`
                such that f(times[i]) = path1[i] for all i, whilst `times` and `shapelet` similarly define the unique
                continuous piecewise affine path `g` such that g(times[i]) = shapelet[i].  It should return a tensor of
                shape (...,) describing the similarity between `path` and `shapelet`.
            max_shapelet_length: The maximum length for a shapelet. (As if it grows too long then it cannot be compared
                against.)
            lengths_per_shapelet: The number of lengths that each shapelet is compared against.
            num_continuous_samples: We compute a minimum over s in {start, start + 1/num_samples, start + 2/num_samples,
                ..., end - 1/num_samples, end}, where `start` and `end` are the endpoints of the times that this is
                called with. Defaults to the same as the length of the number of times.
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
        self.num_continuous_samples = int(num_continuous_samples) if num_continuous_samples is not None else None
        self.max_shapelet_length = max_shapelet_length
        self.lengths_per_shapelet = lengths_per_shapelet
        self.scale_length_gradients = scale_length_gradients

        self.lengths = torch.nn.Parameter(torch.empty(num_shapelets * lengths_per_shapelet))
        self.shapelets = torch.nn.Parameter(torch.empty(num_shapelets, num_shapelet_samples, in_channels))

        self.reset_parameters()

        if scale_length_gradients == 'auto':
            scale = max_shapelet_length
        else:
            scale = scale_length_gradients
        self.lengths.register_hook(lambda grad: scale * grad)

    def extra_repr(self):
        return "in_channels={}, num_shapelets={}, num_shapelet_samples={}, num_continuous_samples={}, " \
               "lengths_per_shapelet={}, max_shapelet_length={}".format(self.in_channels, self.num_shapelets,
                                                                        self.num_shapelet_samples,
                                                                        self.num_continuous_samples,
                                                                        self.lengths_per_shapelet,
                                                                        self.max_shapelet_length)

    def reset_parameters(self, times=None, path=None):
        with torch.no_grad():
            self.lengths.uniform_(self.max_shapelet_length / 2, self.max_shapelet_length)
            if times is None:
                assert path is None, "Both times and path must be either None or not None."
                self.shapelets.uniform_(-1, 1)
            else:
                assert path is not None, "Both times and path must be either None or not None."
                assert path.ndimension() == 3, "path must be a 3 dimensional tensor of shape (num_shapelets, " \
                                               "length, channels)"
                assert path.size(0) == self.num_shapelets

                lengths = self.lengths[:self.num_shapelets]

                _impl.check_inputs(times, path, lengths, self.max_shapelet_length)
                start_times = times[0] + torch.rand_like(lengths) * (times[-1] - times[0] - lengths)

                for start_time, length, shapelet, path_elem in zip(start_times, lengths, self.shapelets, path):
                    shapelet_times = torch.linspace(start_time, start_time + length, self.num_shapelet_samples)
                    shapelet.copy_(_impl.unsafe_add_knots((times[0], times[1:-1], times[-1]),
                                                          (path_elem[0], path_elem[1:-1], path_elem[-1]),
                                                          shapelet_times,
                                                          False)[1])

    def clip_length(self):
        """Clips the length of the shapelets to valid values. Should be called after every backward pass. (i.e. after
        optimiser.step())"""
        with torch.no_grad():
            self.lengths.clamp_(0.01, self.max_shapelet_length)

    def forward(self, times, path):
        # times is of shape (length,)
        # path is of shape (..., length, in_channels)

        if not torch.isfinite(path).any():
            # Done in Python as there's no torch::isfinite in the C++
            # We explicitly check this because otherwise the error message that results is perfectly unhelpful.
            raise ValueError('path cannot have non-finite values.')

        if isinstance(self.discrepancy_fn, discrepancies.CppDiscrepancy):
            discrepancy_fn = self.discrepancy_fn.fn
            discrepancy_arg = self.discrepancy_fn.arg
        else:
            discrepancy_fn = lambda times, path1, path2, args: self.discrepancy_fn(times, path1, path2)
            discrepancy_arg = torch.Tensor()

        if self.num_continuous_samples is None:
            num_continuous_samples = len(times)
        else:
            num_continuous_samples = self.num_continuous_samples

        times = torch.as_tensor(times, dtype=path.dtype, device=path.device)
        max_shapelet_length = torch.as_tensor(self.max_shapelet_length, dtype=path.dtype, device=path.device)
        shapelets_repeated = self.shapelets.repeat(self.lengths_per_shapelet, 1, 1)
        return _impl.shapelet_transform(times, path, self.lengths, shapelets_repeated, max_shapelet_length,
                                        num_continuous_samples, discrepancy_fn, discrepancy_arg)
