import torch

from . import _impl


class GeneralisedShapeletTransform(torch.nn.Module):
    """Applies a generalised shapelet transform.

    Each shapelet that it is compared against will be a piecewise linear function.
    """
    def __init__(self, in_channels, num_shapelets, num_shapelet_samples, discrepancy_fn,  max_shapelet_length,
                 num_continuous_samples, scale_length_gradients='auto'):
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
            num_continuous_samples: We compute a minimum over s in {start, start + 1/num_samples, start + 2/num_samples,
                ..., end - 1/num_samples, end}.
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
        return _impl.shapelet_transform(times, path, self.lengths, self.shapelets, self.max_shapelet_length,
                                        self.num_continuous_samples)
