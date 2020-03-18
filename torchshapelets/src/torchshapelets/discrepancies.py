import math
try:
    import signatory
except ImportError:
    signatory = None
import torch

from . import _impl


class CppDiscrepancy(torch.nn.Module):
    """Discrepancy functions can either be coded in Python or C++.

    If they're done in C++ then they can be parallelised over, whereas if they are done in Python then the GIL implies
    that there's some serialisation.

    In order to facilitate this, C++ discrepancy functions should be subclasses of this class. They should have the
    C++ function as a 'fn' attribute, and precisely one Tensor, that they get to control, as an 'arg' attribute.
    """
    # Every instance of a subclass must have two attributes available:
    # A function 'fn' with signature (Tensor, Tensor, Tensor, tuple) -> Tensor
    # A tuple 'args'.

    def forward(self, time, path1, path2):
        # We never actually call this forward method as part of the shapelet transform, but it's here in case people
        # want to try calling this outside of the shapelet transform.
        return self.fn(time, path1, path2, self.arg)


class L2Discrepancy(CppDiscrepancy):
    """Computes the L2 discrepancy between two paths.

    Should be initialised with some number of `in_channels`, and integer, and whether to be a `psuedometric`, a boolean.

    Should then called with three arguments: `times`, `path1`, `path2`.

    Let f be the unique continuous piecewise linear function such that
    ```
    f(times[i]) == path1[i] for all i
    ```
    and let g be the unique continuous piecewise linear function such that
    ```
    g(times[i]) == path2[i] for all i
    ```
    where `times` is a strictly increasing 1D tensor of shape (length,) for some  value `length`, and `path1` is a
    tensor of shape (..., length, in_channels) and `path2` is a tensor of shape (length, in_channels). [So yes, `path2`
    has no batch dimensions.]
    Finally if `pseudometric` is False then let A = 1 (the scalar); if `pseudometric` is True let A_ij be a learnt
    matrix of size (in_channels, in_channels).

    Then this function computes
    ```
    sqrt( \int_{times[0]}^{times[-1]} || Af(t) - Ag(t) ||_2^2 dt )
    ```
    i.e. the L2 norm of Af - Ag.
    Which will be a tensor of shape (...).
    """
    fn = _impl.l2_discrepancy

    def __init__(self, in_channels, pseudometric=True):
        super(L2Discrepancy, self).__init__()

        self.in_channels = in_channels
        self.pseudometric = pseudometric

        if pseudometric:
            linear = torch.empty(in_channels, in_channels, requires_grad=True)
            torch.nn.init.kaiming_uniform_(linear, a=math.sqrt(5))
            self.arg = torch.nn.Parameter(linear)
        else:
            self.arg = torch.nn.Parameter(torch.Tensor())

        
class LogsignatureDiscrepancy(torch.nn.Module):
    """Calculates the p-logsignature distance between two paths."""
    def __init__(self, in_channels, depth, p=2, pseudometric=True):
        """
        Arguments:
            in_channels: The number of input channels of the path.
            depth: An integer describing the depth of the logsignature transform to take.
            p: A number in [1, \infty] specifying the parameter p of the distance. Defaults to 2.
            pseudometric: Whether to take a learnt linear transformation beforehand. Defaults to True.
        """
        super(LogsignatureDiscrepancy, self).__init__()

        if signatory is None:
            raise ImportError("Signatory must be installed to compute logsignature discrepancies. It can be found at "
                              "`https://github.com/patrick-kidger/signatory`. See also the installation instructions "
                              "for `torchshapelets` at "
                              "`https://github.com/jambo6/generalised_shapelets/tree/master/torchshapelets`. ")

        self.in_channels = in_channels
        self.depth = depth
        self.p = p
        self.pseudometric = pseudometric

        if pseudometric:
            logsignature_channels = signatory.logsignature_channels(in_channels + 1, depth)  # +1 for time
            self.linear = torch.nn.Linear(logsignature_channels, logsignature_channels, bias=False)
        else:
            self.register_parameter('linear', None)
        
    def forward(self, times, path1, path2):
        # times has shape (length,)
        # path1 has shape (..., length, channels)
        # path2 has shape (*, length, channels)

        path1_batch_dims = path1.shape[:-2]
        path2_batch_dims = path2.shape[:-2]

        # append time to both paths
        time_channel1 = time_channel2 = times.unsqueeze(-1)
        for dim in path1_batch_dims:
            time_channel1 = time_channel1.unsqueeze(0).expand(dim, *time_channel1.shape)
        for dim in path2_batch_dims:
            time_channel2 = time_channel2.unsqueeze(0).expand(dim, *time_channel2.shape)
        path1 = torch.cat([time_channel1, path1], dim=-1)
        path2 = torch.cat([time_channel2, path2], dim=-1)

        # Create a single batch dimension for compatibility with Signatory
        path1 = path1.view(-1, path1.size(-2), path1.size(-1))
        path2 = path2.view(-1, path2.size(-2), path2.size(-1))

        logsignature1 = signatory.logsignature(path1, self.depth)
        logsignature2 = signatory.logsignature(path2, self.depth)

        logsignature1 = logsignature1.view(*path1_batch_dims, logsignature1.size(-1))
        logsignature2 = logsignature2.view(*path2_batch_dims, logsignature2.size(-1))

        for _ in path1_batch_dims:
            logsignature2.unsqueeze_(0)
        for _ in path2_batch_dims:
            logsignature1.unsqueeze_(-2)
        logsignature1 = logsignature1.expand(*path1_batch_dims, *path2_batch_dims, logsignature1.size(-1))
        logsignature2 = logsignature2.expand(*path1_batch_dims, *path2_batch_dims, logsignature1.size(-1))

        logsignature_diff = logsignature1 - logsignature2
        if self.pseudometric:
            logsignature_diff = self.linear(logsignature_diff)

        out = logsignature_diff.norm(p=self.p, dim=-1)
        return out