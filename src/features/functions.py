import numpy as np
import torch


def pytorch_rolling(x, dimension, window_size, step_size=1, return_same_size=False):
    """ Outputs an expanded tensor to perform rolling window operations on a pytorch tensor.

    Given an input tensor of shape [N, L, C] and a window length W, computes an output tensor of shape [N, L, C, W]
    where the final dimension contains the values from the current timestep to timestep - W + 1.

    Args:
        x (torch.Tensor): Tensor of shape [N, L, C].
        dimension (int): Dimension to open.
        window_size (int): Length of the rolling window.
        step_size (int): Window step, defaults to 1.
        return_same_size (bool): Set True to return a tensor of the same size as the input tensor with nan values filled
                                 where insufficient prior window lengths existed. Otherwise returns a reduced size
                                 tensor from the paths that had sufficient data.

    Returns:
        torch.Tensor: Tensor of shape [N, L, C, W] where the window values are opened into the fourth W dimension.
    """
    if return_same_size:
        x_dims = list(x.size())
        x_dims[dimension] = window_size - 1
        nans = np.nan * torch.zeros(x_dims)
        x = torch.cat((nans, x), dim=dimension)

    # Unfold ready for mean calculations
    unfolded = x.unfold(dimension, window_size, step_size)

    return unfolded

