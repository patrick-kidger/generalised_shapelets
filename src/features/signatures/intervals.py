from definitions import *
import torch
import numpy as np
import signatory
from src.data.make_dataset import UcrDataset
from src.features.signatures.augmentations import AddTime


class IntervalSignature():
    """ Signature computation over intervals. """
    def __init__(self, depth=3, logsig=False, interval_depth=2, dyadic=False):
        self.depth = depth
        self.logsig = logsig
        self.interval_depth = interval_depth
        self.dyadic = dyadic

    def transform(self, data):
        # Setup signature computation function
        path_class = signatory.Path(data, depth=self.depth)
        sig_func = path_class.logsignature if self.logsig else path_class.signature

        # Set intervals
        intervals = get_descending_intervals(data.size(1), depth=self.interval_depth, dyadic=self.dyadic)

        # Compute
        signatures = compute_interval_signature(sig_func, intervals, squeeze=True)

        return signatures


def compute_interval_signature(sig_func, intervals, squeeze=True):
    """Applies signature transform across multiple intervals.

    Args:
        sig_func (signatory.Path.func): Either of the signature.Path.signature or logisgnature methods.
        intervals (list): List of tuples where each tuple is of the form (start_idx, end_idx)
        squeeze (bool): Set True to squeeze out the interval dimension onto the feature dimensions.

    Returns:
        torch.Tensor: [N, C_new] if squeezed else [N, N_intervals, C_new]
    """
    # Simply apply and stack
    signatures = torch.stack([
        sig_func(start_idx, end_idx) for start_idx, end_idx in intervals
    ], dim=1)

    if squeeze:
        signatures = signatures.view(signatures.size(0), -1)

    return signatures


def get_descending_intervals(length, depth, dyadic=True):
    """Gets start and endpoints of decreasing width intervals.

    If we have a path of length L, this will give the successive indexes that split the path into size L/n where n
    increases until it hits the specified depth.

    Args:
        length (int): Length of the path.
        depth (int): How deep to go in the dyadic process.
        dyadic (bool): Set true for dyadic intervals.

    Returns:
        list: List of tuples where each tuple consists of (start_index, end_index).
    """
    # To hold start and end indexes
    idxs = []

    # Split the indexes
    full_idxs = np.arange(0, length)
    for i in range(1, depth + 1):
        end = 2 ** i if dyadic else i
        idxs.append([x[[0, -1]].tolist() for x in np.array_split(full_idxs, end)])
    idxs = [tuple(x) for l in idxs for x in l]

    return idxs



if __name__ == '__main__':
    dataset = UcrDataset(ds_name='Beef', multivariate=False)
    data = AddTime().transform(dataset.data)

    signatures = IntervalSignature(depth=3, logsig=False, interval_depth=4, dyadic=False).transform(data)