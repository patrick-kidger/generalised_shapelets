from torch.utils.data import Dataset
import torch
import signatory
from src.data.make_dataset import UcrDataset
from src.features.functions import pytorch_rolling
from src.features.signatures.augmentations import apply_augmentation_list


class ShapeletDataset(Dataset):
    """Dataset generator for useage in shapelet learning.

    The assumed form of the input data is to be of shape [N, L, C]. Given a window size, W, the data will be transformed
    onto shape [N, L-W, F] where F are some new features that may be the original points, signature values, wavelet
    basis coefficients, etc.
    """
    def __init__(self, data, labels, window_size):
        """
        Args:
            data (torch.Tensor): A tensor with dimensions [N, L, C].
            labels (torch.Tensor): A tensor of labels.
            window_size (int): The sub-interval window size.
        """
        self.labels = labels
        self.window_size = window_size

        self.data = self._init_data(data)
        self.shapelet_len = self.data.size(2)

    def roll_data(self, data, window_size):
        return pytorch_rolling(data, dimension=1, window_size=window_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def size(self, *args):
        return self.data.size(*args)


class PointsDataset(ShapeletDataset):
    """ Dataset for when we wish to consider the data with their point values. """
    def __init__(self, data, labels, window_size):
        super(PointsDataset, self).__init__(data, labels, window_size)

    def _init_data(self, data):
        # Unroll
        data_rolled = self.roll_data(data, self.window_size)

        # Convert [N, L-W, C, W] -> [N, L-W, C*W] and return
        data_out = data_rolled.reshape(data_rolled.size(0), data_rolled.size(1), -1)

        return data_out


class SigletDataset(ShapeletDataset):
    """Contains options for generating the signatures (over rolling windows) of the dataset.

    The input must be a path of shape [N, L, C]. First it is converted to a rolling path of shape [N, L-W, C, W], then
    it is reshaped to [N * (L-W), W, C], the log-signature is applied, and a final reshape gives a tensor of shape
    [N, L-W, SIG_DIM].
    """
    def __init__(self, data, labels, window_size, depth, aug_list=['addtime'], ds_length=None, num_window_size=None):
        self.depth = depth
        self.aug_list = aug_list
        self.ds_length = ds_length
        self.num_window_size = num_window_size
        self.max_window = 100
        super(SigletDataset, self).__init__(data, labels, window_size)


    def _init_data(self, data):
        s = []
        window_sizes = [int(x) for x in torch.linspace(5, min(100, self.ds_length), steps=5)]
        for window_size in window_sizes:
            # Unroll the data
            data_rolled = self.roll_data(data, window_size)

            # Reshapes so we can use signatory
            data_tricked = data_rolled.reshape(-1, data.size(2), window_size).transpose(1, 2)

            # Any augmentations
            data_augs = apply_augmentation_list(data_tricked, aug_list=self.aug_list)

            # Compute the signatures
            signatures = signatory.logsignature(data_augs, depth=self.depth)

            # Reshape to [N, L-W, F] and return
            signatures_untricked = signatures.reshape(data_rolled.size(0), data_rolled.size(1), -1)
            s.append(signatures_untricked)
        return torch.cat(s, axis=1)


if __name__ == '__main__':
    dataset = UcrDataset(ds_name='GunPoint')
    SigletDataset(dataset.data, dataset.labels, 40, depth=4)

