"""
Generic structure for loading and holding datasets that can be used model running.
"""
from jamesshapelets.definitions import *
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset


class UcrDataset(Dataset):
    """ Simple structure for working with UCR time-series data. """
    def __init__(self, ds_name, multivariate=False):
        """
        Args:
            ds_name (str): Name of the dataset to be loaded.
            multivariate (bool): Set True if the dataset is a multivariate dataset.
        """
        self.ds_name = ds_name
        self.multivariate = multivariate

        # Get the data and labels
        self.data, self.labels, self.original_idxs, self.info, self.n_classes = get_dataset(ds_name, multivariate=multivariate)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def size(self, *args):
        """ Copy of torch size. """
        return self.data.size(*args)

    def get_original_train_test(self):
        """ Returns the original train/test split as TimeSeriesDatasets. """
        # Make copies
        train_ds, test_ds = deepcopy(self), deepcopy(self)

        # Switch out the data
        train_ds.data, train_ds.labels = self.data[self.original_idxs[0]], self.labels[self.original_idxs[0]]
        test_ds.data, test_ds.labels = self.data[self.original_idxs[1]], self.labels[self.original_idxs[1]]

        return train_ds, test_ds

    def to_ml(self):
        """ Returns (data, labels) format, ready for use in machine-learning. """
        return self.data, self.labels


def get_dataset(ds_name, multivariate=False):
    """Gets a dataset with a given name.

    Args:
        ds_name (str): Name of the dataset.
        multivariate (bool): Set True if multivariate data.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, info): Data, labels, original split idxs, and dataset information.

    """
    # Get save_dir
    uv_mv = 'multivariate' if multivariate else 'univariate'
    save_dir = DATA_DIR + '/interim/{}/{}'.format(uv_mv, ds_name)

    # Load
    data = load_pickle(save_dir + '/data.pkl')
    labels = load_pickle(save_dir + '/labels.pkl').view(-1)
    n_classes = len(np.unique(labels))
    if n_classes > 2:
        labels = labels.long()

    # Get original train/test indexes
    original_idxs = load_pickle(save_dir + '/original_idxs.pkl')

    # Get additional information
    summaries = load_pickle(DATA_DIR + '/interim/summaries/{}.pkl'.format(uv_mv))
    if ds_name in summaries.index:
        summaries = summaries.loc[ds_name]
    else:
        summaries = None
    info = {
        'summary': summaries
    }

    return data, labels, original_idxs, info, n_classes


if __name__ == '__main__':
    UcrDataset(ds_name='Coffee')