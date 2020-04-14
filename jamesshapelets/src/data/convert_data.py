"""
convert_data.py
============================
This module converts arff data (format used by UCR) to torch tensors. The data is taken from data/raw/ and saved in
data/interim using an identical directory structure as in /raw.
"""
from definitions import *
import numpy as np
import torch
from tqdm import tqdm
from sktime.utils.load_data import load_from_arff_to_dataframe
from sklearn.preprocessing import LabelEncoder


def create_torch_data(train_file, test_file):
    """Creates torch tensors for test and training from the UCR arff format.

    Args:
        train_file (str): The location of the training data arff file.
        test_file (str): The location of the testing data arff file.

    Returns:
        data_train, data_test, labels_train, labels_test: All as torch tensors.
    """
    # Get arff format
    train_data, train_labels = load_from_arff_to_dataframe(train_file)
    test_data, test_labels = load_from_arff_to_dataframe(test_file)

    def convert_data(data):
        # Expand the series to numpy
        data_expand = data.applymap(lambda x: x.values).values
        # Single array, then to tensor
        data_numpy = np.stack([np.vstack(x).T for x in data_expand])
        tensor_data = torch.Tensor(data_numpy)
        return tensor_data

    train_data, test_data = convert_data(train_data), convert_data(test_data)

    # Encode labels as often given as strings
    encoder = LabelEncoder().fit(train_labels)
    train_labels, test_labels = encoder.transform(train_labels), encoder.transform(test_labels)
    train_labels, test_labels = torch.Tensor(train_labels), torch.Tensor(test_labels)

    return train_data, test_data, train_labels, test_labels


def convert_all_files(arff_folder, subfolder):
    """ Convert all files from a given /raw/{subfolder} into torch data to be stored in /interim. """
    # Time for a big for loop
    for ds_name in tqdm([x for x in os.listdir(arff_folder) if os.path.isdir(arff_folder + '/' + x)]):
        # File locations
        train_file = arff_folder + '/{}/{}_TRAIN.arff'.format(ds_name, ds_name)
        test_file = arff_folder + '/{}/{}_TEST.arff'.format(ds_name, ds_name)

        # Ready save dir
        save_dir = DATA_DIR + '/interim/{}/{}'.format(subfolder, ds_name)

        # If files don't exist, skip.
        if any([x.split('/')[-1] not in os.listdir(arff_folder + '/{}'.format(ds_name)) for x in (train_file, test_file)]):
            print('No files found for folder: {}'.format(ds_name))
            continue
        elif os.path.isdir(save_dir):
            print('Files already exist for: {}'.format(ds_name))
            continue
        else:
            train_data, test_data, train_labels, test_labels = create_torch_data(train_file, test_file)

            # Compile train and test data together
            data = torch.cat([train_data, test_data])
            labels = torch.cat([train_labels, test_labels])

            # Save original train test indexes in case we wish to use original splits
            original_idxs = (np.arange(0, train_data.size(0)), np.arange(train_data.size(0), data.size(0)))

            # Save data
            save_pickle(data, save_dir + '/data.pkl')
            save_pickle(labels, save_dir + '/labels.pkl')
            save_pickle(original_idxs, save_dir + '/original_idxs.pkl')



if __name__ == '__main__':
    # Convert
    subfolders = ['univariate', 'multivariate']
    for subfolder in subfolders:
        arff_folder = DATA_DIR + '/raw/{}'.format(subfolder)
        print(arff_folder)
        convert_all_files(arff_folder, subfolder)

