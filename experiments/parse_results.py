import collections as co
import json
import os
import pathlib
import statistics
import sys
import pickle
import pandas as pd

here = pathlib.Path(__file__).resolve().parent


def get(foldername):
    for filename in os.listdir(foldername):
        if 'model' not in filename:
            with open(foldername / filename, 'r') as f:
                content = json.load(f)
            yield content['test_metrics']['accuracy']


def _create_folder_if_not_exist(filename):
    """ Makes a folder if the folder component of the filename does not already exist. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def save_pickle(obj, filename, protocol=4, create_folder=True):
    """ Basic pickle/dill dumping.

    Given a python object and a filename, the method will save the object under that filename.

    Args:
        obj (python object): The object to be saved.
        filename (str): Location to save the file.
        protocol (int): Pickling protocol (see pickle docs).
        create_folder (bool): Set True to create the folder if it does not already exist.

    Returns:
        None
    """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol)


def main(dataset_folder, save_df=True):
    dataset_folder = here / 'results' / dataset_folder

    has_no_dash = False
    for foldername in os.listdir(dataset_folder):
        if '-' not in foldername:
            has_no_dash = True
            break

    means = {}
    stds = {}
    for foldername in os.listdir(dataset_folder):
        if has_no_dash:
            dataset_name = str(dataset_folder.name)
            setting = foldername
        else:
            foldername_split = foldername.split('-', maxsplit=1)
            dataset_name = foldername_split[0]
            setting = foldername_split[1]
        if dataset_name not in means:
            means[dataset_name] = {}
        if dataset_name not in stds:
            stds[dataset_name] = {}
        values = list(get(dataset_folder / foldername))
        means[dataset_name][setting] = statistics.mean(values)
        if len(values) > 1:
            stds[dataset_name][setting] = statistics.stdev(values)

    headings = co.OrderedDict()  # Used as an ordered set here
    for mean in means.values():
        for setting in mean:
            headings[setting] = None

    means = co.OrderedDict(sorted(means.items(), key=lambda x: x[0]))

    dataset_column_width = max(len(dataset_name) for dataset_name in means) + 1
    column_width = max(max(len(heading) for heading in headings), 12)
    print(' ' * dataset_column_width, end='')
    for heading in headings:
        print('| {{:{}}} '.format(column_width).format(heading), end='')
    print('')
    print('-' * dataset_column_width, end='')
    for _ in headings:
        print('+' + '-' * (column_width + 2), end='')
    print('')
    for dataset_name, mean in means.items():
        std = stds[dataset_name]
        print('{{:{}}}'.format(dataset_column_width).format(dataset_name), end='')
        for heading in headings:
            mean_print = '{:.3f}'.format(mean[heading]) if heading in mean else '  -  '
            std_print = '~{:.3f}'.format(std[heading]) if heading in std else '      '
            print('|' + ' ' * (column_width - 10) + mean_print + std_print + ' ', end='')
        print('')
    print('-' * dataset_column_width, end='')
    for _ in headings:
        print('+' + '-' * (column_width + 2), end='')
    print('')
    print('{{:{}}}'.format(dataset_column_width).format('Wins'), end='')

    wins = {k: 0 for k in headings}
    for dataset_name, mean in means.items():
        max_regularisation_name = None
        max_regularisation_value = -1.
        for regularisation_name, regularisation_value in mean.items():
            if regularisation_value > max_regularisation_value:
                max_regularisation_value = regularisation_value
                max_regularisation_name = regularisation_name
        wins[max_regularisation_name] += 1

    for heading in headings:
        print('| {{:{}}} '.format(column_width).format(wins[heading]), end='')
    print('')

    # Generate and return a dataframe
    if save_df:
        folder_name = dataset_folder.name
        df = pd.DataFrame.from_dict(means).T
        save_pickle(df, './results/dataframes/{}.pkl'.format(folder_name))


if __name__ == '__main__':
    assert len(sys.argv) == 2
    dataset = sys.argv[1]
    main(dataset)
