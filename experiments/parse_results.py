import collections as co
import json
import math
import os
import pathlib
import statistics
import sys
import pickle
import pandas as pd
from copy import deepcopy

here = pathlib.Path(__file__).resolve().parent


def get(foldername):
    for filename in os.listdir(foldername):
        if 'model' not in filename:
            with open(foldername / filename, 'r') as f:
                content = json.load(f)
            yield content['test_metrics']['accuracy']


def write_string(save_loc, string):
    with open(save_loc, "w") as file:
        file.write(string)
    file.close()


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


def load_pickle(filename):
    """ Basic dill/pickle load function.

    Args:
        filename (str): Location of the object.

    Returns:
        python object: The loaded object.
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj


def main(dataset_folder, save_df=True):
    dataset_folder = here / 'results' / dataset_folder

    has_no_dash = False
    for foldername in os.listdir(dataset_folder):
        if '-' not in foldername:
            has_no_dash = True
            break

    values = {}
    min_num_observations = math.inf
    for foldername in os.listdir(dataset_folder):
        if has_no_dash:
            dataset_name = str(dataset_folder.name)
            setting = foldername
        else:
            foldername_split = foldername.split('-', maxsplit=1)
            dataset_name = foldername_split[0]
            setting = foldername_split[1]
        if dataset_name not in values:
            values[dataset_name] = {}
        value = list(get(dataset_folder / foldername))
        min_num_observations = min(min_num_observations, len(value))
        values[dataset_name][setting] = value

    means = {k: {} for k in values}
    stds = {k: {} for k in values}
    for dataset_name, settings in values.items():
        for setting, value in settings.items():
            means[dataset_name][setting] = statistics.mean(value[:min_num_observations])
            if len(value) > 1:
                stds[dataset_name][setting] = statistics.stdev(value[:min_num_observations])

    headings = co.OrderedDict()  # Used as an ordered set here
    for mean in means.values():
        for setting in mean:
            headings[setting] = None

    means = co.OrderedDict(sorted(means.items(), key=lambda x: x[0]))

    print('Num observations: ' + str(min_num_observations))
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

    return means, wins, stds


def generate_dataframes(means, wins, stds):
    """ Generates dataframes from means, stds and wins. """
    # Save mean
    means = pd.DataFrame.from_dict(means).T

    # Save wins
    wins = pd.DataFrame.from_dict(wins, orient='index')
    wins = wins.T
    wins.index = ['Wins']
    wins = wins[means.columns]

    stds = pd.DataFrame.from_dict(stds).T
    if stds.shape[1] > 0:
        stds = stds[means.columns]
    else:
        stds = None

    return means, wins, stds


def generate_table(save_loc, means, wins, stds, round=3):
    """ Generates latex ready tables from means, wins, stds. """
    means, wins, stds = generate_dataframes(means, wins, stds)

    n_cols = len(means.columns)
    means = means.round(round)

    if stds is not None:
        stds = stds.round(round)
        zfill = lambda x: x.astype(str).str.ljust(width=round + 2, fillchar='0')
        new_means = deepcopy(means)
        for col in means.columns:
            new_means[col] = zfill(means[col]) + ' $\pm$ ' + zfill(stds[col])
        means = new_means

    # Convert onto a win frame
    column_format = 'l' + 'c' * n_cols
    top_section = (means.to_latex(float_format="%.3f", column_format=column_format, na_rep='-', escape=False)
                        .split('\\\\\n\\bottom')[0])
    bottom_section = 'Wins' + wins.to_latex().split('Wins')[1]
    tex_string = top_section + '\\\\ \n\midrule\n' + bottom_section

    # Write the table
    write_string(save_loc, tex_string)



if __name__ == '__main__':
    assert len(sys.argv) == 2
    dataset = sys.argv[1]
    means, wins, stds = main(dataset)

    # Save the table to results
    save_loc = '../paper/results/data/{}.tex'.format(dataset)
    generate_table(save_loc, means, wins, stds)
