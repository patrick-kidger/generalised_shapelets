import collections as co
import copy
import json
import math
import os
import pathlib
import statistics
import sys
import pandas as pd


here = pathlib.Path(__file__).resolve().parent


def get(foldername):
    for filename in os.listdir(foldername):
        if 'model' not in filename:
            with open(foldername / filename, 'r') as f:
                content = json.load(f)
            yield content['test_metrics']['accuracy']


def main(dataset_folder):
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
        max_regularisation_names = set()
        max_regularisation_value = -1.
        for regularisation_name, regularisation_value in mean.items():
            if regularisation_value > max_regularisation_value:
                max_regularisation_value = regularisation_value
                max_regularisation_names = {regularisation_name}
            elif regularisation_value == max_regularisation_value:
                max_regularisation_names.add(regularisation_name)
        for name in max_regularisation_names:
            wins[name] += 1

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

    # Slight column name hack
    means.columns = pd.MultiIndex.from_arrays([['\\textbf{Discrepancy}'] * n_cols, means.columns])

    if stds is not None:
        stds = stds.round(round)
        zfill = lambda x: x.astype(str).str.ljust(width=round + 2, fillchar='0')
        new_means = copy.deepcopy(means)
        for col in means.columns:
            new_means[col] = zfill(means[col]) + ' $\pm$ ' + zfill(stds[col])
        means = new_means

    # Convert onto a win frame
    column_format = 'l' + 'c' * n_cols
    top_section = (means.to_latex(float_format="%.3f", column_format=column_format, na_rep='-', escape=False, index_names=True)
                        .split('\\\\\n\\bottom')[0])

    # Make column heading centreed
    top_section = top_section.replace('{l}{\\textbf{Discrepancy}}', '{c}{\\textbf{Discrepancy}}')
    # Add dataset col name
    top_split = top_section.split('\\\\\n{} ')
    discrepancy_string = '\\\\\n\\textbf{Dataset} '
    top_section_discrepancy = top_split[0] + discrepancy_string + top_split[1]

    bottom_section = 'Wins' + wins.to_latex().split('Wins')[1]
    tex_string = top_section_discrepancy + '\\\\ \n\midrule\n' + bottom_section

    # Write the table
    with open(save_loc, "w") as file:
        file.write(tex_string)


if __name__ == '__main__':
    assert len(sys.argv) in (2, 3)
    dataset = sys.argv[1]
    means, wins, stds = main(dataset)

    if len(sys.argv) == 3 and sys.argv[2] == '--save':
        # Save the table to results
        save_loc = '../paper/results/data/{}.tex'.format(dataset)
        generate_table(save_loc, means, wins, stds)
