import collections as co
import json
import os
import pathlib
import statistics
import sys


here = pathlib.Path(__file__).resolve().parent


def get(foldername):
    for filename in os.listdir(foldername):
        with open(foldername / filename, 'r') as f:
            content = json.load(f)
        yield content['test_metrics']['accuracy']


def main(dataset_folder):
    dataset_folder = here / 'results' / dataset_folder

    means = {}
    stds = {}
    for foldername in os.listdir(dataset_folder):
        foldername_split = foldername.split('-', maxsplit=1)
        dataset_name = foldername_split[0]
        regularisation = foldername_split[1]
        if dataset_name not in means:
            means[dataset_name] = {}
            stds[dataset_name] = {}
        values = list(get(dataset_folder / foldername))
        means[dataset_name][regularisation] = statistics.mean(values)
        if len(values) > 1:
            stds[dataset_name][regularisation] = statistics.stdev(values)

    headings = co.OrderedDict()  # Used as an ordered set here
    for mean in means.values():
        for key in mean:
            headings[key] = None

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
    for (dataset_name, mean), std in zip(means.items(), stds.values()):
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


if __name__ == '__main__':
    assert len(sys.argv) == 2
    dataset = sys.argv[1]
    main(dataset)
