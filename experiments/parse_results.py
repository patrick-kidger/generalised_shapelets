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

    results = {}
    for foldername in os.listdir(dataset_folder):
        foldername_split = foldername.split('-', maxsplit=1)
        dataset_name = foldername_split[0]
        regularisation = foldername_split[1]
        if dataset_name not in results:
            results[dataset_name] = {}
        results[dataset_name][regularisation] = statistics.mean(get(dataset_folder / foldername))

    headings = co.OrderedDict()  # Used as an ordered set here
    for result in results.values():
        for key in result:
            headings[key] = None

    results = co.OrderedDict(sorted(results.items(), key=lambda x: x[0]))

    dataset_column_width = max(len(dataset_name) for dataset_name in results) + 1
    column_width = max(max(len(heading) for heading in headings), 5)
    print(' ' * dataset_column_width, end='')
    for heading in headings:
        print('| {{:{}}} '.format(column_width).format(heading), end='')
    print('')
    print('-' * dataset_column_width, end='')
    for _ in headings:
        print('+' + '-' * (column_width + 2), end='')
    print('')
    for dataset_name, result in results.items():
        print('{{:{}}}'.format(dataset_column_width).format(dataset_name), end='')
        for heading in headings:
            print('| {{:{}.3f}} '.format(column_width).format(result.get(heading, float('nan'))), end='')
        print('')
    print('-' * dataset_column_width, end='')
    for _ in headings:
        print('+' + '-' * (column_width + 2), end='')
    print('')
    print('{{:{}}}'.format(dataset_column_width).format('Wins'), end='')

    wins = {k: 0 for k in headings}
    for dataset_name, result in results.items():
        max_regularisation_name = None
        max_regularisation_value = -1.
        for regularisation_name, regularisation_value in result.items():
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
