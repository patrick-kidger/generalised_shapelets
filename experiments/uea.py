import collections as co
import numpy as np
import pathlib
import sklearn.model_selection
import sktime.utils.load_data
import torch
import random

import common


here = pathlib.Path(__file__).resolve().parent


def _pad(channel, maxlen):
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[:channel.size(0)] = channel
    return out


valid_dataset_names = ('ArticularyWordRecognition',
                       'FaceDetection',
                       'NATOPS',
                       'AtrialFibrillation',
                       'FingerMovements',
                       'PEMS-SF',
                       'BasicMotions',
                       'HandMovementDirection',
                       'PenDigits',
                       'CharacterTrajectories',
                       'Handwriting',
                       'PhonemeSpectra',
                       'Cricket',
                       'Heartbeat',
                       'RacketSports',
                       'DuckDuckGeese',
                       'InsectWingbeat',
                       'SelfRegulationSCP1',
                       'EigenWorms',
                       'JapaneseVowels',
                       'SelfRegulationSCP2',
                       'Epilepsy',
                       'Libras',
                       'SpokenArabicDigits',
                       'ERing',
                       'LSST',
                       'StandWalkJump',
                       'EthanolConcentration',
                       'MotorImagery',
                       'UWaveGestureLibrary')

long_datasets = {'EigenWorms', 'MotorImagery', 'StandWalkJump', 'EthanolConcentration', 'Cricket', 'SelfRegulationSCP2'}

large_datasets = {'InsectWingbeat', 'ElectricDevices', 'PenDigits', 'SpokenArabicDigits', 'FaceDetection',
                  'PhonemeSpectra', 'LSST', 'UWaveGestureLibrary', 'CharacterTrajectories'}


# Ordered by chanels * dataset size * num_classes * length ** 2, i.e. the cost of evaluating shaplets on them.
datasets_by_cost = ('ERing',
                    'RacketSports',
                    'PenDigits',
                    'BasicMotions',
                    'Libras',
                    'JapaneseVowels',
                    'AtrialFibrillation',
                    'FingerMovements',
                    'NATOPS',
                    'Epilepsy',
                    'LSST',
                    'Handwriting',
                    'UWaveGestureLibrary',
                    'StandWalkJump',
                    'HandMovementDirection',
                    'ArticularyWordRecognition',
                    'SelfRegulationSCP1',
                    'CharacterTrajectories',
                    'SelfRegulationSCP2',
                    'Heartbeat',
                    'FaceDetection',
                    'SpokenArabicDigits',
                    'EthanolConcentration',
                    'Cricket',
                    'DuckDuckGeese',
                    'PEMS-SF',
                    'InsectWingbeat',
                    'PhonemeSpectra',
                    'MotorImagery',
                    'EigenWorms')


def get_data(dataset_name, missing_rate, noise_channels):
    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_filename = here / 'data' / 'UEA' / 'Multivariate_ts' / dataset_name / dataset_name
    train_X, train_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
    test_X, test_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TEST.ts')
    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    amount_train = train_X.shape[0]
    all_X = np.concatenate((train_X, test_X), axis=0)
    all_y = np.concatenate((train_y, test_y), axis=0)

    lengths = torch.tensor([len(Xi[0]) for Xi in all_X])
    maxlen = lengths.max()
    # Each channel is a pandas.core.series.Series object of length corresponding to the length of the time series
    all_X = torch.stack([torch.stack([_pad(channel, maxlen) for channel in batch], dim=0) for batch in all_X], dim=0)
    all_X = all_X.transpose(-1, -2)

    if noise_channels != 0:
        generator = torch.Generator().manual_seed(45678)
        noise_X = torch.randn(all_X.size(0), all_X.size(1), noise_channels, dtype=all_X.dtype, generator=generator)
        all_X = torch.cat([all_X, noise_X], dim=2)

    times = torch.linspace(0, all_X.size(1) - 1, all_X.size(1), dtype=all_X.dtype)

    # Handle missingness: remove values and replace them with the linear interpolation of the non-missing points.
    if missing_rate > 0:
        generator = torch.Generator().manual_seed(56789)
        for batch_index in range(all_X.size(0)):
            for channel_index in range(all_X.size(2)):
                randperm = torch.randperm(all_X.size(1) - 2, generator=generator) + 1  # keep the start and end
                removed_points = randperm[:int(all_X.size(1) * missing_rate)].sort().values

                prev_removed_point = removed_points[0]
                prev_unremoved_point = prev_removed_point - 1
                prev_unremoved_points = [prev_unremoved_point]
                for removed_point in removed_points[1:]:
                    if prev_removed_point != removed_point - 1:
                        prev_unremoved_point = removed_point - 1
                    prev_removed_point = removed_point
                    prev_unremoved_points.append(prev_unremoved_point)

                next_removed_point = removed_points[-1]
                next_unremoved_point = next_removed_point + 1
                next_unremoved_points = [next_unremoved_point]
                for removed_point in reversed(removed_points[:-1]):
                    if next_removed_point != removed_point + 1:
                        next_unremoved_point = removed_point + 1
                    next_removed_point = removed_point
                    next_unremoved_points.append(next_unremoved_point)
                next_unremoved_points = reversed(next_unremoved_points)
                for prev_unremoved_point, removed_point, next_unremoved_point in zip(prev_unremoved_points,
                                                                                     removed_points,
                                                                                     next_unremoved_points):
                    stream = all_X[batch_index, :, channel_index]
                    prev_stream = stream[prev_unremoved_point]
                    next_stream = stream[next_unremoved_point]
                    prev_time = times[prev_unremoved_point]
                    next_time = times[next_unremoved_point]
                    time = times[removed_point]
                    ratio = (time - prev_time) / (next_time - prev_time)
                    stream[removed_point] = prev_stream + ratio * (next_stream - prev_stream)

    # Now fix the labels to be integers from 0 upwards
    targets = co.OrderedDict()
    counter = 0
    for yi in all_y:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    all_y = torch.tensor([targets[yi] for yi in all_y])

    # use original train/test splits
    trainval_X, test_X = all_X[:amount_train], all_X[amount_train:]
    trainval_y, test_y = all_y[:amount_train], all_y[amount_train:]

    train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(trainval_X, trainval_y,
                                                                              train_size=0.8,
                                                                              random_state=0,
                                                                              shuffle=True,
                                                                              stratify=trainval_y)

    val_X = common.normalise_data(val_X, train_X)
    test_X = common.normalise_data(test_X, train_X)
    train_X = common.normalise_data(train_X, train_X)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

    train_dataloader = common.dataloader(train_dataset, batch_size=1024)
    val_dataloader = common.dataloader(val_dataset, batch_size=1024)
    test_dataloader = common.dataloader(test_dataset, batch_size=1024)

    num_classes = counter
    input_channels = train_X.size(-1)

    assert num_classes >= 2, "Have only {} classes.".format(num_classes)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels


def _subfolder(dataset_name, dataset_detail, result_subfolder):
    return dataset_name + dataset_detail + '-' + result_subfolder


def main(dataset_name,                        # dataset parameters
         missing_rate=0.,                     #
         noise_channels=0,                    #
         result_folder=None,                  # saving parameters
         result_subfolder='',                 #
         dataset_detail='',                   #
         epochs=250,                          # training parameters
         num_shapelets_per_class=3,           # model parameters
         num_shapelet_samples=None,           #
         discrepancy_fn='L2',                 #
         max_shapelet_length_proportion=1.0,  #
         num_continuous_samples=None,         #
         ablation_pseudometric=True,          # For ablation studies
         ablation_learntlengths=True,         #
         ablation_similarreg=True,            #
         old_shapelets=False):                # Whether to toggle off all of our innovations and use old-style shapelets

    times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels = get_data(dataset_name,
                                                                                                     missing_rate,
                                                                                                     noise_channels)

    return common.main(times,
                       train_dataloader,
                       val_dataloader,
                       test_dataloader,
                       num_classes,
                       input_channels,
                       result_folder,
                       _subfolder(dataset_name, dataset_detail, result_subfolder),
                       epochs,
                       num_shapelets_per_class,
                       num_shapelet_samples,
                       discrepancy_fn,
                       max_shapelet_length_proportion,
                       num_continuous_samples,
                       ablation_pseudometric,
                       ablation_learntlengths,
                       ablation_similarreg,
                       old_shapelets)


def hyperparameter_search():
    result_folder = 'uea_hyperparameter_search'
    for dataset_name in datasets_by_cost[:9]:
        for num_shapelets_per_class in (2, 3, 5):
            for max_shapelet_length_proportion in (0.3, 0.5, 1.0):
                result_subfolder = 'old-' + str(num_shapelets_per_class) + '-' + str(max_shapelet_length_proportion)
                print("Starting comparison: " + dataset_name + '-' + result_subfolder)
                main(dataset_name,
                     result_folder=result_folder,
                     result_subfolder=result_subfolder,
                     num_shapelets_per_class=num_shapelets_per_class,
                     max_shapelet_length_proportion=max_shapelet_length_proportion,
                     old_shapelets=True)


def missing_and_length_test():
    seed = 5678
    for i in range(3):
        result_folder = 'uea_missing_and_length'
        for dataset_name in ('JapaneseVowels', 'BasicMotions', 'FingerMovements'):
            for missing_rate in (0.1, 0.3, 0.5):
                for discrepancy_fn in ('L2', 'logsig-3'):
                    for learntlengths in (True, False):
                        seed = common.handle_seeds(seed)
                        result_subfolder = discrepancy_fn + '-' + str(learntlengths)
                        dataset_detail = str(int(missing_rate * 100))
                        full_result_subfolder = _subfolder(dataset_name, dataset_detail, result_subfolder)
                        if common.assert_not_done(result_folder, full_result_subfolder, n_done=3):
                            print("Starting comparison: " + full_result_subfolder)
                            main(dataset_name,
                                 result_folder=result_folder,
                                 result_subfolder=result_subfolder,
                                 dataset_detail=dataset_detail,
                                 missing_rate=missing_rate,
                                 discrepancy_fn=discrepancy_fn,
                                 ablation_learntlengths=learntlengths)


def comparison_test():
    seed = 5678
    for i in range(3):
        result_folder = 'uea_comparison'

        for dataset_name in datasets_by_cost[:9]:
            for discrepancy_fn in ('L2', 'logsig-3'):
                seed = common.handle_seeds(seed)
                result_subfolder = discrepancy_fn
                dataset_detail = ''
                full_result_subfolder = _subfolder(dataset_name, dataset_detail, result_subfolder)
                if common.assert_not_done(result_folder, full_result_subfolder, n_done=3):
                    print("Starting comparison: " + full_result_subfolder)
                    main(dataset_name,
                         result_folder=result_folder,
                         result_subfolder=result_subfolder,
                         discrepancy_fn=discrepancy_fn)

            seed = common.handle_seeds(seed)
            result_subfolder = 'old'
            dataset_detail = ''
            full_result_subfolder = _subfolder(dataset_name, dataset_detail, result_subfolder)
            if common.assert_not_done(result_folder, full_result_subfolder, n_done=3):
                print("Starting comparison: " + full_result_subfolder)
                main(dataset_name,
                     result_folder=result_folder,
                     result_subfolder=result_subfolder,
                     old_shapelets=True)
