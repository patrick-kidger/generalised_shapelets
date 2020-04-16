import collections as co
import numpy as np
import pathlib
import sklearn.model_selection
import sktime.utils.load_data
import torch

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
    test_X, test_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
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

    train_dataloader = common.dataloader(train_dataset, batch_size=2048)
    val_dataloader = common.dataloader(val_dataset, batch_size=2048)
    test_dataloader = common.dataloader(test_dataset, batch_size=2048)

    num_classes = counter
    input_channels = train_X.size(-1)

    assert num_classes >= 2, "Have only {} classes.".format(num_classes)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels


def main(dataset_name,                        # dataset parameters
         missing_rate=0.,                     #
         noise_channels=0,                    #
         result_folder=None,                  # saving parameters
         result_subfolder='',                 #
         epochs=1000,                         # training parameters
         num_shapelets_per_class=3,           # model parameters
         num_shapelet_samples=None,           #
         discrepancy_fn='L2',                 #
         max_shapelet_length_proportion=1.0,  #
         lengths_per_shapelet=1,              #
         num_continuous_samples=None,         #
         metric_type='general',
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
                       dataset_name + '-' + result_subfolder,
                       epochs,
                       num_shapelets_per_class,
                       num_shapelet_samples,
                       discrepancy_fn,
                       max_shapelet_length_proportion,
                       lengths_per_shapelet,
                       num_continuous_samples,
                       metric_type,
                       ablation_pseudometric,
                       ablation_learntlengths,
                       ablation_similarreg,
                       old_shapelets)


def comparison_test():
    result_folder = 'uea_comparison'
    # Note that the most expensive datasets really will take a phenomenally long time to do, so be prepared to control-C
    # this at some point.
    for dataset_name in datasets_by_cost:
        pseudometric = dataset_name in large_datasets
        print("Starting comparison: L2, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder='L2',
             discrepancy_fn='L2',
             ablation_pseudometric=pseudometric)
        print("Starting comparison: L2_squared, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder='L2',
             discrepancy_fn='L2_squared',
             ablation_pseudometric=pseudometric)
        print("Starting comparison: logsig-3, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder='logsig-3',
             discrepancy_fn='logsig-3',
             ablation_pseudometric=pseudometric)
        print("Starting comparison: old-L2_squared, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder='old-L2_squared',
             discrepancy_fn='L2_squared',
             ablation_pseudometric=pseudometric,
             old_shapelets=True)


standard_dataset_names = ('JapaneseVowels', 'BasicMotions', 'FingerMovements')


def missing_rate_test():
    result_folder = 'uea_missingness'
    for dataset_name in standard_dataset_names:
        for missing_rate in (0.1, 0.3, 0.5):
            result_subfolder = str(int(missing_rate * 100))
            for discrepancy_fn in ('L2', 'L2_squared', 'logsig-3'):
                run_subfolder = result_subfolder + '-' + discrepancy_fn
                print("Starting comparison: " + run_subfolder)
                main(dataset_name,
                     result_folder=result_folder,
                     result_subfolder=run_subfolder,
                     missing_rate=missing_rate,
                     discrepancy_fn=discrepancy_fn)


def noise_test():
    result_folder = 'uea_noise'
    for dataset_name in standard_dataset_names:
        for noise_channels in (3, 9, 30):
            result_subfolder = str(noise_channels)
            for discrepancy_fn in ('L2',):
                for pseudometric in (True, False):
                    run_subfolder = result_subfolder + '-' + discrepancy_fn + '-' + str(pseudometric)
                    print("Starting comparison: " + run_subfolder)
                    main(dataset_name,
                         noise_channels=noise_channels,
                         result_folder=result_folder,
                         result_subfolder=run_subfolder,
                         discrepancy_fn=discrepancy_fn,
                         ablation_pseudometric=pseudometric)
                run_subfolder = result_subfolder + '-old-' + discrepancy_fn
                print("Starting comparison: " + run_subfolder)
                main(dataset_name,
                     noise_channels=noise_channels,
                     result_folder=result_folder,
                     result_subfolder=run_subfolder,
                     discrepancy_fn=discrepancy_fn,
                     old_shapelets=True)


def length_test():
    result_folder = 'uea_length'
    for dataset_name in standard_dataset_names:
        for discrepancy_fn in ('L2',):
            for learnt_lengths in (True, False):
                run_subfolder = discrepancy_fn + str(learnt_lengths)
                print("Starting comparison: " + run_subfolder)
                main(dataset_name,
                     result_folder=result_folder,
                     result_subfolder=run_subfolder,
                     discrepancy_fn=discrepancy_fn,
                     ablation_learntlengths=learnt_lengths)
