import collections as co
import numpy as np
import pathlib
import sklearn.model_selection
import sktime.utils.load_data
import torch
import torchshapelets

import common


here = pathlib.Path(__file__).resolve().parent


def _pad(channel, maxlen):
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[:channel.size(0)] = channel
    return out


valid_dataset_names = {'ArticularyWordRecognition',
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
                       'UWaveGestureLibrary'}


def get_data(dataset_name, missing_rate, device, noise_channels):
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
        noise_X = torch.randn(all_X.size(0), all_X.size(1), noise_channels, dtype=all_X.dtype, device=all_X.device,
                              generator=generator)
        all_X = torch.cat([all_X, noise_X], dim=2)

    times = torch.linspace(0, all_X.size(1) - 1, all_X.size(1), dtype=all_X.dtype, device=device)

    # Handle missingness
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

    train_X = common.normalise_data(train_X, train_X)
    val_X = common.normalise_data(val_X, train_X)
    test_X = common.normalise_data(test_X, train_X)

    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)
    test_X = test_X.to(device)
    test_y = test_y.to(device)

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


def main(dataset_name, missing_rate,           # dataset parameters
         result_folder, result_subfolder,      # saving parameters
         noise_channels=0,                     # also a dataset parameter
         epochs=100, device='cpu',             # training parameters
         num_shapelets_per_class=3,            # model parameters
         num_shapelet_samples=None,            #
         discrepancy_fn='L2',                  #
         max_shapelet_length_proportion=None,  #
         num_continuous_samples=None,          #
         ablation_init=True,                   # For ablation studies
         ablation_log=True,                    #
         ablation_pseudometric=True,           #
         ablation_learntlengths=True,          #
         ablation_similarreg=True,             #
         ablation_lengthreg=True,              #
         ablation_pseudoreg=True,              #
         old_shapelets=False):                 # Whether to toggle off all of our innovations and use old-style
                                               # shapelets.

    (times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels) = get_data(dataset_name,
                                                                                                       missing_rate,
                                                                                                       device,
                                                                                                       noise_channels)

    assert times.is_floating_point(), "Whoops, times isn't floating point."

    if old_shapelets:
        num_continuous_samples = times.size(0)

    # Select some sensible options based on the length of the dataset
    timespan = times[-1] - times[0]
    if max_shapelet_length_proportion is None:
        max_shapelet_length_proportion = min((10 / timespan).sqrt(), 1)
    max_shapelet_length = timespan * max_shapelet_length_proportion
    if num_shapelet_samples is None:
        num_shapelet_samples = int(max_shapelet_length_proportion * times.size(0))
    if num_continuous_samples is None:
        num_continuous_samples = times.size(0)

    if discrepancy_fn == 'L2':
        discrepancy_fn = torchshapelets.L2Discrepancy(in_channels=input_channels, pseudometric=ablation_pseudometric)
    elif discrepancy_fn == 'L2_squared':
        def discrepancy_fn(times, path, shapelet):
            return ((path - shapelet) ** 2).sum(dim=(-1, -2))
        discrepancy_fn.parameters = lambda: []
    elif discrepancy_fn == 'DTW':
        def discrepancy_fn(times, path, shapelet):
            memo = [[torch.tensor(float('inf'), dtype=path.dtype, device=path.device)
                     for _ in range(path.size(-2) + 1)]
                    for _ in range(shapelet.size(-2) + 1)]
            memo[0][0] = torch.tensor(0, dtype=path.dtype, device=path.device)
            for i in range(path.size(-2)):
                for j in range(shapelet.size(-2)):
                    cost = (path[..., i, :] - shapelet[j, :]).norm(dim=-1)
                    memo[i + 1][j + 1] = cost + torch.min(torch.min(memo[i][j + 1], memo[i + 1][j]), memo[i][j])
            return memo[-1][-1]
        discrepancy_fn.parameters = lambda: []
    elif 'logsig' in discrepancy_fn:
        # expects e.g. 'logsig-4'
        split_desc = discrepancy_fn.split('-')
        assert len(split_desc) == 2
        assert split_desc[0] == 'logsig'
        depth = int(split_desc[1])
        discrepancy_fn = torchshapelets.LogsignatureDiscrepancy(in_channels=input_channels, depth=depth,
                                                                pseudometric=ablation_pseudometric)

    num_shapelets = num_shapelets_per_class * num_classes

    model = common.LinearShapeletTransform(in_channels=input_channels,
                                           out_channels=num_classes,
                                           num_shapelets=num_shapelets,
                                           num_shapelet_samples=num_shapelet_samples,
                                           discrepancy_fn=discrepancy_fn,
                                           max_shapelet_length=max_shapelet_length,
                                           num_continuous_samples=num_continuous_samples,
                                           ablation_log=ablation_log)

    if old_shapelets:
        del model.shapelet_transform.lengths
        model.shapelet_transform.register_buffer('lengths', torch.full_like(model.shapelet_transform.lengths,
                                                                            max_shapelet_length))
    else:
        if ablation_init:
            sample_batch = common.get_sample_batch(train_dataloader, num_shapelets_per_class, num_shapelets)
            model.set_shapelets(times.to('cpu'), sample_batch.to('cpu'))  # smart initialisation of shapelets
        if ablation_learntlengths:
            model.shapelet_transform.lengths.requires_grad_(False)

    if num_classes == 2:
        model = common.SqueezeEnd(model)
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    else:
        loss_fn = torch.nn.functional.cross_entropy
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    history = common.train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, epochs, num_classes,
                                ablation_similarreg, ablation_lengthreg, ablation_pseudoreg)
    results = common.evaluate_model(train_dataloader, val_dataloader, test_dataloader, model, times, loss_fn, history,
                                    num_classes)

    common.save_results(result_folder, result_subfolder, results)
    return results


def comparison_test(group=None):
    if group is None:
        valid_datasets = valid_dataset_names
        invalid_datasets = ('EigenWorms', 'InsectWingbeat')
        pseudometric = True
    elif group == 'small':
        valid_datasets = valid_dataset_names
        invalid_datasets = ('EigenWorms', 'InsectWingbeat',  # super huge, omit always
                            # Long length
                            'Cricket', 'EthanolConcentration', 'MotorImagery', 'SelfRegulationSCP2', 'StandWalkJump',
                            # Large dataset
                            'CharacterTrajectories', 'FaceDetection', 'LSST', 'PenDigits', 'Phoneme',
                            'SpokenArabicDigits')
        pseudometric = False
    elif group == 'longlength':
        valid_datasets = ('Cricket', 'EthanolConcentration', 'MotorImagery', 'SelfRegulationSCP2', 'StandWalkJump')
        invalid_datasets = ()
        pseudometric = True
    elif group == 'largedataset':
        valid_datasets = ('CharacterTrajectories', 'FaceDetection', 'LSST', 'PenDigits', 'Phoneme',
                          'SpokenArabicDigits')
        invalid_datasets = ()
        pseudometric = True
    else:
        raise ValueError
    result_folder = 'comparison'
    for dataset_name in valid_datasets:
        if dataset_name in invalid_datasets:
            continue
        print("Starting comparison: L2, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name+'-L2',
             missing_rate=0.,
             discrepancy_fn='L2',
             old_shapelets=False,
             ablation_pseudometric=pseudometric)
        print("Starting comparison: L2_squared, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-L2_squared',
             missing_rate=0.,
             num_shapelets_per_class=3,
             discrepancy_fn='L2_squared',
             old_shapelets=False,
             ablation_pseudometric=pseudometric)
        print("Starting comparison: logsig-3, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-logsig-3',
             missing_rate=0.,
             discrepancy_fn='logsig-3',
             old_shapelets=False,
             ablation_pseudometric=pseudometric)
        print("Starting comparison: old-L2_squared, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-old-L2_squared',
             missing_rate=0.,
             num_shapelets_per_class=3,
             discrepancy_fn='L2_squared',
             old_shapelets=True,
             ablation_pseudometric=pseudometric)
        print("Starting comparison: old-DTW, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-old-DTW',
             missing_rate=0.,
             num_shapelets_per_class=3,
             discrepancy_fn='DTW',
             old_shapelets=True,
             ablation_pseudometric=pseudometric)


standard_dataset_names = ('LSST', 'SpokenArabicDigits', 'FaceDetection')


def missing_rate_test():
    result_folder = 'missingness'
    for dataset_name in standard_dataset_names:
        for missing_rate in (0.3, 0.5, 0.7):
            result_subfolder = dataset_name + str(int(missing_rate * 100))
            print("Starting comparison: L2, " + result_subfolder)
            main(dataset_name,
                 result_folder=result_folder,
                 result_subfolder=result_subfolder + '-L2',
                 missing_rate=missing_rate,
                 discrepancy_fn='L2',
                 old_shapelets=False)
            print("Starting comparison: logsig-3, " + result_subfolder)
            main(dataset_name,
                 result_folder=result_folder,
                 result_subfolder=result_subfolder + '-logsig-3',
                 missing_rate=missing_rate,
                 discrepancy_fn='logsig-3',
                 old_shapelets=False)


def ablation_test():
    result_folder = 'ablation'
    for dataset_name in standard_dataset_names:
        print("Starting comparison: init-L2, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-init-L2',
             missing_rate=0.,
             discrepancy_fn='L2',
             ablation_init=False)
        print("Starting comparison: init-logsig-3, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-init-logsig-3',
             missing_rate=0.,
             discrepancy_fn='logsig-3',
             ablation_init=False)
        print("Starting comparison: log-L2, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-log-L2',
             missing_rate=0.,
             discrepancy_fn='L2',
             ablation_log=False)
        print("Starting comparison: log-logsig-3, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-log-logsig-3',
             missing_rate=0.,
             discrepancy_fn='logsig-3',
             ablation_log=False)
        print("Starting comparison: pseudometric-L2, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-pseudometric-L2',
             missing_rate=0.,
             discrepancy_fn='L2',
             ablation_pseudometric=False)
        print("Starting comparison: pseudometric-logsig-3, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-pseudometric-logsig-3',
             missing_rate=0.,
             discrepancy_fn='logsig-3',
             ablation_pseudometric=False)
        print("Starting comparison: learntlengths-L2, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-learntlengths-L2',
             missing_rate=0.,
             discrepancy_fn='L2',
             ablation_learntlengths=False)
        print("Starting comparison: learntlengths-logsig-3, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-learntlengths-logsig-3',
             missing_rate=0.,
             discrepancy_fn='logsig-3',
             ablation_learntlengths=False)


def noise_test():
    result_folder = 'noise'
    for dataset_name in standard_dataset_names:
        for noise_channels in (3, 9, 30):
            print("Starting comparison: L2, " + dataset_name + str(noise_channels))
            main(dataset_name,
                 noise_channels=noise_channels,
                 result_folder=result_folder,
                 result_subfolder=dataset_name + str(noise_channels) + '-L2',
                 missing_rate=0.,
                 discrepancy_fn='L2',
                 ablation_init=False)
            print("Starting comparison: logsig-3, " + dataset_name + str(noise_channels))
            main(dataset_name,
                 noise_channels=noise_channels,
                 result_folder=result_folder,
                 result_subfolder=dataset_name + str(noise_channels) + '-logsig-3',
                 missing_rate=0.,
                 discrepancy_fn='logsig-3',
                 ablation_init=False)
