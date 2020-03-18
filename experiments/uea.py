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
                       'PEMS - SF',
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


def get_data(dataset_name, device):
    # We begin by loading both the train and test data and using our own train/val/test split.
    # The reason for this is that (a) by default there is no val split and (b) the sizes of the train/test splits are
    # really janky by default. (e.g. LSST has 2459 training samples and 2466 test samples.)

    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_filename = here / 'data' / 'UEA' / 'Multivariate_ts' / dataset_name / dataset_name
    train_X, train_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
    test_X, test_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    lengths = torch.tensor([len(Xi[0]) for Xi in X])
    maxlen = lengths.max()
    # X is now a numpy array of shape (batch, channel)
    # Each channel is a pandas.core.series.Series object of length corresponding to the length of the time series
    X = torch.stack([torch.stack([_pad(channel, maxlen) for channel in batch], dim=0) for batch in X], dim=0)
    # X is now a tensor of shape (batch, channel, length)
    X = X.transpose(-1, -2)
    # X is now a tensor of shape (batch, length, channel)

    # This isn't perfect because of the padding, but good enough!
    X = common.normalise_data(X)

    times = torch.linspace(0, X.size(1) - 1, X.size(1), dtype=X.dtype)

    # Now fix the labels to be integers from 0 upwards
    targets = co.OrderedDict()
    counter = 0
    for yi in y:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    y = torch.tensor([targets[yi] for yi in y])

    # 0.7/0.15/0.15 train/val/test split
    train_X, testval_X, train_y, testval_y = sklearn.model_selection.train_test_split(X, y,
                                                                                      train_size=0.7,
                                                                                      random_state=0,
                                                                                      shuffle=True,
                                                                                      stratify=y)
    test_X, val_X, test_y, val_y = sklearn.model_selection.train_test_split(testval_X, testval_y,
                                                                            train_size=0.5,
                                                                            random_state=1,
                                                                            shuffle=True,
                                                                            stratify=testval_y)

    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)
    test_X = test_X.to(device)
    test_y = test_y.to(device)
    times = times.to(device)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

    train_dataloader = common.dataloader(train_dataset, batch_size=2048)
    val_dataloader = common.dataloader(val_dataset, batch_size=2048)
    test_dataloader = common.dataloader(test_dataset, batch_size=2048)

    num_classes = counter
    input_channels = X.size(-1)

    assert num_classes >= 2, "Have only {} classes.".format(num_classes)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels


def main(dataset_name,                         # dataset parameters
         epochs=1000, device='cpu',            # training parameters
         num_shapelets_per_class=3,            # model parameters
         num_shapelet_samples=None,            #
         discrepancy_fn='L2',                  #
         max_shapelet_length_proportion=None,  #
         num_continuous_samples=None):         #

    (times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels) = get_data(dataset_name,
                                                                                                       device)

    assert times.is_floating_point(), "Whoops, times isn't floating point."

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
        discrepancy_fn = torchshapelets.L2Discrepancy(in_channels=input_channels, pseudometric=True)
    elif 'logsig' in discrepancy_fn:
        # expects e.g. 'logsig-4'
        depth = int(discrepancy_fn.split('-')[1])
        discrepancy_fn = torchshapelets.LogsignatureDiscrepancy(in_channels=input_channels, depth=depth)

    num_shapelets = num_shapelets_per_class * num_classes

    model = common.LinearShapeletTransform(in_channels=input_channels,
                                           out_channels=num_classes,
                                           num_shapelets=num_shapelets,
                                           num_shapelet_samples=num_shapelet_samples,
                                           discrepancy_fn=discrepancy_fn,
                                           max_shapelet_length=max_shapelet_length,
                                           num_continuous_samples=num_continuous_samples).to(device)

    sample_batch = common.get_sample_batch(train_dataloader, num_shapelets_per_class, num_shapelets)
    model.set_shapelets(times, sample_batch)  # smart initialisation of shapelets

    if num_classes == 2:
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    else:
        loss_fn = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    history = common.train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, epochs, num_classes)
    results = common.evaluate_model(train_dataloader, val_dataloader, test_dataloader, model, times, loss_fn, history,
                                    num_classes)
    return results
