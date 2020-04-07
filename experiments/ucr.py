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


valid_dataset_names = ('ACSF1',
                       'Adiac',
                       'AllGestureWiimoteX',
                       'AllGestureWiimoteY',
                       'AllGestureWiimoteZ',
                       'ArrowHead',
                       'Beef',
                       'BeetleFly',
                       'BirdChicken',
                       'BME',
                       'Car',
                       'CBF',
                       'Chinatown',
                       'ChlorineConcentration',
                       'CinCECGTorso',
                       'Coffee',
                       'Computers',
                       'CricketX',
                       'CricketY',
                       'CricketZ',
                       'Crop',
                       'DiatomSizeReduction',
                       'DistalPhalanxOutlineAgeGroup',
                       'DistalPhalanxOutlineCorrect',
                       'DistalPhalanxTW',
                       'DodgerLoopDay',
                       'DodgerLoopGame',
                       'DodgerLoopWeekend',
                       'Earthquakes',
                       'ECG200',
                       'ECG5000',
                       'ECGFiveDays',
                       'ElectricDevices',
                       'EOGHorizontalSignal',
                       'EOGVerticalSignal',
                       'EthanolLevel',
                       'FaceAll',
                       'FaceFour',
                       'FacesUCR',
                       'FiftyWords',
                       'Fish',
                       'FordA',
                       'FordB',
                       'FreezerRegularTrain',
                       'FreezerSmallTrain',
                       'Fungi',
                       'GestureMidAirD1',
                       'GestureMidAirD2',
                       'GestureMidAirD3',
                       'GesturePebbleZ1',
                       'GesturePebbleZ2',
                       'GunPoint',
                       'GunPointAgeSpan',
                       'GunPointMaleVersusFemale',
                       'GunPointOldVersusYoung',
                       'Ham',
                       'HandOutlines',
                       'Haptics',
                       'Herring',
                       'HouseTwenty',
                       'InlineSkate',
                       'InsectEPGRegularTrain',
                       'InsectEPGSmallTrain',
                       'InsectWingbeatSound',
                       'ItalyPowerDemand',
                       'LargeKitchenAppliances',
                       'Lightning2',
                       'Lightning7',
                       'Mallat',
                       'Meat',
                       'MedicalImages',
                       'MelbournePedestrian',
                       'MiddlePhalanxOutlineAgeGroup',
                       'MiddlePhalanxOutlineCorrect',
                       'MiddlePhalanxTW',
                       'MixedShapesRegularTrain',
                       'MixedShapesSmallTrain',
                       'MoteStrain',
                       'NonInvasiveFetalECGThorax1',
                       'NonInvasiveFetalECGThorax2',
                       'OliveOil',
                       'OSULeaf',
                       'PhalangesOutlinesCorrect',
                       'Phoneme',
                       'PickupGestureWiimoteZ',
                       'PigAirwayPressure',
                       'PigArtPressure',
                       'PigCVP',
                       'PLAID',
                       'Plane',
                       'PowerCons',
                       'ProximalPhalanxOutlineAgeGroup',
                       'ProximalPhalanxOutlineCorrect',
                       'ProximalPhalanxTW',
                       'RefrigerationDevices',
                       'Rock',
                       'ScreenType',
                       'SemgHandGenderCh2',
                       'SemgHandMovementCh2',
                       'SemgHandSubjectCh2',
                       'ShakeGestureWiimoteZ',
                       'ShapeletSim',
                       'ShapesAll',
                       'SmallKitchenAppliances',
                       'SmoothSubspace',
                       'SonyAIBORobotSurface1',
                       'SonyAIBORobotSurface2',
                       'StarLightCurves',
                       'Strawberry',
                       'SwedishLeaf',
                       'Symbols',
                       'SyntheticControl',
                       'ToeSegmentation1',
                       'ToeSegmentation2',
                       'Trace',
                       'TwoLeadECG',
                       'TwoPatterns',
                       'UMD',
                       'UWaveGestureLibraryAll',
                       'UWaveGestureLibraryX',
                       'UWaveGestureLibraryY',
                       'UWaveGestureLibraryZ',
                       'Wafer',
                       'Wine',
                       'WordSynonyms',
                       'Worms',
                       'WormsTwoClass',
                       'Yoga')

long_datasets = {'HouseTwenty', 'Rock', 'HandOutlines', 'PigCVP', 'PigArtPressure', 'PigAirwayPressure', 'InlineSkate',
                 'EthanolLevel', 'CinCECGTorso', 'SemgHandSubjectCh2', 'SemgHandMovementCh2', 'SemgHandGenderCh2',
                 'ACSF1',  'EOGVerticalSignal', 'EOGHorizontalSignal', 'Haptics', 'StarLightCurves', 'Phoneme',
                 'MixedShapesSmallTrain', 'MixedShapesRegularTrain', 'Mallat', 'UWaveGestureLibraryAll',
                 'WormsTwoClass', 'Worms', 'NonInvasiveFetalECGThorax2', 'NonInvasiveFetalECGThorax1'}

large_datasets = {'ChlorineConcentration', 'Crop',
                  'ECG5000', 'ElectricDevices', 'FaceAll', 'FacesUCR', 'FordA', 'FordB', 'FreezerRegularTrain',
                  'FreezerSmallTrain', 'InsectWingbeatSound', 'ItalyPowerDemand', 'MelbournePedestrian',
                  'MoteStrain', 'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
                  'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Yoga', 'HandOutlines',
                  'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'StarLightCurves',
                  'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2'}

electric_device_datasets = ('CinCECGtorso', 'ECG200', 'ECG5000', 'ECGFiveDays', 'NonInvasiveFetalECGThorax1',
                            'NonInvasiveFetalECGThorax2', 'TwoLeadECG', 'ACSF1', 'Computers', 'ElectricDevices',
                            'HouseTwenty', 'LargeKitchenAppliances', 'PowerCons', 'RefrigerationDevices', 'ScreenType',
                            'SmallKitchenAppliances')

irregular_length_datasets = {'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ', 'GesturePebbleZ1', 'GesturePebbleZ2',
                             'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'PLAID'}

missing_data_datasets = {'DodgerLoopGame', 'DodgerLoopWeekend'}

# Ordered by dataset size * num_classes * length ** 2, i.e. the cost of evaluating shaplets on them.
datasets_by_cost = ('Chinatown',
                    'ItalyPowerDemand',
                    'SmoothSubspace',
                    'SonyAIBORobotSurface1',
                    'SonyAIBORobotSurface2',
                    'MoteStrain',
                    'TwoLeadECG',
                    'ECGFiveDays',
                    'BME',
                    'CBF',
                    'ECG200',
                    'GunPoint',
                    'UMD',
                    'DodgerLoopGame',
                    'DodgerLoopWeekend',
                    'Coffee',
                    'FreezerSmallTrain',
                    'GunPointAgeSpan',
                    'GunPointMaleVersusFemale',
                    'GunPointOldVersusYoung',
                    'ToeSegmentation1',
                    'Wine',
                    'SyntheticControl',
                    'ArrowHead',
                    'MelbournePedestrian',
                    'PowerCons',
                    'DiatomSizeReduction',
                    'DistalPhalanxOutlineAgeGroup',
                    'DistalPhalanxOutlineCorrect',
                    'MiddlePhalanxOutlineAgeGroup',
                    'MiddlePhalanxOutlineCorrect',
                    'ProximalPhalanxOutlineAgeGroup',
                    'ProximalPhalanxOutlineCorrect',
                    'ToeSegmentation2',
                    'ShapeletSim',
                    'BeetleFly',
                    'BirdChicken',
                    'FaceFour',
                    'Fungi',
                    'Plane',
                    'MiddlePhalanxTW',
                    'DistalPhalanxTW',
                    'ProximalPhalanxTW',
                    'InsectEPGSmallTrain',
                    'PhalangesOutlinesCorrect',
                    'Symbols',
                    'FreezerRegularTrain',
                    'Trace',
                    'Beef',
                    'Herring',
                    'Meat',
                    'MedicalImages',
                    'ChlorineConcentration',
                    'OliveOil',
                    'Ham',
                    'DodgerLoopDay',
                    'Wafer',
                    'FacesUCR',
                    'Lightning2',
                    'ECG5000',
                    'Lightning7',
                    'PickupGestureWiimoteZ',
                    'TwoPatterns',
                    'InsectEPGRegularTrain',
                    'Strawberry',
                    'ShakeGestureWiimoteZ',
                    'Car',
                    'Yoga',
                    'SwedishLeaf',
                    'FaceAll',
                    'InsectWingbeatSound',
                    'GesturePebbleZ1',
                    'Earthquakes',
                    'GesturePebbleZ2',
                    'OSULeaf',
                    'Computers',
                    'Fish',
                    'WormsTwoClass',
                    'Crop',
                    'CricketX',
                    'CricketY',
                    'CricketZ',
                    'CinCECGTorso',
                    'Adiac',
                    'Mallat',
                    'WordSynonyms',
                    'MixedShapesSmallTrain',
                    'ElectricDevices',
                    'LargeKitchenAppliances',
                    'RefrigerationDevices',
                    'ScreenType',
                    'SmallKitchenAppliances',
                    'HouseTwenty',
                    'Rock',
                    'GestureMidAirD1',
                    'GestureMidAirD2',
                    'GestureMidAirD3',
                    'UWaveGestureLibraryX',
                    'UWaveGestureLibraryY',
                    'UWaveGestureLibraryZ',
                    'Worms',
                    'AllGestureWiimoteX',
                    'AllGestureWiimoteY',
                    'AllGestureWiimoteZ',
                    'Haptics',
                    'SemgHandGenderCh2',
                    'FiftyWords',
                    'FordA',
                    'FordB',
                    'ACSF1',
                    'InlineSkate',
                    'MixedShapesRegularTrain',
                    'StarLightCurves',
                    'SemgHandSubjectCh2',
                    'SemgHandMovementCh2',
                    'EthanolLevel',
                    'UWaveGestureLibraryAll',
                    'EOGHorizontalSignal',
                    'EOGVerticalSignal',
                    'Phoneme',
                    'ShapesAll',
                    'PLAID',
                    'HandOutlines',
                    'PigAirwayPressure',
                    'PigArtPressure',
                    'PigCVP',
                    'NonInvasiveFetalECGThorax1',
                    'NonInvasiveFetalECGThorax2')


def get_data(dataset_name, device):
    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_filename = here / 'data' / 'UCR' / 'Univariate_ts' / dataset_name / dataset_name
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

    times = torch.linspace(0, all_X.size(1) - 1, all_X.size(1), dtype=all_X.dtype, device=device)

    # Now fix the labels to be integers from 0 upwards
    targets = co.OrderedDict()
    counter = 0
    for yi in all_y:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    all_y = torch.tensor([targets[yi] for yi in all_y])

    missing_data = torch.isnan(all_X).any()

    # use original train/test splits
    trainval_X, test_X = all_X[:amount_train], all_X[amount_train:]
    trainval_y, test_y = all_y[:amount_train], all_y[amount_train:]

    # If any classes have only a single training example then add them to the training set, not the validation set.
    # (If we don't handle this explicitly then train_test_split throws an error.)
    _, inverse_indices, counts = trainval_y.unique(return_inverse=True, return_counts=True)
    if (counts == 1).any():
        append_extras = True
        mask_single = counts[inverse_indices] == 1
        index_tensor = torch.arange(trainval_X.size(0))
        indices_single = index_tensor.masked_select(mask_single)
        indices_multiple = index_tensor.masked_select(~mask_single)
        extra_X = trainval_X[indices_single]
        extra_y = trainval_y[indices_single]
        trainval_X = trainval_X[indices_multiple]
        trainval_y = trainval_y[indices_multiple]
    else:
        append_extras = False

    train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(trainval_X, trainval_y,
                                                                              train_size=0.8,
                                                                              random_state=0,
                                                                              shuffle=True,
                                                                              stratify=trainval_y)
    if append_extras:
        train_X = torch.cat([train_X, extra_X], dim=0)
        train_y = torch.cat([train_y, extra_y], dim=0)

    val_X = common.normalise_data(val_X, train_X)
    test_X = common.normalise_data(test_X, train_X)
    train_X = common.normalise_data(train_X, train_X)

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

    assert num_classes >= 2, "Have only {} classes.".format(num_classes)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, missing_data


def main(dataset_name,                         # dataset parameters
         result_folder=None, result_subfolder=None,      # saving parameters
         epochs=1000, device='cpu',            # training parameters
         num_shapelets_per_class=4,            # model parameters
         num_shapelet_samples=None,            #
         discrepancy_fn='L2',                  #
         max_shapelet_length_proportion=1.0,   #
         num_continuous_samples=None,          #
         ablation_init=True,                   # For ablation studies
         ablation_log=True,                    #
         ablation_pseudometric=True,           #
         ablation_learntlengths=True,          #
         ablation_similarreg=True,             #
         ablation_lengthreg=False,             #
         ablation_pseudoreg=False,             #
         old_shapelets=False):                 # Whether to toggle off all of our innovations and use old-style
                                               # shapelets.

    (times, train_dataloader, val_dataloader, test_dataloader, num_classes, missing_data) = get_data(dataset_name,
                                                                                                     device)

    input_channels = 1

    assert times.is_floating_point(), "Whoops, times isn't floating point."

    if old_shapelets:
        discrepancy_fn = 'L2_squared'
        max_shapelet_length_proportion = 0.3
        ablation_log = False
        ablation_pseudometric = False
        ablation_learntlengths = False
        ablation_similarreg = False
        ablation_lengthreg = False
        ablation_pseudoreg = False
        num_continuous_samples = None

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
            # Note that this treats the inputs as piecewise constant, not piecewise linear.
            return ((path - shapelet) ** 2).sum(dim=(-1, -2))
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

    if num_classes == 2:
        out_channels = 1
    else:
        out_channels = num_classes

    model = common.LinearShapeletTransform(in_channels=input_channels,
                                           out_channels=out_channels,
                                           num_shapelets=num_shapelets,
                                           num_shapelet_samples=num_shapelet_samples,
                                           discrepancy_fn=discrepancy_fn,
                                           max_shapelet_length=max_shapelet_length,
                                           num_continuous_samples=num_continuous_samples,
                                           ablation_log=ablation_log)

    if old_shapelets:
        new_lengths = torch.full_like(model.shapelet_transform.lengths, max_shapelet_length)
        del model.shapelet_transform.lengths
        model.shapelet_transform.register_buffer('lengths', new_lengths)
    if ablation_init:
        sample_batch = common.get_sample_batch(train_dataloader, num_shapelets_per_class, num_shapelets)
        model.set_shapelets(times.to('cpu'), sample_batch.to('cpu'))  # smart initialisation of shapelets
    if not ablation_learntlengths:
        model.shapelet_transform.lengths.requires_grad_(False)

    if num_classes == 2:
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    else:
        loss_fn = torch.nn.functional.cross_entropy

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    history = common.train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, epochs, num_classes,
                                ablation_similarreg, ablation_lengthreg, ablation_pseudoreg)
    results = common.evaluate_model(train_dataloader, val_dataloader, test_dataloader, model, times, loss_fn, history,
                                    num_classes)
    results.num_shapelets_per_class = num_shapelets_per_class
    results.num_shapelet_samples = num_shapelet_samples
    results.max_shapelet_length_proportion = max_shapelet_length_proportion
    results.ablation_init = ablation_init
    results.ablation_log = ablation_log
    results.ablation_pseudometric = ablation_pseudometric
    results.ablation_learntlengths = ablation_learntlengths
    results.ablation_similarreg = ablation_similarreg
    results.ablation_lengthreg = ablation_lengthreg
    results.ablation_pseudoreg = ablation_pseudoreg
    results.old_shapelets = old_shapelets
    if result_folder is not None:
        common.save_results(result_folder, result_subfolder, results)
    return results


def comparison_test():
    pseudometric = False  # meaningless in one dimension
    result_folder = 'ucr_comparison'
    for dataset_name in datasets_by_cost:
        # We're actually perfectly capable of handling irregular lengths, but they're a pain to batch over.
        # Similarly we can handle missing data (see the UEA experiment), but for simplicity we leave them out of our
        # large-scale study.
        if dataset_name in irregular_length_datasets | missing_data_datasets:
            continue
        print("Starting comparison: L2, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name+'-L2',
             discrepancy_fn='L2',
             old_shapelets=False,
             ablation_pseudometric=pseudometric)
        print("Starting comparison: logsig-3, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-logsig-3',
             discrepancy_fn='logsig-3',
             old_shapelets=False,
             ablation_pseudometric=pseudometric)
        print("Starting comparison: old-L2_squared, " + dataset_name)
        main(dataset_name,
             result_folder=result_folder,
             result_subfolder=dataset_name + '-old-L2_squared',
             num_shapelets_per_class=3,
             discrepancy_fn='L2_squared',
             old_shapelets=True,
             ablation_pseudometric=pseudometric)
