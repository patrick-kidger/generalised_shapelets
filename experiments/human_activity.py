import pathlib
import sklearn.model_selection
import torch
import torchshapelets

import common


here = pathlib.Path(__file__).resolve().parent


def _read_file_data(filename, type_):
    with open(filename) as acc_file:
        _acc_file_data = acc_file.read().strip().split('\n')
        acc_file_data = []
        for line in _acc_file_data:
            acc_file_data.append(tuple(type_(i) for i in line.strip().split(' ')))
    return acc_file_data


def get_data(device):
    """There are 3795 samples, each of length 200, with 7 channels. (6 from the data + 1 from time).
    Each label is an integer 0, 1, 2, 3, 4, 5.
    """
    base_loc = str(here / 'data/HumanActivity/RawData')
    labels_file_data = _read_file_data(base_loc + '/labels.txt', int)

    X = []
    y = []

    last_experiment_number = None
    last_user_number = None
    for experiment_number, user_number, activity_number, start, end in labels_file_data:
        # There are 12 classes:
        # 1 Walking
        # 2 Walking upstairs
        # 3 Walking downstairs
        # 4 Sitting
        # 5 Standing
        # 6 Lieing down
        # 7 Standing to siting
        # 8 Sitting to standing
        # 9 Siting to lieing down
        # 10 Lieing down to sitting
        # 11 Standing to lieing down
        # 12 Lieing down to standing
        # But some have very few samples, and without them it's basically a balanced classification problem.
        if activity_number > 6:
            continue

        end += 1
        if experiment_number != last_experiment_number or user_number != last_user_number:
            acc_filename = 'acc_exp{:02}_user{:02}.txt'.format(experiment_number, user_number)
            gyro_filename = 'gyro_exp{:02}_user{:02}.txt'.format(experiment_number, user_number)
            acc_file_data = torch.tensor(_read_file_data(base_loc + '/' + acc_filename, float))
            gyro_file_data = torch.tensor(_read_file_data(base_loc + '/' + gyro_filename, float))
            # Is a tensor of shape (length, channels=6)
            both_data = torch.cat([acc_file_data, gyro_file_data], dim=1)
        last_experiment_number = experiment_number
        last_user_number = user_number

        # minimum length is 74
        # maximum length is 2032
        # I think what they did in the original dataset was split it up into pieces roughly 74 steps long. It's not
        # obvious that it's going to be that easy to learn from short series so here we split it up into pieces 200
        # steps long, and apply fill-forward padding to the end if it's still at least of length 100 and discard it if
        # it's shorter. This doesn't affect much of our dataset.
        threshold = 100
        normed_length = 200
        for start_ in range(start, end, normed_length):
            start_plus = start_ + normed_length
            if start_plus > end:
                too_short = True
                if start_plus - end < threshold:
                    continue  # skip data
                end_ = min(start_plus, end)
            else:
                too_short = False
                end_ = start_plus
            Xi = both_data[start_:end_]
            if too_short:
                Xi = torch.cat([Xi, Xi[-1].repeat(start_plus - end, 1)], dim=0)
            X.append(Xi)
            y.append(activity_number - 1)
    X = torch.stack(X, dim=0)
    y = torch.tensor(y)

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

    train_X = common.normalise_data(train_X, train_X)
    val_X = common.normalise_data(val_X, train_X)
    test_X = common.normalise_data(test_X, train_X)

    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)
    test_X = test_X.to(device)
    test_y = test_y.to(device)
    times = torch.linspace(0, X.size(1) - 1, X.size(1), dtype=X.dtype, device=device)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

    train_dataloader = common.dataloader(train_dataset, batch_size=2048)
    val_dataloader = common.dataloader(val_dataset, batch_size=2048)
    test_dataloader = common.dataloader(test_dataset, batch_size=2048)

    return times, train_dataloader, val_dataloader, test_dataloader


def main(epochs=1000, device='cpu',            # training parameters
         num_shapelets_per_class=2,            # model parameters
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

    times, train_dataloader, val_dataloader, test_dataloader = get_data(device)

    assert times.is_floating_point(), "Whoops, times isn't floating point."

    if old_shapelets:
        num_continuous_samples = times.size(0)

        def discrepancy_fn(times, path, shapelet):
            return ((path - shapelet) ** 2).sum(dim=(-1, -2))

    # Select some sensible options based on the length of the dataset
    timespan = times[-1] - times[0]
    if max_shapelet_length_proportion is None:
        max_shapelet_length_proportion = min((4 / timespan).sqrt(), 1)
    max_shapelet_length = timespan * max_shapelet_length_proportion
    if num_shapelet_samples is None:
        num_shapelet_samples = int(max_shapelet_length_proportion * times.size(0))
    if num_continuous_samples is None:
        num_continuous_samples = times.size(0)

    if discrepancy_fn == 'L2':
        discrepancy_fn = torchshapelets.L2Discrepancy(in_channels=6, pseudometric=ablation_pseudometric)
    elif 'logsig' in discrepancy_fn:
        # expects e.g. 'logsig-4'
        depth = int(discrepancy_fn.split('-')[1])
        discrepancy_fn = torchshapelets.LogsignatureDiscrepancy(in_channels=6, depth=depth,
                                                                pseudometric=ablation_pseudometric)

    num_shapelets = num_shapelets_per_class * 6

    model = common.LinearShapeletTransform(in_channels=6,
                                           out_channels=6,
                                           num_shapelets=num_shapelets,
                                           num_shapelet_samples=num_shapelet_samples,
                                           discrepancy_fn=discrepancy_fn,
                                           max_shapelet_length=max_shapelet_length,
                                           num_continuous_samples=num_continuous_samples,
                                           ablation_log=ablation_log).to(device)

    if old_shapelets:
        del model.shapelet_transform.lengths
        model.shapelet_transform.register_buffer('lengths', torch.full_like(model.shapelet_transform.lengths,
                                                                            max_shapelet_length))
    else:
        if ablation_init:
            sample_batch = common.get_sample_batch(train_dataloader, num_shapelets_per_class, num_shapelets)
            model.set_shapelets(times, sample_batch)  # smart initialisation of shapelets
        if ablation_learntlengths:
            model.shapelet_transform.lengths.requires_grad_(False)

    loss_fn = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    history = common.train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, epochs, 6,
                                ablation_similarreg, ablation_lengthreg, ablation_pseudoreg)
    results = common.evaluate_model(train_dataloader, val_dataloader, test_dataloader, model, times, loss_fn, history,
                                    6)
    return results
