import pathlib
import sklearn.model_selection
import torch

import common


here = pathlib.Path(__file__).resolve().parent


def _read_file_data(filename, type_):
    with open(filename) as acc_file:
        _acc_file_data = acc_file.read().strip().split('\n')
        acc_file_data = []
        for line in _acc_file_data:
            acc_file_data.append(tuple(type_(i) for i in line.strip().split(' ')))
    return acc_file_data


def get_data(threshold=74, normed_length=74):
    """Returns the Human Activity dataset, after you've downloaded it. (Run `python download.py` first).

    Note that the dataset needs shuffling before you do anything with it!

    With the default arguments:
    X will be of size (9703 74, 6)
    y will be of size (9703)
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
        # obvious that it's going to be that easy to learn from short series so here we split it up into pieces
        # 'normed_length' steps long, and apply fill-forward padding to the end if it's still at least of length
        # 'threshold'' and discard it if it's shorter. This doesn't affect much of our dataset.
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

    train_X, valtest_X, train_y, valtest_y = sklearn.model_selection.train_test_split(X, y,
                                                                                      train_size=0.8,
                                                                                      random_state=0,
                                                                                      shuffle=True,
                                                                                      stratify=y)

    val_X, test_X, val_y, test_y = sklearn.model_selection.train_test_split(valtest_X, valtest_y,
                                                                            train_size=0.5,
                                                                            random_state=1,
                                                                            shuffle=True,
                                                                            stratify=valtest_y)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

    train_dataloader = common.dataloader(train_dataset, batch_size=1024)
    val_dataloader = common.dataloader(val_dataset, batch_size=1024)
    test_dataloader = common.dataloader(test_dataset, batch_size=1024)

    times = torch.linspace(0, train_X.size(1) - 1, train_X.size(1), dtype=train_X.dtype, device=train_X.device)

    return times, train_dataloader, val_dataloader, test_dataloader


def main(result_folder=None,                  # saving parameters
         result_subfolder=None,               #
         epochs=1000,                         # training parameters
         num_shapelets_per_class=4,           # model parameters
         num_shapelet_samples=None,           #
         discrepancy_fn='L2',                 #
         max_shapelet_length_proportion=1.0,  #
         lengths_per_shapelet=1,              #
         num_continuous_samples=None,         #
         metric_type='general',               #
         ablation_pseudometric=True,          # For ablation studies
         ablation_learntlengths=True,         #
         ablation_similarreg=True,            #
         old_shapelets=False):                # Whether to toggle off all of our innovations and use old-style shapelets

    times, train_dataloader, val_dataloader, test_dataloader = get_data()

    input_channels = 6
    num_classes = 6

    return common.main(times,
                       train_dataloader,
                       val_dataloader,
                       test_dataloader,
                       num_classes,
                       input_channels,
                       result_folder,
                       result_subfolder,
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
