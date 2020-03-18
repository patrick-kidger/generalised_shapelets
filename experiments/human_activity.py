import pathlib
import torch

import common
import models


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
    final_index = torch.tensor(199).repeat(y.size(0))

    return common.split_data(X, y, final_index, device=device)


def main(epochs=1000, device='cuda',  # training parameters
         hidden_channels=32,          # model parameters
         **kwargs):                   # kwargs passed on to cdeint

    times, train_dataloader, val_dataloader, test_dataloader = get_data(device)



    loss_fn = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = common.train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, epochs,
                                num_classes=6, kwargs=kwargs)
    results = common.evaluate_model(train_dataloader, val_dataloader, test_dataloader, model, times, loss_fn,
                                    history, num_classes=6, kwargs=kwargs)
    return results
