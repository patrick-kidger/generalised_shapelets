import os
import pathlib
import torch

import common


here = pathlib.Path(__file__).resolve().parent


def _load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def get_data():
    tensors = _load_data(here / 'speech_commands_data')
    train_dataset = torch.utils.data.TensorDataset(tensors['train_X'], tensors['train_y'])
    val_dataset = torch.utils.data.TensorDataset(tensors['val_X'], tensors['val_y'])
    test_dataset = torch.utils.data.TensorDataset(tensors['test_X'], tensors['test_y'])

    train_dataloader = common.dataloader(train_dataset, batch_size=1024)
    val_dataloader = common.dataloader(val_dataset, batch_size=1024)
    test_dataloader = common.dataloader(test_dataset, batch_size=1024)

    train_X = tensors['train_X']
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

    input_channels = 40
    num_classes = 10

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


def james1():
    main(result_folder='speech_commands', result_subfolder='L2')


def james2():
    main(result_folder='speech_commands', result_subfolder='L2-diagonal', metric_type='diagonal')


james4 = james3 = james2

james6 = james5 = james1
