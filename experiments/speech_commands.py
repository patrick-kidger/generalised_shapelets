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
    tensors = _load_data(here / 'data/speech_commands_data')
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
         num_continuous_samples=None,         #
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
                       num_continuous_samples,
                       ablation_pseudometric,
                       ablation_learntlengths,
                       ablation_similarreg,
                       old_shapelets)


def hyperparameter_search():
    result_folder = 'speech_commands_hyperparameter_search'
    for num_shapelets_per_class in (2, 3, 5):
        for max_shapelet_length_proportion in (0.3, 0.5, 1.0):
            result_subfolder = 'old-' + str(num_shapelets_per_class) + '-' + str(max_shapelet_length_proportion)
            print("Starting comparison: " + result_subfolder)
            main(result_folder=result_folder,
                 result_subfolder=result_subfolder,
                 num_shapelets_per_class=num_shapelets_per_class,
                 max_shapelet_length_proportion=max_shapelet_length_proportion,
                 old_shapelets=True)


def comparison_test(seed, old):
    common.handle_seed(seed)
    main(result_folder='speech_commands',
         result_subfolder='old' if old else 'L2',
         old_shapelets=old)
