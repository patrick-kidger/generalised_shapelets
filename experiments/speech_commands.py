import os
import pathlib
import torch
import torchshapelets

import common


here = pathlib.Path(__file__).resolve().parent


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def get_data():
    tensors = load_data(here / 'speech_commands_data')
    train_dataset = torch.utils.data.TensorDataset(tensors['train_X'], tensors['train_y'])
    val_dataset = torch.utils.data.TensorDataset(tensors['val_X'], tensors['val_y'])
    test_dataset = torch.utils.data.TensorDataset(tensors['test_X'], tensors['test_y'])

    train_dataloader = common.dataloader(train_dataset, batch_size=1024)
    val_dataloader = common.dataloader(val_dataset, batch_size=1024)
    test_dataloader = common.dataloader(test_dataset, batch_size=1024)

    train_X = tensors['train_X']
    times = torch.linspace(0, train_X.size(1) - 1, train_X.size(1), dtype=train_X.dtype, device=train_X.device)

    return times, train_dataloader, val_dataloader, test_dataloader


def main(result_folder=None, result_subfolder=None,      # saving parameters
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

    times, train_dataloader, val_dataloader, test_dataloader = get_data()

    input_channels = 40
    num_classes = 10

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

    model = common.LinearShapeletTransform(in_channels=input_channels,
                                           out_channels=num_classes,
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
    pseudometric = True
    result_folder = 'speech_commands'
    print("Starting comparison: L2")
    main(result_folder=result_folder,
         result_subfolder='L2',
         discrepancy_fn='L2',
         old_shapelets=False,
         ablation_pseudometric=pseudometric)
    print("Starting comparison: logsig-3")
    main(result_folder=result_folder,
         result_subfolder='logsig-3',
         discrepancy_fn='logsig-3',
         old_shapelets=False,
         ablation_pseudometric=pseudometric)
    print("Starting comparison: old-L2_squared")
    main(result_folder=result_folder,
         result_subfolder='old-L2_squared',
         num_shapelets_per_class=3,
         discrepancy_fn='L2_squared',
         old_shapelets=True,
         ablation_pseudometric=pseudometric)
