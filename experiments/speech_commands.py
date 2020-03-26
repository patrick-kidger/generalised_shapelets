import os
import pathlib
import torch
import torchaudio
import torchshapelets

import common


here = pathlib.Path(__file__).resolve().parent


def get_data(device):
    base_loc = here / 'data' / 'SpeechCommands'
    train_X = torch.empty(16623, 16000, 1)  # 18537
    val_X = torch.empty(2324, 16000, 1)  # 2576
    test_X = torch.empty(2362, 16000, 1)  # 2566
    train_y = torch.empty(16623, dtype=torch.long)
    val_y = torch.empty(2324, dtype=torch.long)
    test_y = torch.empty(2362, dtype=torch.long)

    with open(base_loc / 'validation_list.txt') as f:
        val_list = set(f.read().strip().split('\n'))
    with open(base_loc / 'testing_list.txt') as f:
        test_list = set(f.read().strip().split('\n'))

    train_batch_index = -1
    val_batch_index = -1
    test_batch_index = -1
    y_index = 0
    for foldername in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
        loc = base_loc / foldername
        for filename in os.listdir(loc):
            audio, _ = torchaudio.load_wav(loc / filename, channels_first=False,
                                           normalization=False)  # for forward compatbility if they fix it
            audio = audio / 2 ** 15  # Normalization argument doesn't seem to work so we do it manually.

            # A few samples are shorter than the full length; for simplicity we discared them.
            # (Specifically, 2370 samples out of 23682 are shorter than the full length.)
            if len(audio) != 16000:
                continue

            fileloc = str(foldername + '/' + filename)
            if fileloc in val_list:
                X = val_X
                y = val_y
                batch_index = val_batch_index
                val_batch_index += 1
            elif fileloc in test_list:
                X = test_X
                y = test_y
                batch_index = test_batch_index
                test_batch_index += 1
            else:
                X = train_X
                y = train_y
                batch_index = train_batch_index
                train_batch_index += 1

            X[batch_index] = audio
            y[batch_index] = y_index
        y_index += 1

    def stft(X):  # takes an X of shape (batch, length=16000, channels=1)
        # shape (batch, frequencies=81, length=100, real,imag=2)
        X = torch.stft(X.squeeze(-1), 160, hop_length=160, center=False, normalized=True)
        X = X[:, :50]  # take the first 50 frequencies
        X = X.transpose(1, 2)  # now of shape (batch, length=100, frequencies=50, real,imag=2)
        X = X.reshape(X.size(0), 100, 100)  # shape (batch, length=100, channels=100)
        return X

    train_X = stft(train_X)
    val_X = stft(val_X)
    test_X = stft(test_X)

    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)
    test_X = test_X.to(device)
    test_y = test_y.to(device)
    times = torch.linspace(0, train_X.size(1) - 1, train_X.size(1), device=device)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

    train_dataloader = common.dataloader(train_dataset, batch_size=128)
    val_dataloader = common.dataloader(val_dataset, batch_size=128)
    test_dataloader = common.dataloader(test_dataset, batch_size=128)

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

    in_channels = 100
    num_classes = 10

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
        discrepancy_fn = torchshapelets.L2Discrepancy(in_channels=in_channels, pseudometric=ablation_pseudometric)
    elif 'logsig' in discrepancy_fn:
        # expects e.g. 'logsig-4'
        depth = int(discrepancy_fn.split('-')[1])
        discrepancy_fn = torchshapelets.LogsignatureDiscrepancy(in_channels=in_channels, depth=depth,
                                                                pseudometric=ablation_pseudometric)

    num_shapelets = num_shapelets_per_class * num_classes

    model = common.LinearShapeletTransform(in_channels=in_channels,
                                           out_channels=num_classes,
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
            assert not torch.isinf(model.shapelet_transform.shapelets).any()
        if ablation_learntlengths:
            model.shapelet_transform.lengths.requires_grad_(False)


    # class MyGRU(torch.nn.Module):
    #     def __init__(self):
    #         super(MyGRU, self).__init__()
    #         self.gru = torch.nn.GRU(input_size=in_channels, hidden_size=32, batch_first=True)
    #         self.linear = torch.nn.Linear(32, num_classes)
    #     def forward(self, times, X):
    #         _, x = self.gru(X)
    #         x = x.squeeze(0)
    #         return self.linear(x), None, None, None
    #     def clip_length(self):
    #         pass
    # model = MyGRU().to(device)
    # ablation_similarreg, ablation_lengthreg, ablation_pseudoreg = False, False, False

    loss_fn = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    history = common.train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, epochs, num_classes,
                                ablation_similarreg, ablation_lengthreg, ablation_pseudoreg)
    results = common.evaluate_model(train_dataloader, val_dataloader, test_dataloader, model, times, loss_fn, history,
                                    num_classes)
    return results
