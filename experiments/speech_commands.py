import json
import os
import pathlib
import random
import re
import scipy.io.wavfile
import torch
import torchaudio
import torchshapelets
import tqdm

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


def _get_sample(foldername):
    loc = here / 'data' / 'SpeechCommands' / foldername
    filenames = list(os.listdir(loc))
    while True:
        filename = random.choice(filenames)
        audio, _ = torchaudio.load_wav(loc / filename, channels_first=False,
                                       normalization=False)  # for forward compatbility if they fix it
        audio = audio / 2 ** 15  # Normalization argument doesn't seem to work so we do it manually.

        # A few samples are shorter than the full length; for simplicity we discard them.
        if len(audio) != 16000:
            continue
        return audio.squeeze()  # shape 16000


def invert(model_filename, find_closest=True):
    """Inverts the MFCC shapelet to find the corresponding audio shapelet."""

    # Get the shapelets we're going to invert
    state_dict = torch.load(here / 'results/speech_commands' / (model_filename + '_model'))
    weight = state_dict['linear.weight']
    most_informative = weight.argmin(dim=1)
    shapelets = state_dict['shapelet_transform.shapelets']
    shapelet_mfcc = shapelets[most_informative].to('cpu')
    lengths = state_dict['shapelet_transform.lengths']
    length = lengths[most_informative]

    # Get the data we trained on
    tensors = _load_data(here / 'data/speech_commands_data')
    train_audio_X = tensors['train_audio_X']
    train_X = tensors['train_X']
    means = tensors['means']
    stds = tensors['stds']

    if find_closest:
        # Run all of our training samples through the model and pick the ones that have the closest MFCC.

        # Get the details of the model we trained
        with open(here / 'results/speech_commands' / model_filename, 'rb') as f:
            results = json.load(f)
        model_string = results['model']

        def find(value):
            return re.search(value + '=([\.\w]+)', model_string).group(1)

        out_channels = int(find('out_features'))
        num_shapelets, num_shapelet_samples, in_channels = shapelets.shape
        ablation_pseudometric = bool(find('pseudometric'))
        # Assume L2 discrepancy
        discrepancy_fn = common.get_discrepancy_fn('L2', in_channels, ablation_pseudometric)
        max_shapelet_length = float(find('max_shapelet_length'))
        num_continuous_samples = int(find('num_continuous_samples'))
        # Doesn't actually affect anything here
        log = True

        # Recreate the model
        model = common.LinearShapeletTransform(in_channels, out_channels, num_shapelets, num_shapelet_samples,
                                               discrepancy_fn, max_shapelet_length, num_continuous_samples, log)
        model.load_state_dict(state_dict)

        shapelet_similarities = []
        closest_indices = []
        print('Finding init points')
        times = torch.linspace(0, train_X.size(1) - 1, train_X.size(1), dtype=train_X.dtype, device=train_X.device)
        with torch.no_grad():
            for train_Xi in tqdm.tqdm(train_X.split(200)):
                _, shapelet_similarity, closest_index = model(times, train_Xi)
                shapelet_similarities.append(shapelet_similarity)
                closest_indices.append(closest_index)
        shapelet_similarities = torch.cat(shapelet_similarities)
        closest_indices = torch.cat(closest_indices)
        closeset_per_shapelet = shapelet_similarities.argmin(dim=0)
        closest_indices = closest_indices[closeset_per_shapelet, range(len(closeset_per_shapelet))]
        closeset_per_shapelet = closeset_per_shapelet[most_informative]  # just keep the ones for the shapelets we care about
        closest_indices = closest_indices[most_informative]
        print(closeset_per_shapelet)
        print(closest_indices)
    else:
        # These were the ones we found were closest for one of our runs. If you don't want to do a search then you can
        # try using these instead.
        closeset_per_shapelet = torch.tensor([14429, 16271, 22411, 16943, 22223, 18688, 661, 17331, 2731, 6936])
        closest_indices = torch.tensor([36, 43, 25, 67, 40, 54, 50, 65, 11, 50])

    # Assumes that each shapelet as sampled at as many points as the series is long, i.e. that we used
    # num_shapelet_samples=None
    init_audio = train_audio_X[closeset_per_shapelet]
    initial_time = closest_indices * (train_X.size(1) - 1 - length) / train_X.size(1)
    ratio = train_audio_X.size(1) / train_X.size(1)
    initial_time = initial_time * ratio
    audio_length = length * ratio
    audio_times = torch.linspace(0, train_audio_X.size(1) - 1, train_audio_X.size(1), dtype=train_audio_X.dtype,
                                 device=train_audio_X.device)
    init_audio_extract = []
    for audio, init_time, l in zip(init_audio, initial_time, audio_length):
        shapelet_times = torch.linspace(init_time.item(), init_time.item() + l.item(), train_audio_X.size(1))
        audio_extract = torchshapelets._impl.unsafe_add_knots((audio_times[0], audio_times[1:-1], audio_times[-1]),
                                                              (audio[0], audio[1:-1], audio[-1]),
                                                              shapelet_times,
                                                              False)[1]
        init_audio_extract.append(audio_extract)
    init_audio_extract = torch.stack(init_audio_extract).squeeze(-1)

    # Initialise our candidate for inversion at the thing that has the closest MFCC. (This sort of thing is necessary as
    # we're solving an inverse problem here, so we have to use some sort of prior.)
    learnt_audio = torch.empty(10, 16000, requires_grad=True)
    with torch.no_grad():
        learnt_audio.copy_(init_audio_extract)

    # Apply SGD to match the MFCC of our candiate with the MFCC of the shapelet
    optim = torch.optim.SGD([learnt_audio], lr=1.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=1000, cooldown=1000,
                                                           verbose=True, min_lr=1e-3)
    mfcc_transform = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=40)
    scaling = torch.linspace(1, 0.5, 40) ** 2
    print('Starting inversion')
    trange = tqdm.trange(25_000)
    for i in trange:
        learnt_mfcc = mfcc_transform(learnt_audio).transpose(1, 2)
        learnt_mfcc = (learnt_mfcc - means) / (stds + 1e-5)
        diff = (learnt_mfcc - shapelet_mfcc) ** 2
        diff = diff.mean(dim=[0, 1])
        loss = diff.dot(scaling)  # matching lower frequencies is more important than higher frequencies
        # Regularise to be similar to the closest. Again, this corresponds to a prior. There _is_ a potential issue that
        # we just end up learning something that sounds like the init_audio, which we mitigate by taking a small scaling
        # factor, so that we should just end up selecting the thing that is most similar to init_audio along the
        # manifold of those things that match the MFCC, which is the more important criterion here.
        loss = loss + 0.001 * torch.nn.functional.mse_loss(learnt_audio, init_audio_extract)
        if i % 1000 == 0:
            trange.write("Epoch: {} Loss: {}".format(i, loss.item()))
        loss.backward()
        optim.step()
        scheduler.step(loss.item())
        optim.zero_grad()

    # Save results
    wav_length = 16000 * (80 / length)
    classes = ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go')
    for individual_audio, individual_wav_length, class_ in zip(learnt_audio.detach().numpy(), wav_length, classes):
        scipy.io.wavfile.write(class_ + '.wav', int(individual_wav_length), individual_audio)


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
         old_shapelets=False,                 # Whether to toggle off all of our innovations and use old-style shapelets
         save_top_logreg_shapelets=False):

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
                       old_shapelets,
                       save_top_logreg_shapelets)


def comparison_test(seed, old):
    common.handle_seeds(seed)
    main(result_folder='speech_commands',
         result_subfolder='old' if old else 'L2',
         old_shapelets=old)


def analyse():
    main(result_folder='speech_commands',
         result_subfolder='L2',
         epochs=0,
         old_shapelets=False)

if __name__ == '__main__':
    # invert('./results/shapelet_sc', find_closest=False)
    analyse()
