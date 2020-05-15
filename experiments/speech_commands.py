import os
import pathlib
import random
import scipy.io.wavfile
import torch
import torchaudio

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


def invert(model_filename, wav_filename):
    """Inverts the MFCC shapelet to find the corresponding audio shapelet."""

    # Get the shapelet we're going to invert
    state_dict = torch.load(here / 'results/speech_commands' / model_filename)
    weight = state_dict['linear.weight']
    num_classes, num_shapelets = weight.shape
    most_informative = weight.argmin(dim=1)

    shapelets = state_dict['shapelet_transform.shapelets']
    mfcc_shapelet = shapelets[shapelet_index].to('cpu')
    lengths = state_dict['shapelet_transform.lengths']
    length = lengths[shapelet_index]

    # Get the normalisation constants we're applying to the dataset
    tensors = _load_data(here / 'data/speech_commands_data')
    means = tensors['means']
    stds = tensors['stds']

    # Available classes that we're considering
    classes = ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go')
    print(classes[class_index])

    # Initialise our possible audio sample, that we're going to find by SGD. To help with convergence we initialise it
    # as some random perturbation from a training sample - corresponding to a prior on natural speech, necessary because
    # this is an inverse problem and therefore ill posed - but to make sure that whatever we find isn't because of this
    # initialisation, we use a sample from a _different_ class to whichever our shapelet corresponds.
    to_learn = torch.empty(16000, requires_grad=True)
    classes_set = set(classes)
    classes_set.remove(classes[class_index])
    init_audio = _get_sample(random.choice(list(classes_set)))
    with torch.no_grad():
        to_learn.copy_(init_audio)
        to_learn += torch.randn(16000) * 0.1

    optim = torch.optim.SGD([to_learn], lr=1.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=1000, cooldown=1000,
                                                           verbose=True, min_lr=1e-3)
    mfcc_transform = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=128)
    scaling = torch.linspace(0.0001, 0.001, 128 - 40)
    for high in (False, True):
        try:
            for i in range(100_000):
                mfcc_to_learn = mfcc_transform(to_learn.unsqueeze(0)).transpose(1, 2).squeeze(0)
                # lower frequencies we compare against our target
                # higher frequencies we regularise to be close to zero
                lower_frequencies, higher_frequencies = mfcc_to_learn[:, :40], mfcc_to_learn[:, 40:]
                lower_frequencies = (lower_frequencies - means) / (stds + 1e-5)
                loss = torch.nn.functional.mse_loss(lower_frequencies, mfcc_shapelet)
                if high:
                    loss = loss + (scaling * higher_frequencies ** 2).sum()
                if i % 1000 == 0:
                    print("Epoch: {} Loss: {}".format(i, loss.item()))
                loss.backward()
                optim.step()
                scheduler.step(loss.item())
                optim.zero_grad()
        except (KeyboardInterrupt, RuntimeError):  # RuntimeError from interupting the JIT
            pass
        if high is False:
            print('Switching.')

    length = int(16000 * (80 / length))
    scipy.io.wavfile.write(wav_filename, length, to_learn.detach().numpy())


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

    return times, train_dataloader, val_dataloader, test_dataloader, tensors['means'], tensors['stds']


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

    times, train_dataloader, val_dataloader, test_dataloader, _, _ = get_data()

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


def hyperparameter_search(num_shapelets_per_class):
    result_folder = 'speech_commands_hyperparameter_search'
    for max_shapelet_length_proportion in (0.15, 0.3, 0.5, 1.0):
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
