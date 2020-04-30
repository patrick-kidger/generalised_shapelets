import os
import pathlib
import sklearn.model_selection
import tarfile
import torch
import torchaudio
import urllib.request

here = pathlib.Path(__file__).resolve().parent


def _split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (train_tensor, testval_tensor,
     train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                  train_size=0.7,
                                                                                  random_state=0,
                                                                                  shuffle=True,
                                                                                  stratify=stratify)

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                       train_size=0.5,
                                                                       random_state=1,
                                                                       shuffle=True,
                                                                       stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor


def _save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


def download():
    base_base_loc = str(here / '../experiments/data')
    if not os.path.exists(base_base_loc):
        raise RuntimeError("data directory does not exist. Please create a directory called 'data' in the 'experiments'"
                           " directory. (We're going to put a lot of data there, so we don't make it automatically - "
                           "thus giving you the opportunity to make it a symlink rather than a normal directory, so "
                           "that the data can be stored elsewhere if you wish.)")
    base_loc = base_base_loc + '/SpeechCommands'
    loc = base_loc + '/speech_commands.tar.gz'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                               loc)

    with tarfile.open(loc, 'r') as f:
        f.extractall(base_loc)


def _process_data():
    base_loc = here / '..' / 'experiments' / 'data' / 'SpeechCommands'
    X = torch.empty(34975, 16000, 1)
    y = torch.empty(34975, dtype=torch.long)

    batch_index = 0
    y_index = 0
    for foldername in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
        loc = base_loc / foldername
        for filename in os.listdir(loc):
            audio, _ = torchaudio.load_wav(loc / filename, channels_first=False,
                                           normalization=False)  # for forward compatbility if they fix it
            audio = audio / 2 ** 15  # Normalization argument doesn't seem to work so we do it manually.

            # A few samples are shorter than the full length; for simplicity we discard them.
            if len(audio) != 16000:
                continue

            X[batch_index] = audio
            y[batch_index] = y_index
            batch_index += 1
        y_index += 1
    assert batch_index == 34975, "batch_index is {}".format(batch_index)

    # X is of shape (batch=34975, length=16000, channels=1)
    X = torchaudio.transforms.MFCC(log_mels=True)(X.squeeze(-1)).transpose(1, 2).detach()
    # X is of shape (batch=34975, length=81, channels=40). For some crazy reason it requires a gradient, so detach.

    train_X, _, _ = _split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        mean = train_Xi.mean()
        std = train_Xi.std()
        out.append((Xi - mean) / (std + 1e-5))
    X = torch.stack(out, dim=-1)

    train_X, val_X, test_X = _split_data(X, y)
    train_y, val_y, test_y = _split_data(y, y)

    return train_X, val_X, test_X, train_y, val_y, test_y


def main():
    download()
    train_X, val_X, test_X, train_y, val_y, test_y = _process_data()
    loc = here / '..' / 'experiments' / 'speech_commands_data'
    if not os.path.exists(loc):
        os.mkdir(loc)
    # _save_data(loc, train_X=train_X, val_X=val_X, test_X=test_X, train_y=train_y, val_y=val_y, test_y=test_y)


if __name__ == '__main__':
    main()
