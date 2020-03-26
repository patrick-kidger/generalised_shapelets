import os
import pathlib
import urllib.request
import tarfile


here = pathlib.Path(__file__).resolve().parent


def main():
    base_loc = str(here / '../experiments/data/SpeechCommands')
    loc = base_loc + '/speech_commands.tar.gz'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                               loc)

    with tarfile.open(loc, 'r') as f:
        f.extractall(base_loc)


if __name__ == '__main__':
    main()
