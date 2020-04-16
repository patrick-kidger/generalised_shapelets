import os
import pathlib
import urllib.request
import zipfile


here = pathlib.Path(__file__).resolve().parent


def main():
    base_loc = here / '../experiments/data/HumanActivity'
    loc = base_loc / 'human_activity.zip'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip'
    urllib.request.urlretrieve(url, loc)

    with zipfile.ZipFile(loc, 'r') as zip_ref:
        zip_ref.extractall(base_loc)


if __name__ == '__main__':
    main()
