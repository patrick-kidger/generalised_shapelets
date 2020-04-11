import os
import pathlib
import urllib.request
import zipfile


here = pathlib.Path(__file__).resolve().parent


def main():
    base_base_loc = str(here / '../experiments/data')
    if not os.path.exists(base_base_loc):
        raise RuntimeError("data directory does not exist. Please create a directory called 'data' in the 'experiments'"
                           " directory. (We're going to put a lot of data there, so we don't make it automatically - "
                           "thus giving you the opportunity to make it a symlink rather than a normal directory, so "
                           "that the data can be stored elsewhere if you wish.)")
    base_loc = base_base_loc + 'UEA'
    loc = base_loc + '/Multivariate2018_ts.zip'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip',
                               loc)

    with zipfile.ZipFile(loc, 'r') as f:
        f.extractall(base_loc)


if __name__ == '__main__':
    main()
