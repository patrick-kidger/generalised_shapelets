import os
import pathlib
import urllib.request
import zipfile


here = pathlib.Path(__file__).resolve().parent


def main():
    base_loc = str(here / '../experiments/data/UEA')
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
