"""
main.py
=========================
Main experiment runner. The aim is for everything to be run through this file with different model configurations
imported via config
"""
from definitions import *
from sacred import Experiment
import logging
from src.data.make_dataset import UcrDataset
from src.features.build_features import apply_augmentation_list
from src.experiments.setup import create_fso, basic_gridsearch

import warnings
warnings.simplefilter('ignore', UserWarning)

# Experiment setup
ex_name = 'test'
ex = Experiment(ex_name)
save_dir = MODELS_DIR + '/experiments/{}'.format(ex_name)


# Configuration, setup parameters that can vary here
@ex.config
def my_config():
    pass


# Main run file
@ex.main
def main(_run, ds_name):
    # Add in save_dir
    _run.save_dir = save_dir + '/' + _run._id

    # Get model training datsets
    dataset = UcrDataset(ds_name)


if __name__ == '__main__':
    # Create FSO (this creates a folder to log information into).
    create_fso(ex, save_dir)

    # Run a gridsearch over all parameter combinations.
    basic_gridsearch(ex, config['param_grid'], tqdm_progress=True)
