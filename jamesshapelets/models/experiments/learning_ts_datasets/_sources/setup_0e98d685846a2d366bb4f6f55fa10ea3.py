"""
setup.py
================================================
Various method used in setting up and running sacred experiments.
"""
from jamesshapelets.definitions import *
import os, shutil
from pprint import pprint
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def create_fso(ex, directory, remove_folder=False):
    """
    Creates a file storage observer for a given experiment in the specified directory.

    Check sacred docs for a full explanation but this just sets up the folder to save the information from the runs of
    the given experiment.

    NOTE: This is currently setup to delete/replace the specified folder if it already exists. This should be changed if
    it is not the desired behaviour.

    Args:
        ex (sacred.Experiment): The sacred experiment to be tracked.
        directory (str): The directory to 'watch' the experiment and save information to.

    Returns:
        None
    """
    if remove_folder:
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
    ex.observers.append(FileStorageObserver(directory))


def ready_mongo_observer(ex, db_name='sacred', url='localhost:27017'):
    """Readies a mongo observer for use with sacred.

    Args:
        ex (sacred.Experiment): Sacred experiment to track.
        db_name (str): Name of the mongo database.
        url (str): Host location.
    """
    ex.observers.append(MongoObserver(url=url, db_name=db_name))


@timeit
def basic_gridsearch(ex, grid, tqdm_progress=False):
    """Basic gridsearch for a sacred experiment.

    Given an experiment and a parameter grid, this will iterate over all possible combinations of parameters specified
    in the grid. In an iteration, the experiment configuration is updated and the experiment is run.

    Args:
        ex (sacred.Experiment): A sacred experiment.
        grid (dict): Parameter grid, analogous setup to as with sklearn gridsearches.
        tqdm_progress (bool): Set True to tqdm the loop.
        parallel (bool): Set True for a parallel processing loop.

    Returns:
        None
    """
    # Loop with tqdm if specified
    tqdm_func = tqdm if tqdm_progress else lambda x: x

    # Setup the grid
    param_grid = ParameterGrid(grid)
    grid_len = len(param_grid)

    # with tqdm_func(total=grid_len) as progress_bar:
    for i, params in enumerate(param_grid):
        print('\n\n\nCONFIGURATION {} of {}\n'.format(i+1, grid_len) + '-' * 100)
        pprint(params)
        print('-' * 100)
        # Update conf
        ex.add_config(**params)
        # Get going
        ex.run()
            # progress_bar.update(1)
