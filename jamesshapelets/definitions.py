"""
This file contains basic variables and definitions that we wish to make easily accessible for any script that requires
it.

from definitions import *
"""
from pathlib import Path

ROOT_DIR = '/'.join(str(Path(__file__).resolve().parents[0]).split('/'))
DATA_DIR = ROOT_DIR + '/data'
MODELS_DIR = ROOT_DIR + '/models'

IS_HAVOK = False
if ROOT_DIR == '/home/morrill/Documents/generalised_shapelets/jamesshapelets':
    IS_HAVOK = True

if IS_HAVOK:
    DATA_DIR = '/scratch/morrill/generalised_shapelets/data/jamesdata/data'
    MODELS_DIR = ROOT_DIR + '/models/havok_models'

# Packages/functions used everywhere
from jamesshapelets.src.omni.decorators import *
from jamesshapelets.src.omni.functions import *
