import torch  # must be imported before anything from torchshapelets

from .discrepancies import L2Discrepancy, LogsignatureDiscrepancy
from .regularisation import similarity_regularisation, length_regularisation, pseudometric_regularisation
from .shapelet_transform import GeneralisedShapeletTransform

from ._impl import parallel_testing

__version__ = '0.1.0'

del torch
