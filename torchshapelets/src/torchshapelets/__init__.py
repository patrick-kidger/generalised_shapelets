import torch  # must be imported before anything from torchshapelets

from .discrepancies import CppDiscrepancy, L2Discrepancy, L2DiscrepancySquared, LogsignatureDiscrepancy
from .regularisation import similarity_regularisation
from .shapelet_transform import GeneralisedShapeletTransform

__version__ = '0.1.0'

del torch
