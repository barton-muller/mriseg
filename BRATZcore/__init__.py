"""
Top-level package for cardiac MRI segmentation.

Submodules:
  - configuration:   model & training configuration structures
  - utils:           general helper functions (data I/O, metrics, transforms)
  - model:           network architectures (e.g. UNet, blocks)
  - prediction:      inference routines & batch prediction helpers
  - training:        training loops, loss functions, validation logic
"""

# expose OneDrive path
from .configuration import *  

# expose utility functions: data loading, preprocessing, metrics, visualization
from .utils import *          

# expose model definitions (UNet, blocks, etc.)
from .model import *          

# expose prediction/inference
from .prediction import *     

# expose training loop, loss, and validators
from .training import *