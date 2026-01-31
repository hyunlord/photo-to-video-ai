"""
RIFE (Real-Time Intermediate Flow Estimation) Module

Frame interpolation using optical flow estimation.
Based on ECCV2022-RIFE: https://github.com/hzwer/ECCV2022-RIFE
"""

from .model import Model
from .IFNet import IFNet
from .warplayer import warp

__all__ = ['Model', 'IFNet', 'warp']
