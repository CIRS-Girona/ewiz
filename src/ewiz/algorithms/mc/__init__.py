"""This module is responsible for the motion compensation algorithm.
"""
from ewiz.core import get_inheritors

from .base import MotionCompensationBase
from .pyramidal import MotionCompensationPyramidal

mc_functions = get_inheritors(MotionCompensationBase)
