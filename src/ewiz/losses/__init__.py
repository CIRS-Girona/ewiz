from ewiz.core.utils import get_inheritors

# Base loss import
from .base import LossBase

# Image variance loss imports
from .variance.variance import ImageVariance
from .variance.normalized import NormalizedImageVariance
from .variance.multifocal import MultifocalNormalizedImageVariance

# Gradient magnitude loss imports
from .gradient.gradient import GradientMagnitude
from .gradient.normalized import NormalizedGradientMagnitude
from .gradient.multifocal import MultifocalNormalizedGradientMagnitude


# Import all losses
loss_functions = get_inheritors(parent=LossBase)

# TODO: Check photometric loss format
from .photometric import Photometric
from .smooth import LossSmoothness
from .hybrid import LossHybrid
from .mc import LossMotionCompensation
