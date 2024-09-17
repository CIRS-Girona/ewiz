from ewiz.core.utils import get_inheritors

from .base import ImagerBase
from .count import ImagerCount
from .bilinear import ImagerBilinear


imager_functions = get_inheritors(ImagerBase)
