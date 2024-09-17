from ewiz.core.utils import get_inheritors

from .base import WarperBase
from .dense import WarperDense


warper_functions = get_inheritors(WarperBase)
