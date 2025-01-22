from ewiz.core.utils import get_inheritors

from .base import EncoderBase
from .sum import EncoderSum
from .gaussian import EncoderGaussian

data_encoders = get_inheritors(parent=EncoderBase)
