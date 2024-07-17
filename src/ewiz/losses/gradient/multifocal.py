import torch

from ..base import LossBase
from .normalized import NormalizedGradientMagnitude

from typing import Any, Dict, List, Tuple, Callable, Union


class MultifocalNormalizedGradientMagnitude(LossBase):
    """Multifocal normalized gradient magnitude.
    """
    name = "multifocal_normalized_gradient_magnitude"
    required_keys = ["ie", "start_iwe", "mid_iwe", "end_iwe", "omit_bounds"]

    def __init__(
        self,
        direction: str = "minimize",
        store_history: bool = False,
        precision: str = "64",
        device: str = "cuda",
        *args,
        **kwargs
    ) -> None:
        super().__init__(direction, store_history)
        # TODO: Continue code here
