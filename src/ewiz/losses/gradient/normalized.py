import torch

from ..base import LossBase
from .gradient import GradientMagnitude

from typing import Any, Dict, List, Tuple, Callable, Union


class NormalizedGradientMagnitude(LossBase):
    """Normalized gradient magnitude loss function.
    """
    name = "normalized_gradient_magnitude"
    required_keys = ["ie", "iwe", "omit_bounds"]

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
        self.grad_mag = GradientMagnitude(direction, store_history, precision, device)
        # TODO: Continue code here
