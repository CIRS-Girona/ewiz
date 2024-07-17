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

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(
        self,
        ie: torch.Tensor,
        iwe: torch.Tensor,
        omit_bounds: bool
    ) -> torch.Tensor:
        """Calculates loss function.
        """
        loss_iwe = self.grad_mag.calculate(iwe=iwe, omit_bounds=omit_bounds)
        loss_ie = self.grad_mag.calculate(iwe=ie, omit_bounds=omit_bounds)
        if self.direction == "minimize":
            return loss_ie/loss_iwe
        return loss_iwe/loss_ie
