import torch

from ..base import LossBase

from typing import Any, Dict, List, Tuple, Callable, Union


class ImageVariance(LossBase):
    """Image variance loss function.
    """
    name = "image_variance"
    required_keys = ["iwe", "omit_bounds"]

    def __init__(
        self,
        direction: str = "minimize",
        store_history: bool = False,
        precision: str = "64",
        device: str = "cuda"
    ) -> None:
        super().__init__(direction, store_history)
        self.precision = precision
        self.device = device

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(self, iwe: torch.Tensor, omit_bounds: bool, *args, **kwargs) -> torch.Tensor:
        """Calculates loss function.
        """
        if len(iwe.shape) == 2:
            iwe = iwe[None, None, ...]
        elif len(iwe.shape) == 3:
            iwe = iwe[:, None, ...]
        if self.precision == "64":
            iwe = iwe.double()
        if omit_bounds:
            iwe = iwe[..., 1:-1, 1:-1]

        loss = torch.var(iwe)
        if self.direction == "minimize":
            return -loss
        return loss
