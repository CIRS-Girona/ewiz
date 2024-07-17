import torch

from ..base import LossBase
from ..sobel import Sobel

from typing import Any, Dict, List, Tuple, Callable, Union


class GradientMagnitude(LossBase):
    """Gradient magnitude loss function.
    """
    name = "gradient_magnitude"
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
        self.sobel = Sobel(1, 3, precision, device)

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(self, iwe: torch.Tensor, omit_bounds: bool) -> torch.Tensor:
        """Calculates loss function.
        """
        if len(iwe.shape) == 2:
            iwe = iwe[None, None, ...]
        elif len(iwe.shape) == 3:
            iwe = iwe[:, None, ...]
        if self.precision == "64":
            iwe = iwe.double()

        sobel_image = self.sobel.forward(iwe)/8.0
        sobel_x = sobel_image[:, 0]
        sobel_y = sobel_image[:, 1]
        if omit_bounds:
            sobel_x = sobel_x[..., 1:-1, 1:-1]
            sobel_y = sobel_y[..., 1:-1, 1:-1]

        loss = torch.mean(torch.square(sobel_x) + torch.square(sobel_y))
        if self.direction == "minimize":
            return -loss
        return loss
