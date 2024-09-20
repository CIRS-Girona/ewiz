import torch

from .base import LossBase
from .sobel import Sobel

from typing import Any, Dict, List, Tuple, Callable, Union


class LossSmoothness(LossBase):
    """Smoothness loss.
    """
    name = "smoothness"
    required_keys = ["flow", "omit_bounds"]

    def __init__(
        self,
        direction: str = "minimize",
        store_history: bool = False,
        precision: str = "64",
        device: str = "cuda",
        *args,
        **kwargs
    ) -> None:
        super().__init__(direction, store_history, *args, **kwargs)
        self.precision = precision
        self.device = device
        self.sobel = Sobel(2, 3, self.precision, self.device)

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(self, flow: torch.Tensor, omit_bounds: bool, *args, **kwargs) -> torch.Tensor:
        """Calculates loss function.
        """
        if len(flow.shape) == 3:
            flow = flow[None, ...]
        if self.precision == "64":
            flow = flow.double()
        sobel_image = self.sobel.forward(flow)/8.0
        if omit_bounds:
            if sobel_image.shape[2] > 2 and sobel_image.shape[3] > 2:
                sobel_image = sobel_image[..., 1:-1, 1:-1]
        loss = torch.mean(torch.abs(sobel_image))
        if self.direction == "minimize":
            return loss
        return -loss
