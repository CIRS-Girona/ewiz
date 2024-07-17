import torch

from ..base import LossBase

from typing import Any, Dict, List, Tuple, Callable, Union


class NormalizedImageVariance(LossBase):
    """Normalized image variance loss function.
    """
    name = "normalized_image_variance"
    required_keys = ["ie", "iwe", "omit_bounds"]

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
    def calculate(
        self,
        ie: torch.Tensor,
        iwe: torch.Tensor,
        omit_bounds: bool
    ) -> torch.Tensor:
        """Calculates loss function.
        """
        if len(iwe.shape) == 2:
            ie = ie[None, None, ...]
            iwe = iwe[None, None, ...]
        elif len(iwe.shape) == 3:
            ie = ie[:, None, ...]
            iwe = iwe[:, None, ...]
        if self.precision == "64":
            ie = ie.double()
            iwe = iwe.double()
        if omit_bounds:
            ie = ie[..., 1:-1, 1:-1]
            iwe = iwe[..., 1:-1, 1:-1]

        loss_ie = torch.var(ie)
        loss_iwe = torch.var(iwe)
        if self.direction == "minimize":
            return loss_ie/loss_iwe
        return loss_iwe/loss_ie
