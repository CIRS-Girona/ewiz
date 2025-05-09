import torch

from ..base import LossBase
from .normalized import NormalizedImageVariance

from typing import Any, Dict, List, Tuple, Callable, Union


class MultifocalNormalizedImageVariance(LossBase):
    """Multifocal normalized image variance.
    """
    name = "multifocal_normalized_image_variance"
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
        self.norm_image_var = NormalizedImageVariance(direction, store_history, precision, device)

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(
        self,
        ie: torch.Tensor,
        start_iwe: torch.Tensor,
        mid_iwe: torch.Tensor,
        end_iwe: torch.Tensor,
        omit_bounds: bool,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Calculates loss function.
        """
        loss_start = self.norm_image_var.calculate(ie=ie, iwe=start_iwe, omit_bounds=omit_bounds)
        loss_mid = self.norm_image_var.calculate(ie=ie, iwe=mid_iwe, omit_bounds=omit_bounds)
        loss_end = self.norm_image_var.calculate(ie=ie, iwe=end_iwe, omit_bounds=omit_bounds)
        loss = loss_start + 2*loss_mid + loss_end
        if self.direction == "minimize":
            return loss
        return -loss
