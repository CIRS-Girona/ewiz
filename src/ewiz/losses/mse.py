import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .base import LossBase

from typing import Any, Dict, List, Tuple, Callable, Union


class MultiScaleMSE(LossBase):
    """Multi-scale MSE loss."""

    name = "mse"
    required_keys = ["pred", "gt"]

    def __init__(
        self,
        alpha: float = 0.45,
        epsilon: float = 0.001,
        device: str = "cuda",
        direction: str = "minimize",
        store_history: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(direction, store_history, *args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        self.mse_loss = nn.MSELoss(reduce=False)
        self.read_flag = False
        self.torch_resize = None

    @staticmethod
    def calculate_charbonnier_loss(
        error: torch.Tensor, alpha: float = 0.45, epsilon: float = 0.001
    ) -> torch.Tensor:
        """Calculates charbonnier loss."""
        loss = torch.mul(error, error) + torch.mul(epsilon, epsilon)
        loss = torch.pow(loss, alpha)
        loss = torch.mean(loss, dim=0)
        loss = torch.sum(loss)
        return loss

    def _calculate_mse(
        self, pred: torch.Tensor, gt: torch.Tensor, scale_factor: float
    ) -> torch.Tensor:
        """Calculates MSE."""
        _, _, h, w = pred.shape
        resized_gt = T.Resize((h, w), antialias=True)(gt) / scale_factor
        flow_error = self.mse_loss(pred, resized_gt)
        loss = self.calculate_charbonnier_loss(flow_error, self.alpha, self.epsilon)
        return loss

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(
        self,
        preds: List[torch.Tensor],
        gt: torch.Tensor,
        weights: Union[float, torch.Tensor] = [1.0, 1.0, 1.0, 1.0],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates loss function."""
        total_loss = 0.0
        for i, pred in enumerate(preds):
            scale_factor = gt.shape[-1] / pred.shape[-1]
            mse_loss = self._calculate_mse(pred, gt, scale_factor) * weights[i]
            total_loss += mse_loss
        if self.direction == "minimize":
            return total_loss
        return -total_loss
