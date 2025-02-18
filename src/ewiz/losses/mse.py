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
        self.mse_loss = nn.MSELoss(reduction=None)
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

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(
        self,
        pred: List[torch.Tensor],
        gt: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates loss function."""
        return None
