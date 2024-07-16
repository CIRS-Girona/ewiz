import torch
import torch.nn as nn

from typing import Any, Dict, List, Tuple, Callable, Union


class Sobel(nn.Module):
    """Sobel PyTorch operator.
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        precision: str = "64",
        device: str = "cuda"
    ) -> None:
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.precision = precision
        self.device = device

        kernel_dx = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0]
        ]).to(self.device)
        kernel_dy = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ]).to(self.device)
        if precision == "64":
            kernel_dx = kernel_dx.double()
            kernel_dy = kernel_dy.double()
        # TODO: Continue code here
