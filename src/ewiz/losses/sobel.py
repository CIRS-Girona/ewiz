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
        super().__init__()
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

        self.filter_dx = nn.Conv2d(in_channels, 1, kernel_size, 1, 1, bias=False)
        self.filter_dy = nn.Conv2d(in_channels, 1, kernel_size, 1, 1, bias=False)
        self.filter_dx.weight = nn.Parameter(kernel_dx.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.filter_dy.weight = nn.Parameter(kernel_dy.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        """
        if self.in_channels == 1:
            sobel_x = self.filter_dx(x[..., [0], :, :])
            sobel_y = self.filter_dy(x[..., [0], :, :])
            return torch.cat([sobel_x, sobel_y], dim=1)
        if self.in_channels == 2:
            sobel_xx = self.filter_dx(x[..., [0], :, :])
            sobel_yy = self.filter_dy(x[..., [1], :, :])
            sobel_xy = self.filter_dx(x[..., [1], :, :])
            sobel_yx = self.filter_dy(x[..., [0], :, :])
            return torch.cat([sobel_xx, sobel_yy, sobel_xy, sobel_yx], dim=1)
