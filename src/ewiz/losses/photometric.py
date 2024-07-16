import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .base import LossBase

from typing import Any, Dict, List, Tuple, Callable, Union


class Photometric(LossBase):
    """Photometric loss.
    """
    name = "photometric"
    required_keys = ["flows", "gray_images", "weights"]

    def __init__(
        self,
        alpha: float = 0.45,
        epsilon: float = 0.001,
        use_smooth: bool = True,
        smooth_weight: float = 10.0,
        device: str = "cuda",
        direction: str = "minimize",
        store_history: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(direction, store_history, *args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.use_smooth = use_smooth
        self.smooth_weight = smooth_weight
        self.device = device

    @staticmethod
    def calculate_gradients(flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates flow gradients.
        """
        grad_x = flow[:, :, :, 1:] - flow[:, :, :, -1:]
        grad_y = flow[:, :, 1:] - flow[:, :, -1:]
        return grad_x, grad_y

    @staticmethod
    def calculate_charbonnier_loss(
        gray_error: torch.Tensor, alpha: float = 0.45, epsilon: float = 0.001
    ) -> torch.Tensor:
        """Calculates charbonnier loss.
        """
        loss = torch.mul(gray_error, gray_error) + torch.mul(epsilon, epsilon)
        loss = torch.pow(loss, alpha)
        loss = torch.mean(loss, dim=0)
        loss = torch.sum(loss)
        return loss

    def calculate_smoothness(self, flow: torch.Tensor) -> torch.Tensor:
        """Calculates smoothness loss.
        """
        grad_x, grad_y = self.calculate_gradients(flow)
        grad_xx, grad_xy = self.calculate_gradients(grad_x)
        grad_yx, grad_yy = self.calculate_gradients(grad_y)
        loss = (
            grad_xx.abs().float().mean()
            + grad_xy.abs().float().mean()
            + grad_yx.abs().float().mean()
            + grad_yy.abs().float().mean()
        )
        return loss

    def calculate_photometric(
        self, gray_images: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """Calculates photometric loss.
        """
        _, _, h, w = flow.shape
        images0 = gray_images[..., 0].to(self.device)
        images1 = gray_images[..., 1].to(self.device)
        images0 = T.Resize((h, w), antialias=True)(images0)
        images1 = T.Resize((h, w), antialias=True)(images1)
        warped_images = self._warp_gray_images(images1, flow)
        gray_error = warped_images - images0
        loss = self.calculate_charbonnier_loss(gray_error, self.alpha, self.epsilon)
        return loss

    def _warp_gray_images(
        self, gray_images: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """Warps grayscale images.
        """
        b, _, h, w = gray_images.size()
        gray_images = gray_images.float()
        grid_x = torch.arange(0, w).view(1, -1).reshape(h, 1)
        grid_y = torch.arange(0, h).view(-1, 1).reshape(1, w)
        grid_x = grid_x.view(1, 1, h, w).repeat(b, 1, 1, 1)
        grid_y = grid_y.view(1, 1, h, w).repeat(b, 1, 1, 1)
        grid = torch.cat((grid_x, grid_y), dim=1)
        grid = grid.to(self.device)
        grid = grid + flow

        # Normalize between -1 and 1
        grid[:, 0, :, :] = 2.0*grid[:, 0, :, :].clone()/max(w - 1, 1) - 1.0
        grid[:, 1, :, :] = 2.0*grid[:, 1, :, :].clone()/max(h - 1, 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)

        # Warp images
        warped_images = F.grid_sample(gray_images, grid=grid, align_corners=True)

        # Apply mask
        mask = torch.ones(gray_images.size()).to(self.device)
        mask = F.grid_sample(mask, grid=grid, align_corners=True)
        mask[mask < 0.99999] = 0
        mask[mask > 0] = 1
        warped_images = warped_images*mask
        return warped_images

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(
        self,
        flows: List[torch.Tensor],
        gray_images: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        """Calculates loss function.
        """
        total_loss = 0.0
        smooth_weight = self.smooth_weight
        for i, flow in enumerate(flows):
            if self.use_smooth:
                smooth_loss = self.calculate_smoothness(flow)*smooth_weight
                total_loss += smooth_loss
                smooth_weight /= 2.0
            photo_loss = self.calculate_photometric(gray_images, flow)*weights[i]
            total_loss += photo_loss
        if self.direction == "minimize":
            return total_loss
        return -total_loss
