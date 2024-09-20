import torch
import numpy as np

from .base import ImagerBase

from typing import Any, Dict, List, Tuple, Callable, Union


class ImagerBilinear(ImagerBase):
    """Bilinear imager class.
    """
    name = "bilinear"

    def __init__(
        self,
        image_size: Tuple[int, int],
        image_padding: Tuple[int, int] = (0, 0)
    ) -> None:
        super().__init__(image_size, image_padding)
        self.iwe_func = self._bilinear

    def _bilinear(
        self,
        events: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        """Applies a bilinear operation to create an IWE.
        """
        # Check dimensional compatibility
        if type(weights) == torch.Tensor:
            assert(weights.shape == events.shape[:-1]), (
                f"Shape of events data of size '{events.shape}' "
                f"is incompatible with shape of weights of size '{weights.shape}'."
            )
        # Add batch dimension
        if len(events.shape) == 2:
            events = events[None, ...]
        # Get image properties
        batch_size = len(events)
        h_padding, w_padding = self.image_padding
        h, w = self.image_size

        iwe = events.new_zeros((batch_size, h*w))
        coords = torch.floor(events[..., :2] + 1e-6)
        diff_coords = events[..., :2] - coords
        xs = coords[..., 0] + w_padding
        ys = coords[..., 1] + h_padding

        pos_ids = torch.cat(
            [
                 xs      +  ys     *w,
                 xs      + (ys + 1)*w,
                (xs + 1) +  ys     *w,
                (xs + 1) + (ys + 1)*w
            ],
            dim=-1
        )
        mask_ids = torch.cat(
            [
                (0 <= xs)    *(xs < w)    *(0 <= ys)    *(ys < h),
                (0 <= xs)    *(xs < w)    *(0 <= ys + 1)*(ys + 1 < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys)    *(ys < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys + 1)*(ys + 1 < h)
            ],
            dim=-1
        )
        pos0 = (1 - diff_coords[..., 1])*(1 - diff_coords[..., 0])*weights
        pos1 = (diff_coords[..., 1])    *(1 - diff_coords[..., 0])*weights
        pos2 = (1 - diff_coords[..., 1])*(diff_coords[..., 0])    *weights
        pos3 = (diff_coords[..., 1])    *(diff_coords[..., 0])    *weights
        pixel_vals = torch.cat([pos0, pos1, pos2, pos3], dim=-1)
        pos_ids = (pos_ids*mask_ids).long()
        pixel_vals = pixel_vals*mask_ids
        iwe.scatter_add_(1, pos_ids, pixel_vals)
        iwe = iwe.reshape((batch_size,) + self.image_size).squeeze()
        return iwe
