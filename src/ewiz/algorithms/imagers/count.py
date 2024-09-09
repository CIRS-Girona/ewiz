import torch
import numpy as np

from .base import ImagerBase

from typing import Any, Dict, List, Tuple, Callable, Union


class ImagerCount(ImagerBase):
    """Count imager class.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        image_padding: Tuple[int, int] = (0, 0)
    ) -> None:
        super().__init__(image_size, image_padding)
        self.iwe_func = self._count

    def _count(self, events: torch.Tensor) -> torch.Tensor:
        """Applies a count operation to create an IWE.
        """
        # Add batch dimension
        if len(events.shape) == 2:
            events = events[None, ...]
        # Get image properties
        batch_size = len(events)
        h_padding, w_padding = self.image_padding
        h, w = self.image_size

        iwe = events.new_zeros((batch_size, h*w))
        coords = torch.floor(events[..., :2] + 1e-6)
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
        pixel_vals = torch.ones_like(pos_ids)
        pos_ids = (pos_ids*mask_ids).long()
        pixel_vals = pixel_vals*mask_ids
        iwe.scatter_add_(1, pos_ids, pixel_vals)
        iwe = iwe.reshape((batch_size,), self.image_size).squeeze()
        return iwe
