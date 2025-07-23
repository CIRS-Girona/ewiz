import numpy as np

from .base import EncoderBase

from typing import Any, Dict, List, Tuple, Callable, Union


class EncoderStats(EncoderBase):
    """Encoder statistics."""

    name = "stats"

    def __init__(self, image_size: Tuple[int, int], *args, **kwargs) -> None:
        super().__init__(image_size)

    def encode(self, events: np.ndarray) -> np.ndarray:
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2].astype(np.float32)
        p = events[:, 3].astype(np.int8)

        # Normalize timestamps
        t_min = t.min()
        t_max = t.max()
        t_range = t_max - t_min if t_max > t_min else 1.0
        t_norm = (t - t_min) / t_range

        data = np.zeros((4, self.image_size[0], self.image_size[1]))

        mask_pos = p > 0
        if np.any(mask_pos):
            np.add.at(data[0], (y[mask_pos], x[mask_pos]), 1.0)
            np.maximum.at(data[2], (y[mask_pos], x[mask_pos]), t_norm[mask_pos])
        mask_neg = p == 0
        if np.any(mask_neg):
            np.add.at(data[1], (y[mask_neg], x[mask_neg]), 1.0)
            np.maximum.at(data[3], (y[mask_neg], x[mask_neg]), t_norm[mask_neg])

        return data
