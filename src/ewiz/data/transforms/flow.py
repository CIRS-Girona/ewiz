import numpy as np
import torch

from dataclasses import dataclass
from scipy.ndimage import rotate

from typing import Any, Dict, List, Tuple, Callable, Union


@dataclass(frozen=True)
class FlowRandomHorizontalFlip():
    """Randomly flips flow around the horizontal axis.
    """
    prob: float = 0.5

    def __post_init__(self) -> None:
        """Post-initialization.
        """
        assert 0 <= self.prob <= 1, (
            "Flipping probability should be between 0 and 1. "
            f"Got '{self.prob}' instead."
        )

    def __call__(self, flow: np.ndarray) -> np.ndarray:
        """Call function.
        """
        flow = flow.copy()
        if np.random.rand() <= self.prob:
            flow = np.flip(flow, axis=2)
            return flow
        return flow


@dataclass(frozen=True)
class FlowRandomVerticalFlip():
    """Randomly flips flow around the vertical axis.
    """
    prob: float = 0.5

    def __post_init__(self) -> None:
        """Post-initialization.
        """
        assert 0 <= self.prob <= 1, (
            "Flipping probability should be between 0 and 1. "
            f"Got '{self.prob}' instead."
        )

    def __call__(self, flow: np.ndarray) -> np.ndarray:
        """Call function.
        """
        flow = flow.copy()
        if np.random.rand() <= self.prob:
            flow = np.flip(flow, axis=1)
            return flow
        return flow


@dataclass(frozen=True)
class FlowCenterCrop():
    """Flow center crop.
    """
    out_size: Tuple[int, int]

    def __call__(self, flow: np.ndarray) -> np.ndarray:
        """Call function.
        """
        flow = flow.copy()
        offsets = (
            int((flow.shape[1] - self.out_size[0])/2),
            int((flow.shape[2] - self.out_size[1])/2)
        )
        flow = flow[:, offsets[0]:-offsets[0], offsets[1]:-offsets[1]]
        return flow


@dataclass(frozen=True)
class FlowRandomCrop():
    """Flow random crop.
    """
    out_size: Tuple[int, int]

    def __call__(self, flow: np.ndarray) -> np.ndarray:
        """Call function.
        """
        flow = flow.copy()
        # Compute x-axis indices
        x_index0 = int(np.random.rand()*(flow.shape[2] - self.out_size[1]))
        x_index1 = x_index0 + self.out_size[1]
        # Compute y-axis indices
        y_index0 = int(np.random.rand()*(flow.shape[1] - self.out_size[0]))
        y_index1 = y_index0 + self.out_size[0]
        flow = flow[:, y_index0:y_index1, x_index0:x_index1]
        return flow
