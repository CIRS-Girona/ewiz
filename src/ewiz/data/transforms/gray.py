import numpy as np
import torch

from dataclasses import dataclass
from scipy.ndimage import rotate

from typing import Any, Dict, List, Tuple, Callable, Union


@dataclass(frozen=True)
class GrayRandomHorizontalFlip():
    """Randomly flips grayscale images around the horizontal axis.
    """
    prob: float = 0.5

    def __post_init__(self) -> None:
        """Post-initialization.
        """
        assert 0 <= self.prob <= 1, (
            "Flipping probability should be between 0 and 1. "
            f"Got '{self.prob}' instead."
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call function.
        """
        image = image.copy()
        if np.random.rand() <= self.prob:
            image = np.flip(image, axis=1)
            return image
        return image


@dataclass(frozen=True)
class GrayRandomVerticalFlip():
    """Randomly flips grayscale images around the vertical axis.
    """
    prob: float = 0.5

    def __post_init__(self) -> None:
        """Post-initialization.
        """
        assert 0 <= self.prob <= 1, (
            "Flipping probability should be between 0 and 1. "
            f"Got '{self.prob}' instead."
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call function.
        """
        image = image.copy()
        if np.random.rand() <= self.prob:
            image = np.flip(image, axis=0)
            return image
        return image


@dataclass(frozen=True)
class GrayCenterCrop():
    """Gray center crop.
    """
    out_size: Tuple[int, int]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call function.
        """
        image = image.copy()
        offsets = (
            int((image.shape[1] - self.out_size[1])/2),
            int((image.shape[0] - self.out_size[0])/2)
        )
        image = image[offsets[0]:-offsets[0], offsets[1]:-offsets[1]]
        return image


@dataclass(frozen=True)
class GrayRandomCrop():
    """Gray random crop.
    """
    out_size: Tuple[int, int]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call function.
        """
        image = image.copy()
        # Compute x-axis indices
        x_index0 = int(np.random.rand()*(image.shape[1] - self.out_size[1]))
        x_index1 = x_index0 + self.out_size[1]
        # Compute y-axis indices
        y_index0 = int(np.random.rand()*(image.shape[0] - self.out_size[0]))
        y_index1 = y_index0 + self.out_size[0]
        image = image[y_index0:y_index1, x_index0:x_index1]
        return image


@dataclass(frozen=True)
class GrayRandomRotation():
    """Gray random rotation.
    """
    angle_range: Tuple[float, float]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call function.
        """
        image = image.copy()
        # Generate random angle
        angle = np.random.rand()*(self.angle_range[1] - self.angle_range[0])
        image = rotate(image, -angle, reshape=False, mode="constant")
        return image


@dataclass(frozen=True)
class GrayNormalization():
    """Gray normalization.
    """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call function.
        """
        image = image.copy()
        image = image/255.0
        return image


@dataclass(frozen=True)
class GrayToTensor():
    """Gray to tensor.
    """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call function.
        """
        image = image.copy()
        image = torch.from_numpy(image)
        return image
