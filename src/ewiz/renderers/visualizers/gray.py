import os
import cv2
import numpy as np

from PIL import Image

from .base import VisualizerBase

from typing import Any, Dict, List, Tuple, Callable, Union


class VisualizerGray(VisualizerBase):
    """Grayscale visualizer.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        save_images: bool = False,
        override_images: bool = False,
        image_prefix: str = None,
        out_dir: str = None
    ) -> None:
        super().__init__(
            image_size, save_images, override_images, image_prefix, out_dir
        )
        self.render_func = self._generate_gray_image

    def _generate_gray_image(self, gray_image: np.ndarray) -> np.ndarray:
        """Generates grayscale image.
        """
        gray_image = gray_image.astype(np.uint8)
        return gray_image[..., None]
