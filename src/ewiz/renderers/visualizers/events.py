import os
import cv2
import numpy as np

from PIL import Image

from .base import VisualizerBase

from typing import Any, Dict, List, Tuple, Callable, Union


class VisualizerEvents(VisualizerBase):
    """Events visualizer.
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
        self.render_func = self._generate_events_image
        self.mask_func = self._generate_events_mask

    def _generate_events_image(self, events: np.ndarray) -> np.ndarray:
        """Generates events image.
        """
        events_image = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        events_image *= 255
        w_coords = events[:, 0].astype(np.int32)
        h_coords = events[:, 1].astype(np.int32)

        # Generate colors
        colors = np.zeros((events.shape[0], 3), dtype=np.uint8)
        polarities = events[:, 3]
        red = np.array([255, 0, 0])
        blue = np.array([0, 0, 255])
        colors[polarities >= 1] = red
        colors[polarities < 1] = blue

        # Generate image
        events_image[h_coords, w_coords, :] = colors
        events_image = events_image.astype(np.uint8)
        return events_image

    def _generate_events_mask(self, events: np.ndarray) -> np.ndarray:
        """Generates events mask.
        """
        events_mask = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        w_coords = events[:, 0].astype(np.int32)
        h_coords = events[:, 1].astype(np.int32)

        # Generate mask
        events_mask[h_coords, w_coords] = 1
        return events_mask
