import os
import cv2
import numpy as np

from PIL import Image

from .base import VisualizerBase

from typing import Any, Dict, List, Tuple, Callable, Union


class VisualizerFlow(VisualizerBase):
    """Flow visualizer.
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
        self.render_func = self._generate_flow_image

    # TODO: Check mask format
    def _generate_flow_image(self, flow: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Generates flow image.
        """
        magnitudes: np.ndarray = np.linalg.norm(flow, axis=0)
        angles: np.ndarray = np.arctan2(flow[1, ...], flow[0, ...])
        angles = angles + np.pi
        angles = angles*(180/np.pi/2)
        angles = angles.astype(np.uint8)

        # Generate colors
        colors = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
        colors[..., 0] = np.mod(angles, 180)
        colors[..., 1] = 255
        colors[..., 2] = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX)

        # Generate image
        flow_image = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)
        if mask:
            flow_image = flow_image*mask
        return flow_image
