import os
import cv2
import numpy as np

from PIL import Image

from .base import VisualizerBase

from typing import Any, Dict, List, Tuple, Callable, Union


class VisualizerFlow(VisualizerBase):
    """Flow visualizer."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        save_images: bool = False,
        override_images: bool = False,
        image_prefix: str = None,
        out_dir: str = None,
        vis_type: str = "colors",
    ) -> None:
        super().__init__(
            image_size, save_images, override_images, image_prefix, out_dir
        )
        self.vis_type = vis_type
        if self.vis_type == "colors":
            self.render_func = self._generate_flow_image
        else:
            self.render_func = self._generate_flow_image_with_arrows

    # TODO: Check mask format
    def _generate_flow_image(
        self, flow: np.ndarray, mask: np.ndarray = None
    ) -> np.ndarray:
        """Generates flow image."""
        magnitudes: np.ndarray = np.linalg.norm(flow, axis=0)
        angles: np.ndarray = np.arctan2(flow[1, ...], flow[0, ...])
        angles = angles + np.pi
        angles = angles * (180 / np.pi / 2)
        angles = angles.astype(np.uint8)

        # Generate colors
        colors = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
        colors[..., 0] = np.mod(angles, 180)
        colors[..., 1] = 255
        colors[..., 2] = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX)

        # Generate image
        flow_image = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)
        if mask is not None:
            flow_image = flow_image * mask
        return flow_image

    def _generate_flow_image_with_arrows(
        self,
        flow: np.ndarray,
        scale: float = 100,
        sample_every: int = 10,
        mask: np.ndarray = None,
    ) -> np.ndarray:
        """_summary_"""
        # TODO: Will require modification, maybe add the events image
        image = self._generate_flow_image(flow, mask)
        _, h, w = flow.shape
        for y in range(0, h, sample_every):
            for x in range(0, w, sample_every):
                if mask is not None:
                    if mask[y, x] > 0:
                        dx, dy = flow[:, y, x]
                        start_point = (x, y)
                        end_point = (int(x + scale * dx), int(y + scale * dy))
                        cv2.arrowedLine(
                            image,
                            start_point,
                            end_point,
                            (0, 255, 0),
                            thickness=1,
                            tipLength=0.3,
                        )
                else:
                    dx, dy = flow[:, y, x]
                    start_point = (x, y)
                    end_point = (int(x + scale * dx), int(y + scale * dy))
                    cv2.arrowedLine(
                        image,
                        start_point,
                        end_point,
                        (0, 255, 0),
                        thickness=1,
                        tipLength=0.3,
                    )
        return image
