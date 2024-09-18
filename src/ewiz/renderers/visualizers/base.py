import os
import cv2
import numpy as np

from PIL import Image

from typing import Any, Dict, List, Tuple, Callable, Union


class VisualizerBase():
    """Base visualizer.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        save_images: bool = False,
        override_images: bool = False,
        image_prefix: str = None,
        out_dir: str = None
    ) -> None:
        self.image_size = image_size
        self.save_images = save_images
        self.override_images = override_images
        self.image_prefix = image_prefix
        self.image_count = 0
        self.out_dir = out_dir
        if self.out_dir:
            self.create_dir(self.out_dir)

        # TODO: Add render function
        self.render_func = None
        self.mask_func = None

    # TODO: Add folder creation to utilities
    @staticmethod
    def create_dir(dir_path: str) -> None:
        """Creates directory.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("Created directory:", dir_path)

    def _get_save_path(self) -> str:
        """Gets image save path.
        """
        if self.save_images:
            if self.override_images:
                save_path = os.path.join(self.out_dir, f"{self.image_prefix}.png")
            else:
                save_path = os.path.join(self.out_dir, f"{self.image_prefix}{self.image_count}.png")
                self.image_count += 1
            return save_path

    def _get_image(self, image: np.ndarray) -> np.ndarray:
        """Gets or saves image.
        """
        if self.save_images:
            image_pil: Image.Image = Image.fromarray(image)
            if image_pil.mode == "RGBA":
                image_pil = image_pil.convert("RGB")
            save_path = self._get_save_path()
            image_pil.save(save_path)
        return image

    def render_image(self, *args, **kwargs) -> np.ndarray:
        """Renders image.
        """
        image = self.render_func(*args, **kwargs)
        return image

    # TODO: Check conflict with imager functions
    def render_mask(self, *args, **kwargs) -> np.ndarray:
        """Renders mask.
        """
        image = self.mask_func(*args, **kwargs)
        return image
