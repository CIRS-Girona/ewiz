import os
import cv2
import numpy as np

from PIL import Image

from typing import Any, Dict, List, Tuple, Callable, Union


class WindowManager():
    """Window manager for OpenCV.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
        window_names: List[str],
        refresh_rate: int = 2,
        window_size: Tuple[int, int] = None
    ) -> None:
        self.image_size = image_size
        self.grid_size = grid_size
        self.window_names = window_names
        self.num_windows = len(window_names)
        self.refresh_rate = refresh_rate
        self.window_size = window_size

    def _display_text(
        self,
        text: str,
        image: np.ndarray,
        position: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """Displays text.
        """
        image = cv2.putText(image, text, position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image

    @staticmethod
    def numpy_to_cv(image: np.ndarray) -> np.ndarray:
        """Flips RGB values in image.
        """
        image = image[:, :, ::-1].copy()
        return image

    def render(
        self,
        *args,
        texts: List[str] = None,
        position: Tuple[int, int] = (0, 0)
    ) -> None:
        """Main rendering function.
        If no text is in the image, use None.
        """
        # Create texts
        if texts is None:
            texts = [None for _ in range(self.num_windows)]

        # Create windows
        h = 0
        w = 0
        for i in range(self.num_windows):
            cv2.namedWindow(self.window_names[i], 0)
            h_coord = int(h*self.image_size[0]*1.8 + 100)
            w_coord = int(w*self.image_size[1]*1.8 + 100)
            cv2.moveWindow(self.window_names[i], w_coord, h_coord)
            # TODO: Image creation
            image = self.numpy_to_cv(args[i])
            if texts[i] is not None:
                image = self._display_text(texts[i], image, position)
            if self.window_size is not None:
                cv2.resizeWindow(self.window_names[i], self.window_size[1], self.window_size[0])
                image = cv2.resize(image, (self.window_size[1], self.window_size[0]))
            cv2.imshow(self.window_names[i], image)
            # Update indices
            w += 1
            if w == self.grid_size[1]:
                w = 0
                h += 1
            cv2.waitKey(self.refresh_rate)
