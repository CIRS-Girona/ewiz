import os
import cv2
import imageio
import numpy as np

from PIL import Image

from ewiz.core.utils import create_dir

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
        window_size: Tuple[int, int] = None,
        save_images: bool = False,
        save_dir: str = None
    ) -> None:
        self.image_size = image_size
        self.grid_size = grid_size
        self.window_names = window_names
        self.num_windows = len(window_names)
        self.refresh_rate = refresh_rate
        self.window_size = window_size
        self.save_images = save_images
        self.save_dir = save_dir
        self._init_image_saver()

    def _init_image_saver(self) -> None:
        """Initializes image saver.
        """
        if self.save_images:
            self.save_dirs = [
                create_dir(os.path.join(self.save_dir, window_name))
                for window_name in self.window_names
            ]
            self.image_indices = [0 for i in range(self.num_windows)]

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

            # Save image
            if self.save_images:
                image_path = os.path.join(
                    self.save_dirs[i],
                    "image" + str(self.image_indices[i]) + ".png"
                )
                cv2.imwrite(image_path, image)
                self.image_indices[i] += 1

            # Update indices
            w += 1
            if w == self.grid_size[1]:
                w = 0
                h += 1
            cv2.waitKey(self.refresh_rate)

    def create_gif(self) -> None:
        """Creates GIF of saved images.
        """
        if self.save_images:
            print("Creating GIF.")
            for i, window_name in enumerate(self.window_names):
                all_images = []
                # TODO: Change video limit
                for j in range(344):
                    image_path = os.path.join(self.save_dirs[i], "image" + str(j) + ".png")
                    all_images.append(imageio.imread(image_path))
                imageio.mimsave(os.path.join(self.save_dirs[i], "video.gif"), all_images)
                print("Done GIF.")

    # TODO: Data stride needs to be taken in consideration
    def create_mp4(self) -> None:
        """Creates GIF of saved images.
        """
        if self.save_images:
            print("Creating MP4.")
            for i, window_name in enumerate(self.window_names):
                video_writer = imageio.get_writer(os.path.join(self.save_dirs[i], "video.mp4"), fps=12)
                # TODO: Change video limit
                for j in range(self.image_indices[0]):
                    image_path = os.path.join(self.save_dirs[i], "image" + str(j) + ".png")
                    video_writer.append_data(imageio.imread(image_path))
                video_writer.close()
                print("Done MP4.")
