import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from .base import WriterBase

from typing import Any, Dict, List, Tuple, Callable, Union


class WriterGray(WriterBase):
    """Data writer for grayscale images.
    """
    def __init__(self, out_dir: str) -> None:
        super().__init__(out_dir)
        self._init_gray()

    def _init_gray(self) -> None:
        """Initializes grayscale images file.
        """
        self.gray_path = os.path.join(self.out_dir, "gray.hdf5")
        self.gray_file = h5py.File(self.gray_path, "a")
        self.gray_flag = False

    def write(self, gray_image: np.ndarray, time: int) -> None:
        """Main data writing function.
        """
        gray_image = gray_image[None, ...].astype(np.uint8)
        time: np.ndarray = np.array(time, dtype=np.int64)[None, ...]
        image_size = (gray_image.shape[1], gray_image.shape[2])

        # TODO: Check time format
        if self.gray_flag is False:
            self.time_offset = 0
            if time != 0:
                self._save_time_offset(data_file=self.gray_file, time=time)

            # Create HDF5 groups
            self.gray_images = self.gray_file.create_dataset(
                name="gray_images", data=gray_image,
                chunks=True, maxshape=(None, *image_size), dtype=np.uint8,
                **self.compressor
            )
            self.gray_time = self.gray_file.create_dataset(
                name="time", data=time - self.time_offset,
                chunks=True, maxshape=(None,), dtype=np.int64,
                **self.compressor
            )
            self.gray_flag = True
        else:
            data_points = gray_image.shape[0]
            dataset_points = self.gray_images.shape[0]
            all_points = data_points + dataset_points
            self.gray_images.resize(all_points, axis=0)
            self.gray_images[-data_points:] = gray_image
            self.gray_time.resize(all_points, axis=0)
            self.gray_time[-data_points:] = time - self.time_offset
