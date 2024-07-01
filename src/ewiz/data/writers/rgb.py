import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from .base import WriterBase

from typing import Any, Dict, List, Tuple, Callable, Union


class WriterRGB(WriterBase):
    """Data writer for RGB data.
    """
    def __init__(self, out_dir: str) -> None:
        super().__init__(out_dir)
        self._init_events()
        self._init_rgb()

    def _init_events(self) -> None:
        """Initializes events HDF5 file.
        """
        self.events_path = os.path.join(self.out_dir, "events.hdf5")
        self.events_file = h5py.File(self.events_path, "a")
        self.events_flag = False

    def _init_rgb(self) -> None:
        """Initializes RGB file.
        """
        self.rgb_path = os.path.join(self.out_dir, "rgb.hdf5")
        self.rgb_file = h5py.File(self.rgb_path, "a")
        self.rgb_flag = False

    def write(self, rgb_image: np.ndarray, time: int) -> None:
        """Main data writing function.
        """
        rgb_image = rgb_image[None, ...].astype(np.uint8)
        image_size = (rgb_image.shape[2], rgb_image.shape[3])

        # TODO: Check time format
        if self.rgb_flag is False:
            self.time_offset = 0
            if time != 0:
                self.time_offset = time
            self._save_time_offset(data_file=self.rgb_file, time=time)

            # Create HDF5 groups
            self.rgb_images = self.rgb_file.create_dataset(
                name="rgb_images", data=rgb_image,
                chunks=True, maxshape=(None, 3, *image_size), dtype=np.uint8,
                **self.compressor
            )
            time: np.ndarray = np.array(time, dtype=np.int64)[None, ...]
            self.rgb_time = self.rgb_file.create_dataset(
                name="time", data=time - self.time_offset,
                chunks=True, maxshape=(None,), dtype=np.int64,
                **self.compressor
            )
            self.rgb_flag = True
        else:
            data_points = rgb_image.shape[0]
            dataset_points = self.rgb_images.shape[0]
            all_points = data_points + dataset_points
            self.rgb_images.resize(all_points, axis=0)
            self.rgb_images[-data_points:] = rgb_image
            self.rgb_time.resize(all_points, axis=0)
            self.rgb_time[-data_points:] = time - self.time_offset
