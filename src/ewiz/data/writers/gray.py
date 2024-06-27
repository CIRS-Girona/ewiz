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
        self._init_events()
        self._init_gray()

    def _init_events(self) -> None:
        """Initializes events HDF5 file.
        """
        self.events_path = os.path.join(self.out_dir, "events.hdf5")
        self.events_file = h5py.File(self.events_path, "a")
        self.events_flag = False

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
        image_size = (gray_image.shape[1], gray_image.shape[2])

        # TODO: Check time format
        if self.gray_flag is False:
            self.time_offset = 0
            if time != 0:
                self.time_offset = time
            self._save_time_offset(data_file=self.gray_file, time=time)

            # Create HDF5 groups
            self.gray_images = self.gray_file.create_dataset(
                name="gray_images", data=gray_image,
                chunks=True, maxshape=(None, *image_size), dtype=np.uint8,
                **self.compressor
            )
            time: np.ndarray = np.array(time, dtype=np.int64)[None, ...]
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

    def map_time_to_gray(self) -> None:
        """Maps timestamps to grayscale indices.
        """
        events_group = self.events_file["events"]
        events_time = events_group["time"]
        events_time_offset = self.events_file["time_offset"]
        print("# === Mapping Timestamps to Grayscale Indices === #")
        start_value = np.floor(events_time[0]/1e3)
        end_value = np.ceil(events_time[-1]/1e3)
        sorted_data = (self.gray_time[:] + self.time_offset)/1e3
        data_file = self.gray_file
        data_name = "time_to_gray"
        offset_value = events_time_offset[0]

        # TODO: Review arguments
        self.map_data_in_memory(
            start_value, end_value, sorted_data,
            data_file, data_name, offset_value
        )

    # TODO: Check value difference
    def map_gray_to_events(self) -> None:
        """Maps grayscale indices to events indices.
        """
        events_group = self.events_file["events"]
        events_time = events_group["time"]
        events_time_offset = self.events_file["time_offset"]
        print("# === Mapping Grayscale Indices to Events Indices === #")
        start_value = None
        end_value = None
        sorted_data = events_time
        data_file = self.gray_file
        data_name = "gray_to_events"
        offset_value = self.time_offset
        side = "left"
        chunks = 32
        addition = events_time_offset[0]
        division = 1.0
        array_value = self.gray_time[:]

        # TODO: Review arguments
        self.map_data_out_memory(
            start_value, end_value, sorted_data,
            data_file, data_name, offset_value, side, chunks,
            addition, division, array_value
        )
