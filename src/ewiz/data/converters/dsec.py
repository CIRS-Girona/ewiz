import os
import h5py
import hdf5plugin
import natsort
import cv2
import numpy as np

from tqdm import tqdm

from .base import ConvertBase
from ..writers import WriterEvents, WriterGray, WriterFlow

from typing import Any, Dict, List, Tuple, Callable, Union


class ConvertDSEC(ConvertBase):
    """DSEC to eWiz data converter.
    """
    def __init__(
        self,
        data_dir: str,
        out_dir: str,
        cam_location: str = "left"
    ) -> None:
        super().__init__(data_dir, out_dir)
        self.cam_location = cam_location
        self.sensor_size = (480, 640)
        self._init_events()
        self._init_images()
        self._get_events_time_offset()
        self._init_writers()

    def _init_events(self) -> None:
        """Initializes events file path.
        """
        self.events_path = os.path.join(self.data_dir, "events", self.cam_location, "events.h5")
        self.events_file = h5py.File(self.events_path, "r")
        self.events_x = self.events_file["events/x"]
        self.events_y = self.events_file["events/y"]
        self.events_time = self.events_file["events/t"]
        self.events_pol = self.events_file["events/p"]

    def _init_images(self) -> None:
        """Initializes images file path.
        """
        self.rgb_dir = os.path.join(self.data_dir, "images", self.cam_location, "rectified")
        self.rgb_size = len(os.listdir(self.rgb_dir))
        self.rgb_sorted = natsort.natsorted(os.listdir(self.rgb_dir))
        self.rgb_time_path = os.path.join(self.data_dir, "images", "timestamps.txt")
        self.rgb_time = np.loadtxt(self.rgb_time_path, dtype=np.int64)

    def _init_writers(self) -> None:
        """Initializes writers.
        """
        self.events_writer = WriterEvents(self.out_dir)
        self.gray_writer = WriterGray(self.out_dir)

    def _get_events_time_offset(self) -> None:
        """Gets events time offset.
        """
        if "t_offset" in self.events_file.keys():
            self.events_time_offset = int(self.events_file["t_offset"][()])
        else:
            self.events_time_offset = 0

    def _init_events_stride(self, events_stride: int = 1e4) -> None:
        """Initializes events stride.
        """
        self.events_stride = events_stride
        self.events_indices = np.arange(0, self.events_x.shape[0], self.events_stride)
        if self.events_indices[-1] != self.events_x.shape[0] - 1:
            self.events_indices = np.append(self.events_indices, self.events_x.shape[0] - 1)
        self.events_size = len(self.events_indices) - 1

    def convert(self, events_stride: int = 1e4) -> None:
        """Converts DSEC data.
        """
        print("# === Converting DSEC Data === #")
        print("# === Converting Events === #")
        self._init_events_stride(events_stride)
        progress_bar = tqdm(range(self.events_size))
        for i in progress_bar:
            start = int(self.events_indices[i])
            end = int(self.events_indices[i + 1])
            chunk_size = (end - start, 4)
            events = np.zeros(chunk_size, dtype=np.float64)
            events[:, 0] = self.events_x[start:end]
            events[:, 1] = self.events_y[start:end]
            events[:, 2] = self.events_time[start:end] + self.events_time_offset
            events[:, 3] = self.events_pol[start:end]
            events = events.astype(np.int64)
            self.events_writer.write(events=events)
        # Map time to events
        # TODO: Add option to choose chunk number
        self.events_writer.map_time_to_events()

        print("# === Converting RGB Images === #")
        progress_bar = tqdm(range(self.rgb_size))
        for i in progress_bar:
            image_path = os.path.join(self.rgb_dir, self.rgb_sorted[i])
            image = cv2.imread(image_path, 0)
            image = cv2.resize(image, self.sensor_size)
            time = self.rgb_time[i]
            self.gray_writer.write(gray_image=image, time=time)
        # Map time to grayscale images
        self.gray_writer.map_time_to_gray()
        self.gray_writer.map_gray_to_events()
