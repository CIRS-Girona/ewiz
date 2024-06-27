import os
import h5py
import hdf5plugin
import numpy as np

from typing import Any, Dict, List, Tuple, Callable, Union


class ReaderBase():
    """Base data reader.
    """
    def __init__(
        self,
        data_dir: str
    ) -> None:
        self.data_dir = data_dir
        self._init_events()
        self._init_gray()

    def _init_events(self) -> None:
        """Initializes events file.
        """
        self.events_path = os.path.join(self.data_dir, "events.hdf5")
        self.events_file = h5py.File(self.events_path, "a")
        self.events_group = self.events_file["events"]

        # Events data
        self.events_x = self.events_group["x"]
        self.events_y = self.events_group["y"]
        self.events_time = self.events_group["time"]
        self.events_pol = self.events_group["polarity"]

        # Events data properties
        self.events_time_offset = self.events_file["time_offset"]
        self.time_to_events = self.events_file["time_to_events"]

    def _init_gray(self) -> None:
        """Initializes grayscale images.
        """
        self.gray_flag = False
        self.gray_path = os.path.join(self.data_dir, "gray.hdf5")

        if os.path.exists(self.gray_path):
            # Grayscale images data
            self.gray_file = h5py.File(self.gray_path, "r")
            self.gray_images = self.gray_file["gray_images"]
            self.gray_time = self.gray_file["time"]

            # Grayscale images properties
            self.gray_time_offset = self.gray_file["time_offset"]
            self.time_to_gray = self.gray_file["time_to_gray"]
            self.gray_to_events = self.gray_file["gray_to_events"]

            # Grayscale flag
            self.gray_flag = True
