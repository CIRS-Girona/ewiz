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

    def __getitem__(self, indices: Union[int, slice]) -> Tuple[np.ndarray]:
        """Returns data based on slice.
        """
        if isinstance(indices, slice):
            start, stop, _ = indices.indices(self.events_size)
            return self._get_events_data(start, stop)
        elif isinstance(indices, int):
            return self._get_events_data(indices)

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
        self.events_time_offset = self.events_file["time_offset"][0]
        self.time_to_events = self.events_file["time_to_events"]
        self.events_size = self.events_x.shape[0]

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
            self.gray_time_offset = self.gray_file["time_offset"][0]
            self.time_to_gray = self.gray_file["time_to_gray"]
            self.gray_to_events = self.gray_file["gray_to_events"]

            # Grayscale flag
            self.gray_flag = True

    def _get_events_data(self, start_index: int, end_index: int = None) -> np.ndarray:
        """Combines events data.
        """
        if end_index is not None:
            shape = (end_index - start_index, 4)
            events = np.zeros(shape, dtype=np.float64)
            events[:, 0] = self.events_x[start_index:end_index]
            events[:, 1] = self.events_y[start_index:end_index]
            events[:, 2] = self.events_time[start_index:end_index] + self.events_time_offset
            events[:, 3] = self.events_pol[start_index:end_index]
            return events
        else:
            shape = (1, 4)
            events = np.zeros(shape, dtype=np.float64)
            events[:, 0] = self.events_x[start_index]
            events[:, 1] = self.events_y[start_index]
            events[:, 2] = self.events_time[start_index] + self.events_time_offset
            events[:, 3] = self.events_pol[start_index]
            return events

    # TODO: Check return type
    def _get_gray_images(self, start_index: int, end_index: int = None) -> np.ndarray:
        """Gets grayscale images.
        """
        if self.gray_flag:
            if end_index is None:
                gray_image = self.gray_images[start_index]
                time = self.gray_time[start_index] + self.gray_time_offset
                return gray_image, time
            else:
                gray_image = self.gray_images[start_index:end_index]
                time = self.gray_time[start_index:end_index] + self.gray_time_offset
                return gray_image, time
        else:
            return None, None
