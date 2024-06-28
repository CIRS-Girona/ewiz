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
        data_dir: str,
        clip_mode: str = "events"
    ) -> None:
        self.data_dir = data_dir
        self.clip_mode = clip_mode
        self._init_events()
        self._init_gray()
        self._init_clip()

    # TODO: Modify based on clipping mode
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

    def _init_clip(self) -> None:
        """Initializes clipping function.
        """
        self.clip_func = getattr(self, "_clip_with_" + self.clip_mode)

    def _clip_with_events(self, start_index: int, end_index: int = None) -> Tuple[np.ndarray]:
        """Clips data with events indices.
        """
        if end_index is not None:
            events = self._get_events_data(start_index, end_index)
            start_time = int((events[0, 2] - self.events_time_offset)/1e3)
            end_time = int((events[-1, 2] - self.events_time_offset)/1e3)
            start_gray = int(self.time_to_gray[start_time])
            end_gray = int(self.time_to_gray[end_time])
            gray_images, gray_time = self._get_gray_images(start_gray, end_gray)
            return events, gray_images, gray_time
        else:
            events = self._get_events_data(start_index)
            start_time = int((events[0, 2] - self.events_time_offset)/1e3)
            start_gray = int(self.time_to_gray[start_time])
            gray_images, gray_time = self._get_gray_images(start_gray)
            return events, gray_images, gray_time

    def _clip_with_time(self, start_time: int, end_time: int = None) -> Tuple[np.ndarray]:
        """Clips data with timestamps.
        """
        if end_time is not None:
            start_index = int(self.time_to_events[start_time])
            end_index = int(self.time_to_events[end_time])
            events = self._get_events_data(start_index, end_index)
            start_index = int(self.time_to_gray[start_time])
            end_index = int(self.time_to_gray[end_time])
            gray_images, gray_time = self._get_gray_images(start_index, end_index)
            return events, gray_images, gray_time
        else:
            start_index = int(self.time_to_events[start_time])
            events = self._get_events_data(start_index)
            start_index = int(self.time_to_gray[start_time])
            gray_images, gray_time = self._get_gray_images(start_index)
            return events, gray_images, gray_time

    def _clip_with_images(self, start_index: int, end_index: int = None) -> Tuple[np.ndarray]:
        """Clips data with images.
        """
        if start_index is not None:
            start_events = int(self.gray_to_events[start_index])
            end_events = int(self.gray_to_events[end_index])
            events = self._get_events_data(start_events, end_events)
            gray_images, gray_time = self._get_gray_images(start_index, end_index)
            return events, gray_images, gray_time
        else:
            start_events = int(self.gray_to_events[start_index])
            events = self._get_events_data(start_events)
            gray_images, gray_time = self._get_gray_images(start_index)
            return events, gray_images, gray_time
