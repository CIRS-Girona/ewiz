import os
import h5py
import hdf5plugin
import numpy as np

from ewiz.core.utils import read_json

from typing import Any, Dict, List, Tuple, Callable, Union


# TODO: Revise indexing of grayscale images
class ReaderBase:
    """Base data reader. The data reader class reads a dataset in the eWiz format. The
    created data reader object can then be sliced, like any array, and returns the corresponding data.
    """

    def __init__(self, data_dir: str, clip_mode: str = "events") -> None:
        """
        Args:
            data_dir (str): Dataset directory, should be using the eWiz format.
            clip_mode (str, optional): Clip mode, defines how the data is sliced.
                You can slice using event indices, timestamps, and grayscale images. The
                data reader automatically synchronizes the corresponding data for any clip mode.
                Possible options are: "events", "time", and "images" respectively. Defaults to "events".

        Returns:
            (np.ndarray): Array of raw events of size [N, 4], where N is the number of events.
                Columns in order are x, y, timestamp, and polarity.
            (np.ndarray): A set of grayscale images that were captured between the first and last
                event in the sequence. The array is of size [N, H, W] where N is the number of
                grayscale images, H is the sensor height, and W is the sensor width.
            (np.ndarray): Grayscale timestamps, which correspond to each grayscale image.
                It is of shape [N].

        Examples:
            Let's say we want to get the raw events that took place between the 7th and the
            10th grayscale image in the sequence, we run the data reader as follows:

            >>> reader = ReaderBase(data_dir="/path/to/dataset", clip_mode="images")
            >>> events, grayscale, grayscale_timestamp = reader[7:11]
        """
        self.data_dir = data_dir
        self.clip_mode = clip_mode
        self._init_props()
        self._init_events()
        self._init_gray()
        self._init_clip()
        self._init_size()

    # TODO: Modify based on clipping mode
    def __getitem__(self, indices: Union[int, slice]) -> Tuple[np.ndarray]:
        if isinstance(indices, slice):
            start, stop = self._get_slice_indices(indices)
            data = self.clip_func(start, stop)
            return data
        elif isinstance(indices, int):
            index = self._get_int_index(indices)
            data = self.clip_func(index)
            return data
        else:
            raise TypeError("Invalid slice.")

    def __len__(self) -> int:
        return self.data_size

    def _get_slice_indices(self, indices: slice) -> Tuple[int, int]:
        """Returns slice indices depending on clip mode."""
        size = getattr(self, self.clip_mode + "_size")
        start, stop, _ = indices.indices(size)
        assert (start >= 0 and start < size) and (stop >= 0 and stop < size), (
            f"Slice indices out of range for data size of '{size}'. "
            "Check your slice indices."
        )
        return start, stop

    def _get_int_index(self, index: int) -> int:
        """Returns integer index depending on clip mode."""
        size = getattr(self, self.clip_mode + "_size")
        assert index >= 0 and index < size, (
            f"Integer index out of range for data size of '{size}'. "
            "Check your integer index."
        )
        return index

    # TODO: Modify format
    def _init_props(self) -> None:
        """Initializes properties."""
        props_path = os.path.join(self.data_dir, "props.json")
        self.props = read_json(props_path)

        # Dataset properties
        self.sensor_size = tuple(self.props["sensor_size"])

    def _init_events(self) -> None:
        """Initializes events file."""
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
        self.time_size = self.time_to_events.shape[0]
        self.images_size = None

    def _init_gray(self) -> None:
        """Initializes grayscale images."""
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
            self.images_size = self.gray_to_events.shape[0]

            # Grayscale flag
            self.gray_flag = True

    def _init_size(self) -> None:
        """Initializes dataset size."""
        if self.clip_mode == "events":
            self.data_size = self.events_x.shape[0]
        elif self.clip_mode == "images":
            self.data_size = self.gray_images.shape[0]
        elif self.clip_mode == "time":
            self.data_size = self.time_to_events.shape[0]
        else:
            raise KeyError(f"Clip mode key '{self.clip_mode}' is not supported.")

    def _get_events_data(self, start_index: int, end_index: int = None) -> np.ndarray:
        """Combines events data."""
        if end_index is not None:
            shape = (end_index - start_index, 4)
            events = np.zeros(shape, dtype=np.float64)
            events[:, 0] = self.events_x[start_index:end_index]
            events[:, 1] = self.events_y[start_index:end_index]
            events[:, 2] = (
                self.events_time[start_index:end_index] + self.events_time_offset
            )
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
        """Gets grayscale images."""
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
        """Initializes clipping function."""
        self.clip_func = getattr(self, "_clip_with_" + self.clip_mode)

    def _clip_with_events(
        self, start_index: int, end_index: int = None
    ) -> Tuple[np.ndarray]:
        """Clips data with events indices."""
        if end_index is not None:
            events = self._get_events_data(start_index, end_index)
            start_time = int((events[0, 2] - self.events_time_offset) / 1e3)
            end_time = int((events[-1, 2] - self.events_time_offset) / 1e3)
            # TODO: Organize indexing
            start_gray = (
                int(self.time_to_gray[start_time] - 1)
                if self.time_to_gray[start_time] > 0
                else int(self.time_to_gray[start_time])
            )
            end_gray = int(self.time_to_gray[end_time] + 1)
            gray_images, gray_time = self._get_gray_images(start_gray, end_gray)
            return events, gray_images, gray_time
        else:
            events = self._get_events_data(start_index)
            start_time = int((events[0, 2] - self.events_time_offset) / 1e3)
            # TODO: Organize indexing
            start_gray = (
                int(self.time_to_gray[start_time] - 1)
                if self.time_to_gray[start_time] > 0
                else int(self.time_to_gray[start_time])
            )
            gray_images, gray_time = self._get_gray_images(start_gray)
            return events, gray_images, gray_time

    def _clip_with_time(
        self, start_time: int, end_time: int = None
    ) -> Tuple[np.ndarray]:
        """Clips data with timestamps."""
        if end_time is not None:
            start_index = int(self.time_to_events[start_time])
            end_index = int(self.time_to_events[end_time])
            events = self._get_events_data(start_index, end_index)
            # TODO: Organize indexing
            gray_images = None
            gray_time = None
            # TODO: Add condition for other indexing methods
            if self.gray_flag:
                start_index = (
                    int(self.time_to_gray[start_time] - 1)
                    if self.time_to_gray[start_time] > 0
                    else int(self.time_to_gray[start_time])
                )
                end_index = int(self.time_to_gray[end_time] + 1)
                gray_images, gray_time = self._get_gray_images(start_index, end_index)
            return events, gray_images, gray_time
        else:
            start_index = int(self.time_to_events[start_time])
            events = self._get_events_data(start_index)
            # TODO: Organize indexing
            start_index = (
                int(self.time_to_gray[start_time] - 1)
                if self.time_to_gray[start_time] > 0
                else int(self.time_to_gray[start_time])
            )
            gray_images, gray_time = self._get_gray_images(start_index)
            return events, gray_images, gray_time

    def _clip_with_images(
        self, start_index: int, end_index: int = None
    ) -> Tuple[np.ndarray]:
        """Clips data with images."""
        if end_index is not None:
            start_events = int(self.gray_to_events[start_index])
            end_events = int(self.gray_to_events[end_index])
            events = self._get_events_data(start_events, end_events)
            gray_images, gray_time = self._get_gray_images(start_index, end_index + 1)
            return events, gray_images, gray_time
        else:
            start_events = int(self.gray_to_events[start_index])
            events = self._get_events_data(start_events)
            gray_images, gray_time = self._get_gray_images(start_index)
            return events, gray_images, gray_time
