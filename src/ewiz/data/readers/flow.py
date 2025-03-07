import os
import h5py
import hdf5plugin
import numpy as np

from .base import ReaderBase

# TODO: Check working code
from .utils._sync import FlowSync

from typing import Any, Dict, List, Tuple, Callable, Union


class ReaderFlow(ReaderBase):
    """Flow data reader. The data reader class reads a dataset in the eWiz format. The
    created data reader object can then be sliced, like any array, and returns the corresponding data.
    The flow data reader also includes optical flow data.
    """

    def __init__(
        self, data_dir: str, clip_mode: str = "events", inverse_flow: bool = False
    ) -> None:
        """
        Args:
            data_dir (str): Dataset directory, should be using the eWiz format.
            clip_mode (str, optional): Clip mode, defines how the data is sliced.
                You can slice using event indices, timestamps, and grayscale images. The
                data reader automatically synchronizes the corresponding data for any clip mode.
                Possible options are: "events", "time", and "images" respectively. Defaults to "events".
            inverse_flow (bool, optional): Inverses the optical flow. The default direction depends on the
                dataset itself, it is always returned as a displacement.

        Returns:
            (np.ndarray): Array of raw events of size [N, 4], where N is the number of events.
                Columns in order are x, y, timestamp, and polarity.
            (np.ndarray): A set of grayscale images that were captured between the first and last
                event in the sequence. The array is of size [N, H, W] where N is the number of
                grayscale images, H is the sensor height, and W is the sensor width.
            (np.ndarray): Grayscale timestamps, which correspond to each grayscale image.
                It is of shape [N].
            (np.ndarray): Sliced and synchronized optical flow. The optical is an array of
                size [2, H, W], where 2 corresponds to the X and Y flow displacements respectively,
                H is the sensor height, and W is the sensor width.

        Examples:
            Let's say we want to get the raw events that took place between the 7th and the
            10th grayscale image in the sequence, we run the data reader as follows:

            >>> reader = ReaderFlow(data_dir="/path/to/dataset", clip_mode="images")
            >>> events, grayscale, grayscale_timestamp, flow = reader[7:11]
        """
        super().__init__(data_dir, clip_mode)
        self.inverse_flow = inverse_flow
        self._init_flows()

    def _init_flows(self) -> None:
        """Initializes flows file."""
        self.flow_path = os.path.join(self.data_dir, "flow.hdf5")
        self.flow_file = h5py.File(self.flow_path, "r")
        # Flows data
        self.flows = self.flow_file["flows"]
        self.flows_time = self.flow_file["time"]
        # Initialize synchronizer
        self.flow_syncer = FlowSync(self.flow_file)

    def _clip_with_events(
        self, start_index: int, end_index: int = None
    ) -> Tuple[np.ndarray]:
        """Clips data with events indices."""
        events, gray_images, gray_time = super()._clip_with_events(
            start_index, end_index
        )
        start_time = int((events[0, 2] - self.events_time_offset) / 1e3)
        end_time = int((events[-1, 2] - self.events_time_offset) / 1e3)
        sync_flow = self.flow_syncer.sync(start_time, end_time, self.inverse_flow)
        return events, gray_images, gray_time, sync_flow

    def _clip_with_time(
        self, start_time: int, end_time: int = None
    ) -> Tuple[np.ndarray]:
        """Clips data with timestamps."""
        events, gray_images, gray_time = super()._clip_with_time(start_time, end_time)
        sync_flow = self.flow_syncer.sync(start_time, end_time, self.inverse_flow)
        return events, gray_images, gray_time, sync_flow

    def _clip_with_images(
        self, start_index: int, end_index: int = None
    ) -> Tuple[np.ndarray]:
        """Clips data with images."""
        events, gray_images, gray_time = super()._clip_with_images(
            start_index, end_index
        )
        start_time = int((events[0, 2] - self.events_time_offset) / 1e3)
        end_time = int((events[-1, 2] - self.events_time_offset) / 1e3)
        sync_flow = self.flow_syncer.sync(start_time, end_time, self.inverse_flow)
        return events, gray_images, gray_time, sync_flow
