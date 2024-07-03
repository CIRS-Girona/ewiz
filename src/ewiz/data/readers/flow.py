import os
import h5py
import hdf5plugin
import numpy as np

from .base import ReaderBase
# TODO: Check working code
from .utils._sync import FlowSync

from typing import Any, Dict, List, Tuple, Callable, Union


class ReaderFlow(ReaderBase):
    """Flow data reader.
    """
    def __init__(
        self,
        data_dir: str,
        clip_mode: str = "events"
    ) -> None:
        super().__init__(data_dir, clip_mode)
        self._init_flows()

    def _init_flows(self) -> None:
        """Initializes flows file.
        """
        self.flow_path = os.path.join(self.data_dir, "flow.hdf5")
        self.flow_file = h5py.File(self.flow_path, "r")
        # Flows data
        self.flows = self.flow_file["flows"]
        self.flows_time = self.flow_file["time"]
        # Initialize synchronizer
        self.flow_syncer = FlowSync(self.flow_file)

    def _clip_with_events(self, start_index: int, end_index: int = None) -> Tuple[np.ndarray]:
        """Clips data with events indices.
        """
        events, gray_images, gray_time = super()._clip_with_events(start_index, end_index)
        start_time = int((events[0, 2] - self.events_time_offset)/1e3)
        end_time = int((events[-1, 2] - self.events_time_offset)/1e3)
        sync_flow = self.flow_syncer.sync(start_time, end_time)
        return events, gray_images, gray_time, sync_flow

    def _clip_with_time(self, start_time: int, end_time: int = None) -> Tuple[np.ndarray]:
        """Clips data with timestamps.
        """
        events, gray_images, gray_time = super()._clip_with_time(start_time, end_time)
        sync_flow = self.flow_syncer.sync(start_time, end_time)
        return events, gray_images, gray_time, sync_flow

    def _clip_with_images(self, start_index: int, end_index: int = None) -> Tuple[np.ndarray]:
        """Clips data with images.
        """
        events, gray_images, gray_time = super()._clip_with_images(start_index, end_index)
        start_time = int((events[0, 2] - self.events_time_offset)/1e3)
        end_time = int((events[-1, 2] - self.events_time_offset)/1e3)
        sync_flow = self.flow_syncer.sync(start_time, end_time)
        return events, gray_images, gray_time, sync_flow
