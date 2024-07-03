import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from .base import ConvertBase
from ..writers import WriterEvents, WriterGray, WriterFlow

from typing import Any, Dict, List, Tuple, Callable, Union


class ConvertMVSEC(ConvertBase):
    """MVSEC to eWiz data converter.
    """
    def __init__(
        self,
        data_dir: str,
        out_dir: str
    ) -> None:
        super().__init__(data_dir, out_dir)
        self._init_gt()
        self._init_writers()
        self._get_min_time()

    def _init_events(self) -> None:
        """Initializes events file path.
        """
        for dir, subdirs, files in os.walk(self.data_dir):
            for file in files:
                if "_data.hdf5" in file:
                    self.data_path = os.path.join(dir, file)
                    self.data_file = h5py.File(self.data_path, "r")
                    break
        self.data_group = self.data_file["davis"]["left"]
        self.events = self.data_group["events"]
        self.gray_to_events = self.data_group["image_raw_event_inds"]

    def _init_images(self) -> None:
        """Initializes images file path.
        """
        self.gray_images = self.data_group["image_raw"]
        self.gray_time = self.data_group["image_raw_ts"]

    def _init_gt(self) -> None:
        """Initializes ground truth flow file.
        """
        for dir, subdirs, files in os.walk(self.data_dir):
            for file in files:
                if "_gt.hdf5" in file:
                    self.gt_path = os.path.join(dir, file)
                    self.gt_file = h5py.File(self.gt_path, "r")
                    break
        self.gt_group = self.gt_file["davis"]["left"]
        self.gt_flows = self.gt_group["flow_dist"]
        self.gt_time = self.gt_group["flow_dist_ts"]

    def _init_writers(self) -> None:
        """Initializes writers.
        """
        self.events_writer = WriterEvents(self.out_dir)
        self.gray_writer = WriterGray(self.out_dir)
        self.flow_writer = WriterFlow(self.out_dir)

    def _init_events_stride(self, events_stride: int = 1e4) -> None:
        """Initializes events stride.
        """
        self.events_stride = events_stride
        self.events_indices = np.arange(0, self.events.shape[0], self.events_stride)
        if self.events_indices[-1] != self.events.shape[0] - 1:
            self.events_indices = np.append(self.events_indices, self.events.shape[0] - 1)
        self.events_size = len(self.events_indices) - 1

    def _init_gray_stride(self) -> None:
        """Initializes grayscale stride.
        """
        self.gray_indices = np.arange(0, self.gray_images.shape[0])
        self.gray_size = len(self.gray_indices)

    def _init_gt_stride(self) -> None:
        """Initializes ground truth stride.
        """
        self.gt_indices = np.arange(0, self.gt_flows.shape[0])
        self.gt_size = len(self.gt_indices)

    def _get_min_time(self) -> None:
        """Gets minimum timestamp.
        """
        self.min_time = int(self.events[0, 2]*1e6)
        gray_min_time = int(self.gray_time[0]*1e6)
        gt_min_time = int(self.gt_time[0]*1e6)
        if gray_min_time < self.min_time:
            self.min_time = gray_min_time
        if gt_min_time < self.min_time:
            self.min_time = gt_min_time

    def convert(self, events_stride: int = 1e4) -> None:
        """Converts MVSEC data.
        """
        print("# === Converting MVSEC Data === #")
        print("# === Converting Events === #")
        self._init_events_stride(events_stride)
        progress_bar = tqdm(range(self.events_size))
        for i in progress_bar:
            start = int(self.events_indices[i])
            end = int(self.events_indices[i + 1])
            events = self.events[start:end + 1]
            events[:, 2] = events[:, 2]*1e6 - self.min_time
            events[events[:, 3] < 0, 3] = 0
            events = events.astype(np.int64)
            self.events_writer.write(events=events)
        # Map time to events
        self.events_writer.map_time_to_events()

        print("# === Converting Grayscale Images === #")
        self._init_gray_stride()
        progress_bar = tqdm(range(self.gray_size))
        for i in progress_bar:
            gray_image = self.gray_images[i]
            time = self.gray_time[i]*1e6 - self.min_time
            self.gray_writer.write(gray_image, time)
        # Map time to grayscale images
        self.gray_writer.map_time_to_gray()
        self.gray_writer.map_gray_to_events()

        print("# === Converting Ground Truth Flows === #")
        self._init_gt_stride()
        progress_bar = tqdm(range(self.gt_size))
        for i in progress_bar:
            flow = self.gt_flows[i]
            time = self.gt_time[i]*1e6 - self.min_time
            self.flow_writer.write(flow, time)
        # Map time to flow
        self.flow_writer.map_time_to_flow()
        self.flow_writer.map_flow_to_events()
