"""Before running this converter, MetaVision SDK should be installed on your
system. It is very important to also reference the HDF5 decoder path, refer
to installation instructions in the link below:
https://docs.prophesee.ai/stable/installation/linux.html#chapter-installation-linux
"""

import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from .base import ConvertBase
from ..writers import WriterEvents, WriterGray, WriterFlow

from typing import Any, Dict, List, Tuple, Callable, Union


class ConvertProphesee(ConvertBase):
    """Converts PROPHESEE HDF5 data to eWiz compatible format.
    """
    def __init__(
        self,
        data_dir: str,
        out_dir: str
    ) -> None:
        super().__init__(data_dir, out_dir)
        self._init_events()
        self._init_writers()

    def _init_events(self) -> None:
        """Initializes events file path.
        """
        self.events_path = self.data_dir
        self.events_file = h5py.File(self.events_path, "r")
        self.events_x = self.events_file["CD"]["events"]["x"]
        self.events_y = self.events_file["CD"]["events"]["y"]
        self.events_time = self.events_file["CD"]["events"]["t"]
        self.events_pol = self.events_file["CD"]["events"]["p"]

    def _init_writers(self) -> None:
        """Initializes writers.
        """
        self.events_writer = WriterEvents(self.out_dir)

    def _init_events_stride(self, events_stride: int = 1e4) -> None:
        """Initializes events stride.
        """
        self.events_stride = events_stride
        self.events_indices = np.arange(0, self.events_x.shape[0], self.events_stride)
        if self.events_indices[-1] != self.events_x.shape[0] - 1:
            self.events_indices = np.append(self.events_indices, self.events_x.shape[0] - 1)
        self.events_size = len(self.events_indices) - 1

    def convert(self, events_stride: int = 1e4) -> None:
        """Converts PROPHESEE data.
        """
        print("# === Converting PROPHESEE Data === #")
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
            events[:, 2] = self.events_time[start:end]
            events[:, 3] = self.events_pol[start:end]
            events = events.astype(np.int64)
            self.events_writer.write(events=events)
