import os
import h5py
import hdf5plugin
import numpy as np

from .base import ReaderBase
from .utils.sync import FlowSync

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

    def __getitem__(self, indices: Union[int, slice]) -> Tuple[np.ndarray]:
        """Returns data based on slice.
        """
        data = super().__getitem__(indices)
        raise NotImplementedError

    def _init_flows(self) -> None:
        """Initializes flows file.
        """
        self.flow_path = os.path.join(self.data_dir, "flow.hdf5")
        self.flow_file = h5py.File(self.flow_path, "r")
        # Flows data
        self.flows = self.flow_file["flows"]
        self.flows_time = self.flow_file["time"]
