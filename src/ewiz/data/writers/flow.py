import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from .base import WriterBase

from typing import Any, Dict, List, Tuple, Callable, Union


class WriterFlow(WriterBase):
    """Data writer for flow data.
    """
    def __init__(self, out_dir: str) -> None:
        super().__init__(out_dir)
        self._init_events()
        self._init_flow()

    def _init_events(self) -> None:
        """Initializes events HDF5 file.
        """
        self.events_path = os.path.join(self.out_dir, "events.hdf5")
        self.events_file = h5py.File(self.events_path, "a")
        self.events_flag = False

    def _init_flow(self) -> None:
        """Initializes flow file.
        """
        self.flow_path = os.path.join(self.out_dir, "flow.hdf5")
        self.flow_file = h5py.File(self.flow_path, "a")
        self.flow_flag = False

    # TODO: Check data type
    def write(self, flow: np.ndarray, time: int) -> None:
        """Main data writing function.
        """
        flow = flow[None, ...].astype(np.uint8)
        image_size = (flow.shape[2], flow.shape[3])

        # TODO: Check time format
        if self.flow_flag is False:
            self.time_offset = 0
            if time != 0:
                self.time_offset = time
            self._save_time_offset(data_file=self.flow_file, time=time)

            # Create HDF5 groups
            self.flows = self.flow_file.create_dataset(
                name="flows", data=flow,
                chunks=True, maxshape=(None, 2, *image_size), dtype=np.float64,
                **self.compressor
            )
            time: np.ndarray = np.array(time, dtype=np.int64)[None, ...]
            self.flows_time = self.flow_file.create_dataset(
                name="time", data=time - self.time_offset,
                chunks=True, maxshape=(None,), dtype=np.int64,
                **self.compressor
            )
            self.flow_flag = True
        else:
            data_points = flow.shape[0]
            dataset_points = self.flows.shape[0]
            all_points = data_points + dataset_points
            self.flows.resize(all_points, axis=0)
            self.flows[-data_points:] = flow
            self.flows_time.resize(all_points, axis=0)
            self.flows_time[-data_points:] = time - self.time_offset
