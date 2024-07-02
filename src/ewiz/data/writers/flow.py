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
        flow = flow[None, ...].astype(np.float64)
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

    def map_time_to_flow(self) -> None:
        """Maps timestamps to flow indices.
        """
        events_group = self.events_file["events"]
        events_time = events_group["time"]
        events_time_offset = self.events_file["time_offset"]
        print("# === Mapping Timestamps to Flow Indices === #")
        start_value = np.floor(events_time[0]/1e3)
        end_value = np.ceil(events_time[-1]/1e3)
        sorted_data = (self.flows_time[:] + self.time_offset)/1e3
        data_file = self.flow_file
        data_name = "time_to_flow"
        offset_value = events_time_offset[0]

        # TODO: Review arguments
        self.map_data_in_memory(
            start_value, end_value, sorted_data,
            data_file, data_name, offset_value
        )

    # TODO: Check value difference
    def map_flow_to_events(self) -> None:
        """Maps flow indices to events indices.
        """
        events_group = self.events_file["events"]
        events_time = events_group["time"]
        events_time_offset = self.events_file["time_offset"]
        print("# === Mapping Flow Indices to Events Indices === #")
        start_value = None
        end_value = None
        sorted_data = events_time
        data_file = self.flow_file
        data_name = "flow_to_events"
        offset_value = self.time_offset
        side = "left"
        chunks = 32
        addition = events_time_offset[0]
        division = 1.0
        array_value = self.flows_time[:]

        # TODO: Review arguments
        self.map_data_out_memory(
            start_value, end_value, sorted_data,
            data_file, data_name, offset_value, side, chunks,
            addition, division, array_value
        )
