import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from .base import WriterBase

from typing import Any, Dict, List, Tuple, Callable, Union


class WriterEvents(WriterBase):
    """Event-based data writer.
    """
    def __init__(self, out_dir: str) -> None:
        super().__init__(out_dir)
        self._init_events()

    def _init_events(self) -> None:
        """Initializes events HDF5 file.
        """
        self.events_path = os.path.join(self.out_dir, "events.hdf5")
        self.events_file = h5py.File(self.events_path, "a")
        self.events_flag = False
        # TODO: Check group creation method
        self.events_group = self.events_file.create_group("events")

    def write(self, events: np.ndarray) -> None:
        """Main data writing function.
        """
        if self.events_flag is False:
            self.time_offset = 0
            if events[0, 2] != 0:
                self._save_time_offset(
                    data_file=self.events_file, time=events[0, 2]
                )
                self.time_offset = events[0, 2]

            # Create HDF5 groups
            self.events_x = self.events_group.create_dataset(
                name="x", data=events[:, 0],
                chunks=True, maxshape=(None,), dtype=np.uint16,
                **self.compressor
            )
            self.events_y = self.events_group.create_dataset(
                name="y", data=events[:, 1],
                chunks=True, maxshape=(None,), dtype=np.uint16,
                **self.compressor
            )
            self.events_time = self.events_group.create_dataset(
                name="time", data=events[:, 2] - self.time_offset,
                chunks=True, maxshape=(None,), dtype=np.int64,
                **self.compressor
            )
            self.events_pol = self.events_group.create_dataset(
                name="polarity", data=events[:, 3],
                chunks=True, maxshape=(None,), dtype=bool,
                **self.compressor
            )
            self.events_flag = True
        else:
            data_points = events.shape[0]
            dataset_points = self.events_x.shape[0]
            all_points = data_points + dataset_points
            self.events_x.resize(all_points, axis=0)
            self.events_x[-data_points:] = events[:, 0]
            self.events_y.resize(all_points, axis=0)
            self.events_y[-data_points:] = events[:, 1]
            self.events_time.resize(all_points, axis=0)
            self.events_time[-data_points:] = events[:, 2] - self.time_offset
            self.events_pol.resize(all_points, axis=0)
            self.events_pol[-data_points:] = events[:, 3]

    def map_time_to_events(self) -> None:
        """Maps timestamps to events indices.
        """
        print("# === Mapping Timestamps to Events Indices === #")
        start_value = np.floor(self.events_time[0]/1e3)
        end_value = np.ceil(self.events_time[-1]/1e3)
        sorted_data = self.events_time
        data_file = self.events_file
        data_name = "time_to_events"
        offset_value = 0
        side = "left"
        chunks = 75e3
        addition = 0.0
        division = 1e3

        # TODO: Review arguments
        self.map_data_out_memory(
            start_value, end_value, sorted_data,
            data_file, data_name, offset_value, side, chunks,
            addition, division
        )
