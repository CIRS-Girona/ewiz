import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from typing import Any, Dict, List, Tuple, Callable, Union


class WriterBase():
    """Base data writer for saving event-based data in HDF5 format.
    """
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self._init_compressor()

    def write(self, *args, **kwargs) -> None:
        """Main data writing function.
        """
        raise NotImplementedError

    def _init_compressor(self) -> None:
        """Initializes data compressor.
        """
        self.compressor = hdf5plugin.Blosc(cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE)

    def _save_time_offset(self, data_file: h5py.File, time: int) -> None:
        """Saves time offset to dataset file.
        """
        time = np.array(time, dtype=np.int64)[None, ...]
        data_file.create_dataset(
            name="time_offset", data=time,
            chunks=True, maxshape=(1,), dtype=np.int64,
            **self.compressor
        )

    @staticmethod
    def search_sorted_out_memory(
        data: h5py.Dataset,
        target: float,
        side: str = "left",
        chunks: int = 32,
        addition: float = 0.0,
        division: float = 1.0
    ) -> int:
        """Applies a binary search algorithm to search sorted data without loading
        it all to memory.
        """
        samples = data.shape[0]
        samples_per_chunk = samples//chunks
        sample_indices = np.arange(0, samples, samples_per_chunk)
        if sample_indices[-1] != samples - 1:
            sample_indices = np.append(sample_indices, samples - 1)
        search_index = samples - 1
        lower = 0
        upper = sample_indices.shape[0] - 2
        while lower <= upper:
            middle = (lower + upper)//2
            start_index = int(sample_indices[middle])
            end_index = int(sample_indices[middle + 1])
            data_chunk: np.ndarray = (data[start_index:end_index] + addition)/division
            data_samples = data_chunk.shape[0]
            left = np.searchsorted(data_chunk, target, side="left")
            right = np.searchsorted(data_chunk, target, side="right")
            if left == 0 and right == 0:
                upper = middle - 1
            elif left == data_samples and right == data_samples:
                lower = middle + 1
            else:
                search_index = (
                    left + sample_indices[middle] if side == "left"
                    else right + sample_indices[middle]
                )
                break
        return search_index

    def map_data_out_memory(
        self,
        start_value: int,
        end_value: int,
        sorted_data: h5py.Dataset,
        data_file: h5py.File,
        data_name: str,
        offset_value: float = 0.0,
        side: str = "left",
        chunks: int = 32,
        addition: float = 0.0,
        division: float = 1.0
    ) -> None:
        """Maps data without loading it all to memory.
        """
        start_flag = False
        all_values = np.arange(start_value, end_value)
        num_values = all_values.shape[0]
        progress_bar = tqdm(range(num_values))
        for i in progress_bar:
            target_value = all_values[i] + offset_value
            search_index = self.search_sorted_out_memory(
                sorted_data, target_value, side, chunks, addition, division
            )
            search_index = np.array(search_index, dtype=np.int64)[None, ...]

            # Save to dataset file
            if start_flag is False:
                data_file.create_dataset(
                    name=data_name, data=search_index,
                    chunks=True, maxshape=(num_values,), dtype=np.int64,
                    **self.compressor
                )
                start_flag = True
            else:
                data_points = data_file[data_name].shape[0]
                data_file[data_name].resize(data_points + 1, axis=0)
                data_file[data_name][-1:] = search_index

    # TODO: Currently, this function should be used when we have a low number of
    # data points. The out of memory algorithm needs to be reviewed.
    def map_data_in_memory(
        self,
        start_value: int,
        end_value: int,
        sorted_data: h5py.Dataset,
        data_file: h5py.File,
        data_name: str,
        offset_value: float = 0.0
    ) -> None:
        """Maps data, but requires loading it all to memory.
        """
        start_flag = False
        all_values = np.arange(start_value, end_value)
        num_values = all_values.shape[0]
        progress_bar = tqdm(range(num_values))
        for i in progress_bar:
            target_value = all_values[i] + offset_value
            search_index = np.searchsorted(sorted_data, target_value, side="right")
            search_index = np.array(search_index, dtype=np.int64)[None, ...]

            # Save to dataset file
            if start_flag is False:
                data_file.create_dataset(
                    name=data_name, data=search_index,
                    chunks=True, maxshape=(None,), dtype=np.int64,
                    **self.compressor
                )
                start_flag = True
            else:
                data_points = data_file[data_name].shape[0]
                data_file[data_name].resize(data_points + 1, axis=0)
                data_file[data_name][-1:] = search_index
