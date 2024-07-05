import numpy as np

from ..readers import ReaderBase, ReaderFlow

from typing import Any, Dict, List, Tuple, Callable, Union


class LoaderBase():
    """Base data loader.
    """
    def __init__(
        self,
        data_dir: str,
        data_stride: int = None,
        data_range: Tuple[int, int] = None,
        reader_mode: str = "base"
    ) -> None:
        self.data_dir = data_dir
        self.data_stride = data_stride
        self.data_range = data_range
        self.reader_mode = reader_mode
        self.index = 0
        self.indices: np.ndarray = None

    def __iter__(self) -> Callable:
        """Returns iterator.
        """
        return self

    def __next__(self) -> Tuple[np.ndarray]:
        """Iterates over data.
        """
        if self.index < self.data_size:
            start = int(self.indices[self.index])
            end = int(self.indices[self.index + 1])
            data = self.reader[start:end]

            # If events array is too small, we stop the iterator
            if len(data[0]) < 1:
                raise StopIteration

            self.index += 1
            return data
        raise StopIteration

    # TODO: Remove manual check
    def _init_reader(self, clip_mode: str) -> None:
        """Initializes data reader.
        """
        if self.reader_mode == "base":
            self.reader = ReaderBase(self.data_dir, clip_mode)
        elif self.reader_mode == "flow":
            self.reader = ReaderFlow(self.data_dir, clip_mode)
        else:
            raise KeyError(
                f"Reader mode key '{self.reader_mode}' is not supported."
            )

    def _init_size(self) -> None:
        """Initializes data size.
        """
        self.data_size = len(self.reader)

    def _init_indices(self) -> None:
        """Initializes data indices.
        """
        raise NotImplementedError
