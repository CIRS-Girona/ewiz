import numpy as np

from .base import IteratorBase
from ..readers import ReaderBase, ReaderFlow

from typing import Any, Dict, List, Tuple, Callable, Union


class IteratorGray(IteratorBase):
    """Grayscale images data loader.
    """
    def __init__(
        self,
        data_dir: str,
        data_stride: int = None,
        data_range: Tuple[int, int] = None,
        reader_mode: str = "base",
        *args,
        **kwargs
    ) -> None:
        super().__init__(data_dir, data_stride, data_range, reader_mode, *args, **kwargs)
        self._init_reader(clip_mode="images")
        self._init_size()
        self._init_opts()
        self._init_indices()
        self._init_iter_size()

    def _init_opts(self) -> None:
        """Initializes data options.
        """
        # TODO: Check data stride
        if self.data_stride is None:
            self.data_stride = 1
        if self.data_range is None:
            self.data_range = (0, self.data_size)
        elif self.data_range[1] < 0:
            self.data_range = (self.data_range[0], self.data_size - self.data_range[1])

    def _init_indices(self) -> None:
        """Initializes data indices.
        """
        self.index = 0
        self.indices = np.arange(self.data_range[0], self.data_range[1], self.data_stride)
        if self.indices[-1] != self.data_range[1] - 1:
            self.indices = np.append(self.indices, self.data_range[1] - 1)
