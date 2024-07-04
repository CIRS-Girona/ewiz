import numpy as np

from .base import LoaderBase
from ..readers import ReaderBase, ReaderFlow

from typing import Any, Dict, List, Tuple, Callable, Union


class LoaderEvents(LoaderBase):
    """Events data loader.
    """
    def __init__(
        self,
        data_dir: str,
        data_stride: int = None,
        data_range: Tuple[int, int] = None,
        reader_mode: str = "base"
    ) -> None:
        super().__init__(data_dir, data_stride, data_range, reader_mode)
        self._init_reader(clip_mode="events")
        self._init_size()
        print(self.data_size)
