import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from .base import WriterBase

from typing import Any, Dict, List, Tuple, Callable, Union


class WriterGray(WriterBase):
    """Data writer for grayscale images.
    """
    def __init__(self, out_dir: str) -> None:
        super().__init__(out_dir)

    def write(self, gray_images: np.ndarray, time: np.ndarray) -> None:
        """Main data writing function.
        """
        raise NotImplementedError

    def _init_gray(self) -> None:
        """Initializes grayscale images file.
        """
        raise NotImplementedError
