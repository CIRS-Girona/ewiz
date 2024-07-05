import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from typing import Any, Dict, List, Tuple, Callable, Union


class ConvertBase():
    """Base data converter.
    """
    def __init__(
        self,
        data_dir: str,
        out_dir: str
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = out_dir

    def _init_events(self) -> None:
        """Initializes events file path.
        """
        raise NotImplementedError

    def _init_images(self) -> None:
        """Initializes images file path.
        """
        raise NotImplementedError
