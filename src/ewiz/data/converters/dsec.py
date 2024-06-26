import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from .base import ConvertBase

from typing import Any, Dict, List, Tuple, Callable, Union


class ConvertDSEC(ConvertBase):
    """DSEC to eWiz data converter.
    """
    def __init__(
        self,
        data_dir: str,
        out_dir: str
    ) -> None:
        super().__init__(data_dir, out_dir)
