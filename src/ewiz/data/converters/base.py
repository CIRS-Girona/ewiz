import os
import h5py
import hdf5plugin
import numpy as np

from tqdm import tqdm

from ewiz.core.utils import create_dir, save_json

from typing import Any, Dict, List, Tuple, Callable, Union


class ConvertBase():
    """Base data converter.
    """
    def __init__(
        self,
        data_dir: str,
        out_dir: str,
        sensor_size: Tuple[int, int]
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = create_dir(out_dir)
        self.sensor_size = sensor_size
        self._save_props()

    def _init_events(self) -> None:
        """Initializes events file path.
        """
        raise NotImplementedError

    def _init_images(self) -> None:
        """Initializes images file path.
        """
        raise NotImplementedError

    # TODO: Check format requirements
    def _save_props(self) -> None:
        """Saves camera properties.
        """
        props = {}
        props.update({"sensor_size": self.sensor_size})
        print("# ===== Saving Data Properties ===== #")
        file_path = os.path.join(self.out_dir, "props.json")
        save_json(props, file_path)
        print("# ===== Data Properties Saved ===== #")
