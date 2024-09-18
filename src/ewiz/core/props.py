"""Utilities to import data properties.
"""
import os
import numpy as np

from .utils import read_json

from typing import Any, Dict, List, Tuple, Callable, Union


def import_props(data_dir: str) -> Dict:
    """Imports dataset properties.
    """
    props_path = os.path.join(data_dir, "props.json")
    props = read_json(props_path)

    # Dataset properties
    props["sensor_size"] = tuple(props["sensor_size"])
    return props
