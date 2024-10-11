import os
import h5py
import hdf5plugin
import numpy as np

import roslib
import rosbag

from tqdm import tqdm

from .base import ConvertBase
from ..writers import WriterEvents, WriterGray

from typing import Any, Dict, List, Tuple, Callable, Union


# TODO: Add in separate utilities file
def ros_message_to_cv_image(message: Any) -> np.ndarray:
    """Converts ROS message to OpenCV image.
    """
    image = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.width, -1
    )
    return image
