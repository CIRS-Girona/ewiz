import numpy as np
import cv2

from typing import Any, Dict, List, Tuple, Callable, Union


# TODO: Continue code
class VideoRendererBase():
    """Base video renderer.
    """
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
