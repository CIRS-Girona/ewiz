import cv2
import numpy as np

from typing import Any, Dict, List, Tuple, Callable, Union


def clip_events(events: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Clips and filters events with coordinates bigger than the requested
    image size."""
    events = events[
        np.logical_and(
            np.logical_and(events["x"] >= 0, events["x"] < image_size[1]),
            np.logical_and(events["y"] >= 0, events["y"] < image_size[0]),
        )
    ]
    return events
