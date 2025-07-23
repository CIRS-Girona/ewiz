import cv2
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

from dataclasses import dataclass

from .utils import *

from typing import Any, Dict, List, Tuple, Callable, Union


@dataclass(frozen=True)
class EventsCenterCrop:
    """Events center crop."""

    in_size: Tuple[int, int]
    out_size: Tuple[int, int]

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Call function."""
        events = events.copy()
        offsets = (
            int((self.in_size[0] - self.out_size[0]) / 2),
            int((self.in_size[1] - self.out_size[1]) / 2),
        )
        events = events[
            np.logical_and(
                np.logical_and(
                    events["x"] >= offsets[1],
                    events["x"] < self.in_size[1] - offsets[1],
                ),
                np.logical_and(
                    events["y"] >= offsets[0],
                    events["y"] < self.in_size[0] - offsets[0],
                ),
            )
        ]
        events["x"] = events["x"] - offsets[1]
        events["y"] = events["y"] - offsets[0]
        return events


# TODO: Only nearest neighbor is supported
@dataclass(frozen=True)
class EventsRandomRotation:
    """Events random rotation."""

    in_size: Tuple[int, int]
    angle_range: Tuple[float, float]

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Call function."""
        events = events.copy()
        # Generate random angle
        angle = np.random.rand() * (self.angle_range[1] - self.angle_range[0])
        angle = np.radians(angle)
        # Rotate events
        events["x"] = events["x"] - self.in_size[1]
        events["y"] = events["y"] - self.in_size[0]
        x_coords = events["x"] * np.cos(angle) - events["y"] * np.sin(angle)
        y_coords = events["x"] * np.sin(angle) + events["y"] * np.cos(angle)
        x_coords = x_coords + self.in_size[1]
        y_coords = y_coords + self.in_size[0]
        events["x"] = np.round(x_coords)
        events["y"] = np.round(y_coords)
        # Filter events
        events = events[
            np.logical_and(
                np.logical_and(events["x"] >= 0, events["x"] < self.in_size[1]),
                np.logical_and(events["y"] >= 0, events["y"] < self.in_size[0]),
            )
        ]
        return events


@dataclass(frozen=True)
class EventsUndistort:
    """Undistorts event-based data, using the distortion coefficient."""

    in_size: Tuple[int, int]
    k: np.ndarray
    d: np.ndarray

    def __call__(self, events: np.ndarray) -> np.ndarray:
        reshaped_coords = np.empty((events.shape[0], 1, 2), dtype=np.float32)
        reshaped_coords[:, 0, 0] = events["x"]
        reshaped_coords[:, 0, 1] = events["y"]

        undistorted_coords = cv2.undistortPoints(
            reshaped_coords, self.k, self.d, P=None
        )
        fx, fy = self.k[0, 0], self.k[1, 1]
        cx, cy = self.k[0, 2], self.k[1, 2]

        x_pix = fx * undistorted_coords[:, 0, 0] + cx
        y_pix = fy * undistorted_coords[:, 0, 1] + cy

        # Step 3: Round to nearest pixel
        undistorted_coords[:, 0, 0] = np.round(x_pix).astype(np.int32)
        undistorted_coords[:, 0, 1] = np.round(y_pix).astype(np.int32)

        undistorted_events = events.copy()
        undistorted_events["x"] = undistorted_coords[:, 0, 0]
        undistorted_events["y"] = undistorted_coords[:, 0, 1]
        # Filter events
        undistorted_events = clip_events(undistorted_events.copy(), self.in_size)
        return undistorted_events


@dataclass(frozen=True)
class EventsClip:
    """Clips events to desired image size."""

    image_size: Tuple[int, int]

    def __call__(self, events: np.ndarray) -> np.ndarray:
        events = events.copy()
        events = clip_events(events, self.image_size)
        return events
