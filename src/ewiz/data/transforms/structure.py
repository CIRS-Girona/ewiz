import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

from dataclasses import dataclass

from typing import Any, Dict, List, Tuple, Callable, Union


@dataclass(frozen=True)
class EventsToStructured():
    """Converts unstructured events to structured events.
    """
    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Call function.
        """
        events = events.copy()
        events = unstructured_to_structured(
            events,
            np.dtype([("x", np.int32), ("y", np.int32), ("t", np.float64), ("p", np.int32)])
        )
        return events


@dataclass(frozen=True)
class EventsToUnstructured():
    """Converts structured events to unstructured events.
    """
    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Call function.
        """
        events = events.copy()
        trans_events = np.zeros((events["x"].shape[0], 4), dtype=np.float64)
        trans_events[:, 0] = events["x"]
        trans_events[:, 1] = events["y"]
        trans_events[:, 2] = events["t"]
        trans_events[:, 3] = events["p"]
        return trans_events
