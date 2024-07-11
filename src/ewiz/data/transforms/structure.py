import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

from dataclasses import dataclass

from typing import Any, Dict, List, Tuple, Callable, Union


@dataclass(frozen=True)
class EventsToStructured():
    """Converts unstructured events to structured events.
    """
    pass
