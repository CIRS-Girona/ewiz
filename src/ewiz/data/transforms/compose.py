import numpy as np

from typing import Any, Dict, List, Tuple, Callable, Union


class Compose():
    """General compose class.
    """
    def __init__(
        self,
        transforms: List[Callable],
        use_tonic: bool = False
    ) -> None:
        self.transforms = transforms
        self.use_tonic = use_tonic
        # Structure transforms
        self.to_structured = None
        self.to_unstructured = None

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Call function.
        """
        events_flag = len(data.shape) == 2 and data.shape[1] == 4
        if self.use_tonic and events_flag:
            data = self.to_structured(data)

        # Apply transforms
        for transform in self.transforms:
            if len(data) == 0:
                break
            data = transform(data)

        if self.use_tonic and events_flag:
            data = self.to_unstructured(data)
        return data
