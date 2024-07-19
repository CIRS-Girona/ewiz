import numpy as np

from typing import Any, Dict, List, Tuple, Callable, Union


class EncoderBase():
    """Base encoder.
    """
    name: str = None

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_splits: int
    ) -> None:
        self.image_size = image_size
        self.num_splits = num_splits

    def encode(self, events: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Main encoding function.
        """
        raise NotImplementedError
