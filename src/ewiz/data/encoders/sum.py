import numpy as np

from .base import EncoderBase

from typing import Any, Dict, List, Tuple, Callable, Union


class EncoderSum(EncoderBase):
    """Sum encoder.
    """
    name = "sum"

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_splits: int
    ) -> None:
        super().__init__(image_size, num_splits)

    # TODO: Check polarity
    def encode(self, events: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Main encoding function.
        """
        raise NotImplementedError

    def _apply_sum(self, events: np.ndarray) -> np.ndarray:
        """Applies Gaussian encoding scheme.
        """
        raise NotImplementedError
