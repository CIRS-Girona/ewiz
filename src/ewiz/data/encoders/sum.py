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
        split_size = int(events.shape[0]/self.num_splits)
        encoded_events = np.zeros((2, *self.image_size, self.num_splits))
        for i in range(self.num_splits):
            split = events[split_size*i:split_size*(i + 1), :]
            encoded_events[0, :, :, i] = self._apply_sum(split[split[:, 3] > 0])
            encoded_events[0, :, :, i] = self._apply_sum(split[split[:, 3] < 0])
        encoded_events = encoded_events.astype(np.float32)

        # Normalize encoded events
        if normalize:
            encoded_events = (
                (encoded_events - np.min(encoded_events))
                /(np.max(encoded_events) - np.min(encoded_events))
            )
        return encoded_events

    def _apply_sum(self, events: np.ndarray) -> np.ndarray:
        """Applies Gaussian encoding scheme.
        """
        image = np.zeros(self.image_size)
        np.add.at(
            image, (events[:, 1].astype(np.int32), events[:, 0].astype(np.int32)), 1
        )
        image = image.astype(np.uint8)
        return image
