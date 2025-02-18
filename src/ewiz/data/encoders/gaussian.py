import numpy as np

from .base import EncoderBase

from typing import Any, Dict, List, Tuple, Callable, Union


class EncoderGaussian(EncoderBase):
    """Gaussian encoder.
    """
    name = "gaussian"

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_splits: int
    ) -> None:
        super().__init__(image_size, num_splits)

    # TODO: Check polarity
    def encode(self, events: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Main encoding function.
        """
        split_size = int(events.shape[0]/self.num_splits)
        encoded_events = np.zeros((2, *self.image_size, self.num_splits))
        for i in range(self.num_splits):
            split = events[split_size*i:split_size*(i + 1), :]
            split[:, 2] = (
                (split[:, 2] - np.min(split[:, 2]))
                /(np.max(split[:, 2]) - np.min(split[:, 2]))
            )
            encoded_events[0, :, :, i] = self._apply_gaussian(split[split[:, 3] > 0])
            encoded_events[1, :, :, i] = self._apply_gaussian(split[split[:, 3] == 0])
        encoded_events = encoded_events.astype(np.float32)
        return encoded_events

    def _apply_gaussian(self, events: np.ndarray) -> np.ndarray:
        """Applies Gaussian encoding scheme.
        """
        time = events[:, 2]
        polarity = events[:, 3]

        average_time = np.mean(time)
        standard_time = np.std(time) if time.shape[0] > 1 else 1
        gaussian0 = 1.0/np.sqrt(2.0*np.pi)/standard_time
        gaussian1 = -(time - average_time)**2/(2*standard_time**2)
        distribution = gaussian0*np.exp(gaussian1)

        # TODO: Modify polarity code
        norm_factor = np.sum(np.abs(polarity))/np.sum(distribution)
        image = np.zeros(self.image_size)
        np.add.at(
            image,
            (events[:, 1].astype(np.int32), events[:, 0].astype(np.int32)),
            np.ceil(distribution*norm_factor)
        )
        image = image.astype(np.uint8)
        return image
