import torch
import numpy as np

from typing import Any, Dict, List, Tuple, Callable, Union


class WarperBase():
    """Base events warper class.
    """
    def __init__(
        self,
        image_size: Tuple[int, int]
    ) -> None:
        self.image_size = image_size

    def _get_ref_time(
        self, events: torch.Tensor, direction: Union[str, float] = "start"
    ) -> torch.Tensor:
        """Gets warp reference time.
        """
        if type(direction) is float:
            max_time = torch.max(events[..., 2], dim=-1).values
            min_time = torch.min(events[..., 2], dim=-1).values
            delta_time = max_time - min_time
            warp_time = min_time + delta_time*direction
            return warp_time
        # Convert string input to float
        elif direction == "start":
            return torch.min(events[..., 2], dim=-1).values
        elif direction == "end":
            return torch.max(events[..., 2], dim=-1).values
        elif direction == "mid":
            return self._get_ref_time(events, 0.5)
        elif direction == "random":
            random_direction = np.random.uniform(low=0.0, high=1.0)
            return self._get_ref_time(events, random_direction)
        elif direction == "before":
            return self._get_ref_time(events, -1.0)
        elif direction == "after":
            return self._get_ref_time(events, 2.0)
        error = (
            f"The chosen warp direction '{direction}' is not supported. "
            "Check your warp inputs."
        )
        raise ValueError(error)

    def warp(self, events: torch.Tensor, *args, **kwargs) -> None:
        """Main warping function.
        """
        raise NotImplementedError
