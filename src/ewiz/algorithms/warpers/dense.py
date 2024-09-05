import torch
import numpy as np

from .base import WarperBase

from typing import Any, Dict, List, Tuple, Callable, Union


class WarperDense(WarperBase):
    """Dense events warper.
    """
    def __init__(
        self,
        image_size: Tuple[int, int]
    ) -> None:
        super().__init__(image_size)

    def warp(
        self,
        events: torch.Tensor,
        flow: torch.Tensor,
        direction: Union[str, float],
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Main warping function.
        """
        ref_time = self._get_ref_time(events, direction)
        delta_times = events[..., 2] - ref_time
        # Check dimensions
        if len(events.shape) == 2:
            events = events[None, ...]
            flow = flow[None, ...]
            ref_time = ref_time[None, ...]
            delta_times = delta_times[None, ...]
        # Check dimensional compatibility
        assert(len(delta_times.shape) + 1 == len(flow.shape) - 1 == 3), (
            f"Shape of events timestamps data of size '{len(delta_times.shape)}' "
            f"is incompatible with shape of flow data of size '{len(flow.shape)}'."
        )

        # Warp events
        warped_events = events.clone()
        flat_flow = flow.reshape((flow.shape[0], 2, -1))
        flat_events = events[..., 1].long()*self.image_size[0] + events[..., 0].long()
        # TODO: Continue code
