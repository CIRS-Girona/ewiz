import torch
import numpy as np

from ewiz.losses import LossBase
from .base import MotionCompensationBase

from typing import Any, Dict, List, Tuple, Callable, Union


class MotionCompensationPyramidal(MotionCompensationBase):
    """Pyramidal motion compensation class.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        loss: LossBase,
        optimizer: str = "BFGS",
        flow_inits: Tuple[float, float] = (-20, 20),
        scales: Tuple[int, int] = (1, 5),
        *args,
        **kwargs
    ) -> None:
        super().__init__(image_size, loss, optimizer, flow_inits)
        self.scales = scales

        self.patches_size = {}
        self.patches_stride = {}
        self.grids_size = {}
        # TODO: We do not need to save patches currently
        self.patches = {}
        self.nums_patches = {}
        self.total_num_patches = 0

    def _create_pyramidal_patches(self) -> None:
        """Creates pyramidal patches.
        """
        for i in range(self.scales[0], self.scales[5]):
            size = (self.image_size[0]//(2**i), self.image_size[1]//(2**i))
            self.patches_size[i] = size
            self.patches_stride[i] = size
            self.grids_size[i] = self._create_patches(size, size)
            self.nums_patches[i] = int(self.grids_size[i][0]*self.grids_size[i][1])
            self.total_num_patches += self.nums_patches[i]

    def _load_patch_configs(self, scale: int) -> None:
        """Loads patch configuration.
        """
        self.current_scale = scale
        self.patch_size = self.patches_size[scale]
        self.patch_stride = self.patches_stride[scale]
        self.grid_size = self.grids_size[scale]
        self.num_patches = self.nums_patches[scale]

    def _predict_flow(self, events: np.ndarray, patch_flows: np.ndarray) -> None:
        """Predict flow.
        """
        pass
        # TODO: Continue code
