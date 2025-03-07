import torch
import numpy as np

from ewiz.losses import LossBase

from typing import Any, Dict, List, Tuple, Callable, Union


class MotionCompensationBase:
    """Base motion compensation class."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        loss: LossBase,
        optimizer: str = "BFGS",
        flow_inits: Tuple[float, float] = (-20, 20),
        *args,
        **kwargs,
    ) -> None:
        self.image_size = image_size
        self.loss = loss
        self.optimizer = optimizer
        self.flow_inits = flow_inits

        # Patch variables
        self.num_patches = None

        # Previously optimized flow
        self.optimized_patch_flows = None

    def _init_flow(self) -> np.ndarray:
        """Initializes flow."""
        print("# ===== Initializing Flow ===== #")
        if self.flow_inits is None:
            print("Flow patches initialized to 0...")
            init_flow = np.random.rand(2, self.num_patches).astype(np.float64)
        else:
            print(
                "Flow patches randomly initialized "
                f"between {self.flow_inits[0]} and {self.flow_inits[1]}..."
            )
            init_flow = np.random.rand(2, self.num_patches).astype(np.float64)
            init_flow[0] = (
                init_flow[0] * (self.flow_inits[1] - self.flow_inits[0])
                + self.flow_inits[0]
            )
            init_flow[1] = (
                init_flow[1] * (self.flow_inits[1] - self.flow_inits[0])
                + self.flow_inits[0]
            )
        return init_flow

    # TODO: Find a better use for this function
    def _create_patches(
        self, patch_size: Tuple[int, int], patch_stride: Tuple[int, int]
    ) -> None:
        """Creates patches."""
        h, w = self.image_size
        h_patch, w_patch = patch_size
        h_stride, w_stride = patch_stride
        x_centers = np.arange(0, w - w_patch + w_stride, w_stride) + w_patch // 2
        y_centers = np.arange(0, h - h_patch + h_stride, h_stride) + h_patch // 2
        grid_x, grid_y = np.meshgrid(x_centers, y_centers)
        grid_size = grid_x.shape
        return grid_size

    # ===== General Functions ===== #
    def objective_function(
        self, events: np.ndarray, patch_flow: np.ndarray, patch_flows: np.ndarray
    ) -> None:
        """Main objective function."""
        raise NotImplementedError

    def optimize(self, events: np.ndarray) -> None:
        """Main optimization function."""
        raise NotImplementedError
