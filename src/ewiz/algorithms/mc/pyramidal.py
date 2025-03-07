# TODO: Review general code structure
import torch
import numpy as np
import skimage.transform

from autograd_minimize import minimize

from ewiz.losses import LossBase
from .base import MotionCompensationBase
from .utils import convert_patch_to_dense_flow, update_fine_to_coarse_flow

from typing import Any, Dict, List, Tuple, Callable, Union


class MotionCompensationPyramidal(MotionCompensationBase):
    """Pyramidal motion compensation class."""

    name = "pyramidal"

    def __init__(
        self,
        image_size: Tuple[int, int],
        loss: LossBase,
        optimizer: str = "BFGS",
        flow_inits: Tuple[float, float] = (-20, 20),
        scales: Tuple[int, int] = (1, 5),
        *args,
        **kwargs,
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
        self._create_pyramidal_patches()

    def _create_pyramidal_patches(self) -> None:
        """Creates pyramidal patches."""
        for i in range(self.scales[0], self.scales[1]):
            size = (self.image_size[0] // (2**i), self.image_size[1] // (2**i))
            self.patches_size[i] = size
            self.patches_stride[i] = size
            self.grids_size[i] = self._create_patches(size, size)
            self.nums_patches[i] = int(self.grids_size[i][0] * self.grids_size[i][1])
            self.total_num_patches += self.nums_patches[i]

    def _load_patch_configs(self, scale: int) -> None:
        """Loads patch configuration."""
        self.curr_scale = scale
        self.patch_size = self.patches_size[scale]
        self.patch_stride = self.patches_stride[scale]
        self.grid_size = self.grids_size[scale]
        self.num_patches = self.nums_patches[scale]

    def _predict_flow(self, events: np.ndarray, patch_flows: np.ndarray) -> None:
        """Predict flow."""
        if self.optimized_patch_flows is not None and self.curr_scale == self.scales[0]:
            print("Using optimized patch flows from the previous iteration...")
            init_flow = np.copy(self.optimized_patch_flows[self.curr_scale])
        elif self.curr_scale > self.scales[0]:
            print("Using the coarser patch flow...")
            init_flow = skimage.transform.pyramid_expand(
                patch_flows[self.curr_scale - 1], channel_axis=0
            ).reshape(-1)
            if self.optimized_patch_flows is not None:
                init_flow = (
                    init_flow + self.optimized_patch_flows[self.curr_scale].reshape(-1)
                ) / 2
        else:
            init_flow = self._init_flow()

        # TODO: Add arguments for optimizer options
        optimizer_opts = {"gtol": 1e-5, "disp": True, "maxiter": 80, "eps": 1}
        optimizer_out = minimize(
            fun=lambda flow: self.objective_function(events, flow, patch_flows),
            x0=init_flow,
            backend="torch",
            precision="float64",
            method=self.optimizer,
            torch_device="cuda",
            options=optimizer_opts,
        )
        return optimizer_out

    def _iterate_over_scales(self, events: np.ndarray) -> None:
        """Iterates over all scales."""
        # TODO: Add CUDA option, available optimizers
        patch_flows = {}
        events = torch.from_numpy(events).double().requires_grad_().to("cuda")
        for scale in range(self.scales[0], self.scales[1]):
            self._load_patch_configs(scale)
            print(f"Optimizing for scale {scale}...")
            optimizer_out = self._predict_flow(events, patch_flows)
            patch_flows[scale] = optimizer_out.x.reshape(((2,) + self.grid_size))
        return patch_flows, optimizer_out

    # ===== General Functions ===== #
    def objective_function(
        self,
        events: np.ndarray,
        patch_flow: np.ndarray,
        patch_flows: Dict[int, np.ndarray],
    ) -> None:
        """Main objective function."""
        assert self.curr_scale not in patch_flows.keys(), (
            f"Flow already computed for scale {self.curr_scale}."
            "Check your code logic."
        )
        final_flows = patch_flows.copy()
        final_flows.update({self.curr_scale: patch_flow})
        # TODO: Check if properties change, negative sign
        curr_flow = final_flows[self.curr_scale]
        dense_flow = convert_patch_to_dense_flow(
            curr_flow, self.grid_size, self.image_size
        )
        patch_flow = -final_flows[self.curr_scale].reshape((1, 2) + self.grid_size)[0]
        loss = self.loss.calculate(
            events=events, patch_flow=patch_flow, dense_flow=dense_flow
        )
        return loss

    def optimize(self, events: np.ndarray) -> None:
        """Main optimization function."""
        print("# ===== Starting Optimization ===== #")
        print(f"Total degrees of freedom is {2*self.total_num_patches}...")
        patch_flows, optimizer_out = self._iterate_over_scales(events)
        print("# ===== Optimization Done ===== #")
        patch_flows = update_fine_to_coarse_flow(patch_flows)
        print("# ===== Flow Refined ===== #")
        return patch_flows, optimizer_out
