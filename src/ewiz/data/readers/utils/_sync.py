"""An optimized version of the flow synchronizer.
"""

import h5py
import hdf5plugin
import numpy as np

import cv2
import inverse_optical_flow

from typing import Any, Dict, List, Tuple, Callable, Union


def inverse_flow(flow: np.ndarray) -> np.ndarray:
    """Inverses optical flow."""
    flow = flow.astype(np.float32)
    flow, _ = inverse_optical_flow.max_method(flow)
    flow = np.nan_to_num(flow)
    return flow


class FlowSync:
    """Optical flow synchronizer."""

    def __init__(self, flow_file: h5py.File) -> None:
        self.flow_file = flow_file
        self.flows = self.flow_file["flows"]
        self.flows_time = self.flow_file["time"]
        self.time_offset = self.flow_file["time_offset"][0]
        self.time_to_flow = self.flow_file["time_to_flow"]
        self.flow_to_events = self.flow_file["flow_to_events"]

        self.has_covariances = ("covariances" in self.flow_file) and isinstance(
            self.flow_file["covariances"], h5py.Dataset
        )
        if self.has_covariances:
            self.covariances = self.flow_file["covariances"]

        self._init_flows = self._init_flows_fast

    def _init_grids(self) -> None:
        """Initializes flow grids."""
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(self.flows.shape[3]), np.arange(self.flows.shape[2])
        )
        self.grid_x = self.grid_x.astype(np.float32)
        self.grid_y = self.grid_y.astype(np.float32)
        self.grid_x_init = np.copy(self.grid_x)
        self.grid_y_init = np.copy(self.grid_y)

    def _init_masks(self) -> None:
        """Initializes flow masks."""
        self.mask_x = np.ones(self.grid_x.shape, dtype=bool)
        self.mask_y = np.ones(self.grid_y.shape, dtype=bool)

    def _init_flows_fast(
        self, flow_x: np.ndarray, flow_y: np.ndarray, delta_time: float
    ) -> None:
        """Compensates for the time misalignment between initial flow and events
        start time."""
        self.grid_x_init += flow_x * delta_time
        self.grid_y_init += flow_y * delta_time

    def _propagate_flow(
        self, flow_x: np.ndarray, flow_y: np.ndarray, delta_time: float = 1.0
    ) -> None:
        """Propagates optical flow."""
        flow_x = cv2.remap(flow_x, self.grid_x, self.grid_y, cv2.INTER_NEAREST)
        flow_y = cv2.remap(flow_y, self.grid_x, self.grid_y, cv2.INTER_NEAREST)
        self.mask_x[flow_x == 0] = False
        self.mask_y[flow_y == 0] = False
        self.grid_x += flow_x * delta_time
        self.grid_y += flow_y * delta_time

    def _remap_covariance(
        self, cov: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
    ) -> np.ndarray:
        """Remaps a (2, 2, H, W) covariance field using the flow grid."""
        warped_cov = np.zeros_like(cov)
        for i in range(2):
            for j in range(2):
                warped_cov[i, j] = cv2.remap(
                    cov[i, j],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
        return warped_cov

    def sync(self, start_time: int, end_time: int, inverse: bool = False) -> np.ndarray:
        """Main flow synchronizing function."""
        sync_flow = np.zeros(
            (2, self.flows.shape[2], self.flows.shape[3]), dtype=np.float64
        )
        if self.has_covariances:
            self.accum_cov = np.zeros(
                (2, 2, self.flows.shape[2], self.flows.shape[3]), dtype=np.float64
            )

        # Low flow refresh rate
        occur_index = max(self.time_to_flow[start_time] - 1, 0)
        flow_x = self.flows[occur_index, 0, :, :]
        flow_y = self.flows[occur_index, 1, :, :]
        total_time = (
            start_time / 1e3 - (self.flows_time[occur_index] + self.time_offset) / 1e6
        )
        flows_time = (
            self.flows_time[occur_index + 1] / 1e6 - self.flows_time[occur_index] / 1e6
        )
        delta_time = total_time / flows_time

        # High flow refresh rate
        self._init_grids()
        self._init_masks()
        self._init_flows(flow_x, flow_y, delta_time)
        # Accumulate flow displacements
        while (self.flows_time[occur_index] + self.time_offset) / 1e6 < end_time / 1e3:
            flow_x = self.flows[occur_index, 0, :, :]
            flow_y = self.flows[occur_index, 1, :, :]
            total_time = (
                end_time / 1e3 - (self.flows_time[occur_index] + self.time_offset) / 1e6
            )
            flows_time = (
                self.flows_time[occur_index + 1] / 1e6
                - self.flows_time[occur_index] / 1e6
            )
            delta_time = min(total_time / flows_time, 1.0)
            self._propagate_flow(flow_x, flow_y, delta_time)

            if self.has_covariances:
                cov = self.covariances[occur_index]
                warped_cov = self._remap_covariance(cov, self.grid_x, self.grid_y)
                scaled_cov = (delta_time**2) * warped_cov
                self.accum_cov += scaled_cov

            occur_index += 1

        # Compute flow shift
        shift_x = self.grid_x - self.grid_x_init
        shift_y = self.grid_y - self.grid_y_init
        shift_x[~self.mask_x] = 0
        shift_y[~self.mask_y] = 0
        sync_flow[0, :, :] = shift_x
        sync_flow[1, :, :] = shift_y

        # Inverse flow
        if inverse:
            sync_flow = inverse_flow(sync_flow)

        if self.has_covariances:
            return sync_flow, self.accum_cov
        else:
            return sync_flow, None
