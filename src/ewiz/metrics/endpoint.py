import numpy as np

from .base import MetricsBase

from typing import Any, Dict, List, Tuple, Callable, Union


class EndpointError(MetricsBase):
    """Endpoint error metric.
    """
    def __init__(
        self,
        outlier_thresh: Tuple[int] = (1, 2, 3, 4, 5, 10, 20),
        store_history: bool = False
    ) -> None:
        super().__init__(store_history)
        self.outlier_thresh = outlier_thresh

    def _init_metrics(self) -> None:
        """Initializes metrics.
        """
        self.count = 0
        self.metrics = {"ae": 0.0, "epe": 0.0}
        self.sum_metrics = {"ae": 0.0, "epe": 0.0}
        for thresh in self.outlier_thresh:
            self.metrics.update({f"{thresh}pe": 0.0})
            self.sum_metrics.update({f"{thresh}pe": 0.0})

    @staticmethod
    def get_flow_mask(flow: np.ndarray) -> np.ndarray:
        """Returns flow mask over valid points.
        """
        return np.logical_and(
            np.logical_and(~np.isinf(flow[:, [0], :, :]), ~np.isinf(flow[:, [1], :, :])),
            np.logical_and(np.abs(flow[:, [0], :, :]) > 0, np.abs(flow[:, [1], :, :]) > 0)
        )

    @staticmethod
    def get_events_mask(encoded_events: np.ndarray) -> np.ndarray:
        """Returns mask over pixels where events occurred.
        """
        events_mask = np.sum(np.sum(encoded_events, axis=1, keepdims=True), axis=4)
        return events_mask

    @staticmethod
    def convert_flow_velocity_to_displacement(
        flow: np.ndarray, delta_time: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Converts flow velocity to displacement.
        """
        if type(delta_time) == np.ndarray:
            assert len(delta_time) == len(flow) or len(delta_time) == 1, (
                "Delta time should have the same size as the flow's batch size."
            )
            delta_time = np.reshape(delta_time, (len(delta_time), 1, 1, 1))
            flow = flow*delta_time
        else:
            flow = flow*delta_time
        return flow

    def calculate_endpoint_error(
        self, pred_flow: np.ndarray, gt_flow: np.ndarray, num_pixels: int
    ) -> None:
        """Calculates endpoint error.
        """
        endpoint_error = np.linalg.norm(gt_flow - pred_flow, axis=1)
        self.metrics["epe"] = np.mean(np.sum(endpoint_error, axis=(1, 2))/num_pixels)
        self.sum_metrics["epe"] += self.metrics["epe"]
        for thresh in self.outlier_thresh:
            self.metrics[f"{thresh}pe"] = np.mean(np.sum(endpoint_error > thresh, axis=(1, 2))/num_pixels)
            self.sum_metrics[f"{thresh}pe"] += self.metrics[f"{thresh}pe"]

    def calculate_angular_error(
        self, pred_flow: np.ndarray, gt_flow: np.ndarray, num_pixels: int
    ) -> None:
        """Calculates angular error.
        """
        raise NotImplementedError
