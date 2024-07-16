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
        pred_u, pred_v = pred_flow[:, 0, ...], pred_flow[:, 1, ...]
        gt_u, gt_v = gt_flow[:, 0, ...], gt_flow[:, 1, ...]
        self.metrics["ae"] = np.mean(np.sum(
            np.arccos((1.0 + pred_u*gt_u + pred_v*gt_v)/
            (np.sqrt(1.0 + pred_u*pred_u + pred_v*pred_v)*np.sqrt(1.0 + gt_u*gt_u + gt_v*gt_v))),
            axis=(1, 2)
        )/num_pixels)
        self.sum_metrics["ae"] += self.metrics["ae"]

    @MetricsBase.add_history
    def calculate(
        self,
        pred_flow: np.ndarray,
        gt_flow: np.ndarray,
        encoded_events: np.ndarray = None,
        delta_time: Union[float, np.ndarray] = None,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """Calculates endpoint error.
        """
        if len(pred_flow.shape) == 3:
            pred_flow = pred_flow[None, ...]
        if len(gt_flow.shape) == 3:
            gt_flow = gt_flow[None, ...]

        flow_mask = self.get_flow_mask(gt_flow)
        if encoded_events is not None:
            events_mask = self.get_events_mask(encoded_events)
            total_mask = np.logical_and(flow_mask, events_mask)
        else:
            total_mask = flow_mask

        pred_flow = pred_flow*total_mask
        gt_flow = gt_flow*total_mask
        num_pixels = np.sum(total_mask, axis=(1, 2, 3)) + 1e-5

        if delta_time is not None:
            pred_flow = self.convert_flow_velocity_to_displacement(pred_flow, delta_time)
            gt_flow = self.convert_flow_velocity_to_displacement(gt_flow, delta_time)
        self.calculate_endpoint_error(pred_flow, gt_flow, num_pixels)
        self.calculate_angular_error(pred_flow, gt_flow, num_pixels)
        self.count += 1
        return self.metrics
