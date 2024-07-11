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
