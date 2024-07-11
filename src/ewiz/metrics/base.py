import numpy as np
import json

from typing import Any, Dict, List, Tuple, Callable, Union


class MetricsBase():
    """Base metrics class.
    """
    def __init__(self, store_history: bool = False) -> None:
        self.store_history = store_history
        self._init_metrics()
        self._init_history()

    def _init_metrics(self) -> None:
        """Initializes metrics.
        """
        self.count = 0
        self.metrics = {}
        self.sum_metrics = {}

    def _init_history(self) -> None:
        """Initializes history.
        """
        self.history = {"metrics": []}

    def enable_history(self) -> None:
        """Enables history.
        """
        self.store_history = True
        print("Metrics history enabled.")

    def disable_history(self) -> None:
        """Disables history.
        """
        self.store_history = False
        print("Metrics history disabled.")

    def add_history(func: Callable) -> Callable:
        """Saves history.
        """
        def wrapper(self, *args, **kwargs) -> Dict[str, Any]:
            metrics = func(self, *args, **kwargs)
            if self.store_history:
                self.history["metrics"].append(metrics)
            return metrics
        return wrapper

    def get_history(self) -> Dict[str, Any]:
        """Gets history.
        """
        return self.history.copy()

    def save_history(self, file_path: str = None) -> None:
        """Saves history in '.txt' format.
        """
        if file_path is None:
            file_path = "history.txt"
        history = self.get_history()
        with open(file_path, "w") as file:
            file.write(json.dumps(history))
        print("Metrics history saved successfully in:", file_path)

    def get_average(self) -> Dict[str, float]:
        """Returns average metrics.
        """
        return {
            "a" + metric: value/self.count for metric, value in self.sum_metrics.items()
        }

    def save_average(self, file_path: str = None) -> None:
        """Saves average metrics in '.txt' format.
        """
        if file_path is None:
            file_path = "metrics.txt"
        average_metrics = self.get_average()
        with open(file_path, "w") as file:
            file.write(json.dumps(average_metrics))
        print("Metrics average saved successfully in:", file_path)

    @add_history
    def calculate(self, *args, **kwargs) -> Dict[str, float]:
        """Calculates metrics.
        """
        raise NotImplementedError

    add_history = staticmethod(add_history)
