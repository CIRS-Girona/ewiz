import torch
import numpy as np
import json

from typing import Any, Dict, List, Tuple, Callable, Union


class LossBase():
    """Base loss class.
    """
    required_keys: List[str] = []

    def __init__(
        self,
        direction: str = "minimize",
        store_history: bool = False,
        *args,
        **kwargs
    ) -> None:
        self.direction = direction
        self.store_history = store_history
        self._init_history()

    def _init_history(self) -> None:
        """Initializes history.
        """
        self.history = {"loss": []}

    def enable_history(self) -> None:
        """Enables history.
        """
        self.store_history = True
        print("Loss history enabled.")

    def disable_history(self) -> None:
        """Disables history.
        """
        self.store_history = False
        print("Loss history disabled.")

    def add_history(func: Callable) -> Callable:
        """Saves history.
        """
        def wrapper(self, *args, **kwargs) -> Dict[str, Any]:
            loss = func(self, *args, **kwargs)
            if self.store_history:
                self.history["loss"].append(self.get_item(loss))
            return loss
        return wrapper

    def get_history(self) -> Dict[str, Any]:
        """Gets history.
        """
        return self.history.copy()

    def get_item(self, loss: Union[float, torch.Tensor]) -> float:
        """Gets loss item.
        """
        if isinstance(loss, torch.Tensor):
            return loss.item()
        return loss

    def catch_key_error(func: Callable) -> Callable:
        """Catches key error.
        """
        def wrapper(self, *args, **kwargs) -> None:
            try:
                return func(self, *args, **kwargs)
            except TypeError as error:
                print("The loss function requires the following keys:")
                print(self.required_keys)
                raise error
        return wrapper

    @add_history
    @catch_key_error
    def calculate(self, *args, **kwargs) -> Union[float, torch.Tensor]:
        """Calculates loss.
        """
        raise NotImplementedError

    catch_key_error = staticmethod(catch_key_error)
    add_history = staticmethod(add_history)
