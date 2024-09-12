import torch

from .base import LossBase
from . import loss_functions

from typing import Any, Dict, List, Tuple, Callable, Union


# TODO: Add batch size algorithm
class LossHybrid(LossBase):
    """Hybrid loss function.
    """
    def __init__(
        self,
        losses: List[str],
        weights: List[float],
        batch_size: int = 1,
        direction: str = "minimize",
        store_history: bool = False,
        precision: str = "64",
        device: str = "cuda",
        *args,
        **kwargs
    ) -> None:
        super().__init__(direction, store_history, *args, **kwargs)
        self.losses = losses
        self.weights = weights
        self.batch_size = batch_size
        self.precision = precision
        self.device = device
        self.loss_functions = {name: {
            "function": loss_functions[name](
                direction=direction,
                store_history=store_history,
                precision=precision,
                device=device,
                *args,
                **kwargs
            ),
            "weight": self.weights[i]
        } for i, name in enumerate(self.losses)}

        # Setup required keys
        self.required_keys = []
        for name in self.loss_functions.keys():
            self.required_keys.extend(self.loss_functions[name]["function"].required_keys)

    def update_weights(self, losses: List[str], weights: List[float]) -> None:
        """Updates loss weights.
        """
        for i, name in enumerate(losses):
            self.loss_functions[name]["weight"] = weights[i]
        print("All loss weights updated.")

    # TODO: Run init history from outside
    def _init_history(self) -> None:
        """Initializes history.
        """
        self.history = {"loss": []}
        for name in self.loss_functions.keys():
            self.loss_functions[name]["function"]._init_history()

    # TODO: Check printing
    def enable_history(self) -> None:
        """Enables history.
        """
        self.store_history = True
        for name in self.loss_functions.keys():
            self.loss_functions[name]["function"].store_history = True
        print("Loss history enabled.")

    def disable_history(self) -> None:
        """Disables history.
        """
        self.store_history = False
        for name in self.loss_functions.keys():
            self.loss_functions[name]["function"].store_history = False
        print("Loss history disabled.")

    # TODO: Can be expanded to more than loss only
    def get_history(self) -> Dict[str, Any]:
        """Gets history.
        """
        history = self.history.copy()
        for name in self.loss_functions.keys():
            history.update({name: self.loss_functions[name]["function"].get_history()["loss"]})

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(self, *args, **kwargs) -> Union[float, torch.Tensor]:
        """Calculates loss.
        """
        # TODO: Remove hard-coded CUDA
        comb_loss = torch.zeros(self.batch_size).cuda()
        for name in self.loss_functions.keys():
            loss = self.loss_functions[name]["weight"]*self.loss_functions[name]["function"].calculate(*args, **kwargs)
            comb_loss += loss
        return comb_loss
