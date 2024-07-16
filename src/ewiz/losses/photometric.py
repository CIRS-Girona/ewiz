import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .base import LossBase

from typing import Any, Dict, List, Tuple, Callable, Union


class Photometric(LossBase):
    """Photometric loss.
    """
    name = None
    required_keys = None

    def __init__(
        self,
        direction: str = "minimize",
        store_history: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(direction, store_history, *args, **kwargs)
