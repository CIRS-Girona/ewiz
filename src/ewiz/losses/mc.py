import torch
import numpy as np

from ewiz.algorithms.warpers import warper_functions
from ewiz.algorithms.imagers import imager_functions

from ewiz.losses import LossBase, LossHybrid
from ewiz.algorithms.warpers import WarperBase
from ewiz.algorithms.imagers import ImagerBase

from typing import Any, Dict, List, Tuple, Callable, Union


class LossMotionCompensation(LossBase):
    """Motion compensation loss.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        losses: List[str],
        weights: List[float],
        warper_type: str = "dense",
        imager_type: str = "bilinear",
        batch_size: int = 1,
        direction: str = "minimize",
        store_history: bool = False,
        precision: str = "64",
        device: str = "cuda",
        *args,
        **kwargs
    ) -> None:
        super().__init__(direction, store_history)
        self.loss_func: LossBase = None

        # General properties
        self.image_size = image_size
        self.losses = losses
        self.weights = weights
        self.warper_type = warper_type
        self.imager_type = imager_type
        self.batch_size = batch_size
        self.direction = direction
        self.store_history = store_history
        self.precision = precision
        self.device = device
        self._init_functions()

    def _init_functions(self) -> None:
        """Initializes functions.
        """
        self.loss_func = LossHybrid(
            self.losses, self.weights, self.batch_size, self.direction,
            self.store_history, self.precision, self.device
        )
        self.warper_func: WarperBase = warper_functions[self.warper_type](self.image_size)
        # TODO: Look into image padding argument
        self.imager_func: ImagerBase = imager_functions[self.imager_type](self.image_size)

    # TODO: Add render option
    def _parse_mc_args(
        self,
        events: torch.Tensor,
        dense_flow: torch.Tensor,
        patch_flow: torch.Tensor = None,
        render: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Parses motion compensation arguments.
        """
        # TODO: Set to default
        required_keys = self.loss_func.required_keys
        mc_args = {"omit_bounds": True}

        if "ie" in required_keys:
            ie = self.imager_func.generate_iwe(events)
            mc_args.update({"ie": ie})
        if "iwe" in required_keys:
            warped_events = self.warper_func.warp(
                events=events, flow=dense_flow, direction="start"
            )
            iwe = self.imager_func.generate_iwe(warped_events)
            mc_args.update({"iwe": iwe})
        if "start_iwe" in required_keys:
            start_events = self.warper_func.warp(
                events=events, flow=dense_flow, direction="start"
            )
            start_iwe = self.imager_func.generate_iwe(start_events)
            mc_args.update({"start_iwe": start_iwe})
        if "mid_iwe" in required_keys:
            mid_events = self.warper_func.warp(
                events=events, flow=dense_flow, direction="mid"
            )
            mid_iwe = self.imager_func.generate_iwe(mid_events)
            mc_args.update({"mid_iwe": mid_iwe})
        if "end_iwe" in required_keys:
            end_events = self.warper_func.warp(
                events=events, flow=dense_flow, direction="end"
            )
            end_iwe = self.imager_func.generate_iwe(end_events)
            mc_args.update({"end_iwe": end_iwe})
        # TODO: Change logic type
        if "flow" in required_keys:
            if patch_flow is None:
                mc_args.update({"flow": dense_flow})
            else:
                mc_args.update({"flow": patch_flow})
        return mc_args, None

    @LossBase.add_history
    @LossBase.catch_key_error
    def calculate(
        self,
        events: torch.Tensor,
        dense_flow: torch.Tensor,
        patch_flow: torch.Tensor = None,
        render: bool = False
    ) -> torch.Tensor:
        """Calculates loss function.
        """
        # TODO: Change logic
        mc_args = self._parse_mc_args(events, dense_flow, patch_flow, render)
        mc_args = mc_args[0]
        loss_val = self.loss_func.calculate(**mc_args)
        return loss_val
