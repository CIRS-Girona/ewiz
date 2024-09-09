import torch
import numpy as np

import torchvision.transforms.functional as F

from torchvision.transforms import InterpolationMode

from typing import Any, Dict, List, Tuple, Callable, Union


# TODO: Check flow sign
def convert_patch_to_dense_flow(
    patch_flow: torch.Tensor, grid_size: Tuple[int, int], image_size: Tuple[int, int]
) -> torch.Tensor:
    """Converts from patch to dense flow.
    """
    inter_mode = InterpolationMode.BILINEAR
    patch_flow = -patch_flow.reshape((1, 2) + grid_size)[0]
    flow = F.resize(patch_flow, image_size, inter_mode, antialias=True)
    return flow
