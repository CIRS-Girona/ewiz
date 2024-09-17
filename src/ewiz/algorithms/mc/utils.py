import torch
import numpy as np
import skimage.transform

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

def update_fine_to_coarse_flow(patch_flows: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """Updates from fine to coarse flow.
    """
    scales = patch_flows.keys()
    coarse = min(scales)
    fine = max(scales)
    final_flows = {fine: patch_flows[fine]}
    for i in range(fine, coarse - 1, -1):
        final_flows[i - 1] = skimage.transform.pyramid_reduce(patch_flows[i], channel_axis=0)
    return final_flows

def parse_mc_args(
    self,
    events: torch.Tensor,
    dense_flow: torch.Tensor,
    patch_flow: torch.Tensor = None
) -> Dict[str, torch.Tensor]:
    """Parses motion compensation arguments.
    """
    raise NotImplementedError
