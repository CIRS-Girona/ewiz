import torch
import numpy as np

from torchvision.transforms.functional import gaussian_blur

from typing import Any, Dict, List, Tuple, Callable, Union


class ImagerBase():
    """Base imager class.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        image_padding: Tuple[int, int] = (0, 0)
    ) -> None:
        self._update_image_props(image_size, image_padding)

    def _update_image_props(
        self,
        image_size: Tuple[int, int],
        image_padding: Tuple[int, int] = (0, 0)
    ) -> None:
        """Updates image properties.
        """
        self.image_size = image_size
        self.image_padding = image_padding
        self.image_size = tuple(
            int(size + 2*padding)
            for size, padding in zip(self.image_size, self.image_padding)
        )

    def generate_iwe(
        self,
        events: torch.Tensor,
        sigma_blur: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Main IWE generation function.
        """
        iwe: torch.Tensor = self.iwe_func(events=events, *args, **kwargs)
        if sigma_blur > 0.0:
            if len(iwe.shape) == 2:
                iwe = iwe[None, ...]
            elif len(iwe.shape) == 3:
                iwe = iwe[:, None, ...]
            iwe = gaussian_blur(iwe, kernel_size=3, sigma=sigma_blur)
            iwe = torch.squeeze(iwe)
        return iwe

    def generate_mask(
        self,
        events: torch.Tensor
    ) -> torch.Tensor:
        """Generates mask of events.
        """
        mask = self.generate_iwe(events=events, sigma_blur=0.0) != 0
        mask = mask[..., None, :, :]
        return mask
