"""An example script to run the motion compensation algorithm.
"""
import numpy as np

# Import reader
from ewiz.data.readers import ReaderBase

# Import transforms
from ewiz.data.transforms import Compose
from ewiz.data.transforms.events import EventsCenterCrop

# Import motion compensation
from ewiz.losses import LossMotionCompensation
from ewiz.algorithms.mc import MotionCompensationPyramidal


if __name__ == "__main__":
    # Initialize data reader
    data_reader = ReaderBase(
        data_dir="/home/jad/Documents/datasets/mvsec/ewiz/indoor_flying1",
        clip_mode="time"
    )
    events, _, _ = data_reader[20000:20500]

    # Apply transforms
    ewiz_transforms = [EventsCenterCrop(in_size=(260, 346), out_size=(256, 256))]
    ewiz_compose = Compose(transforms=ewiz_transforms, use_tonic=True)
    events = ewiz_compose(events)

    # Initialize loss function
    mc_loss = LossMotionCompensation(
        image_size=(256, 256),
        losses=[
            "multifocal_normalized_gradient_magnitude",
            "multifocal_normalized_image_variance",
            "smoothness"
        ],
        weights=[1.0, 1.0, 0.0001],
        warper_type="dense",
        imager_type="bilinear",
        batch_size=1,
        direction="minimize",
        store_history=False,
        precision="64",
        device="cuda",
        render_visuals=True
    )

    # Initialize optimizer
    mc_pyramidal = MotionCompensationPyramidal(
        image_size=(256, 256),
        loss=mc_loss,
        optimizer="BFGS",
        flow_inits=(-20, 20),
        scales=(1, 5)
    )

    # Optimize
    mc_pyramidal.optimize(events=events)
