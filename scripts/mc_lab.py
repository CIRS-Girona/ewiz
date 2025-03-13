# General imports
import numpy as np

# Import reader
from ewiz.data.readers import ReaderFlow

# Import transforms
from ewiz.data.transforms import Compose
from ewiz.data.transforms.events import EventsCenterCrop
from ewiz.data.transforms.flow import FlowCenterCrop

# Import motion compensation
from ewiz.losses import LossMotionCompensation
from ewiz.algorithms.mc import MotionCompensationPyramidal

# Import metrics
from ewiz.metrics.endpoint import EndpointError


if __name__ == "__main__":
    # Read data
    data_reader = ReaderFlow(
        data_dir="/home/jad/datasets/mvsec/ewiz/indoor_flying1.bak", clip_mode="time"
    )
    raw_events, grayscale_images, grayscale_timestamps, gt_flow = data_reader[
        20000:20500
    ]

    # Below is the ground truth flow we are using for validation...
    gt_flow = gt_flow / ((raw_events[-1, 2] - raw_events[0, 2]) * 1e-6)

    # Below is the predicted flow...
    pred_flow = None

    # Apply transforms
    events_transforms = Compose(
        [EventsCenterCrop(in_size=(260, 346), out_size=(256, 256))], use_tonic=True
    )
    flow_transforms = Compose(
        [FlowCenterCrop(out_size=(256, 256))],
    )
    raw_events = events_transforms(raw_events)
    gt_flow = flow_transforms(gt_flow)

    # Initialize loss function
    mc_loss = LossMotionCompensation(
        image_size=(256, 256),
        losses=[
            "multifocal_normalized_gradient_magnitude",
            "multifocal_normalized_image_variance",
            "smoothness",
        ],
        weights=[1.0, 1.0, 0.001],
        warper_type="dense",
        imager_type="bilinear",
        batch_size=1,
        direction="minimize",
        store_history=False,
        precision="64",
        device="cuda",
        render_visuals=True,
    )

    # Initialize optimizer
    mc_pyramidal = MotionCompensationPyramidal(
        image_size=(256, 256),
        loss=mc_loss,
        optimizer="BFGS",
        flow_inits=(-200, 200),
        scales=(1, 5),
    )

    # Run motion compensation
    _ = mc_pyramidal.optimize(raw_events)

    # Initialize endpoint error
    endpoint_error = EndpointError()
    pred_flow = mc_pyramidal.get_dense_flow(4)
    endpoint_error.calculate(pred_flow, gt_flow)
    print(endpoint_error.metrics["epe"])
    pred_flow = mc_pyramidal.get_dense_flow(1)
    endpoint_error.calculate(pred_flow, gt_flow)
    print(endpoint_error.metrics["epe"])
