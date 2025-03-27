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

# Visualizer imports
from ewiz.renderers.window import WindowManager
from ewiz.renderers.visualizers import VisualizerFlow


if __name__ == "__main__":
    # Read data
    data_reader = ReaderFlow(
        data_dir="/home/jad/datasets/mvsec/ewiz/indoor_flying1.bak", clip_mode="time"
    )
    raw_events, grayscale_images, grayscale_timestamps, gt_flow = data_reader[
        20000:20500
    ]

    # Below is the ground truth flow we are using for validation...
    gt_flow = gt_flow

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
    # Currently, there is an issue with conversion. For a 0.5 s sequence here, we use 0.05.
    pred_flow = mc_pyramidal.get_dense_flow(4) * 0.05
    endpoint_error.calculate(pred_flow, gt_flow)
    print(endpoint_error.metrics["epe"])

    flow_visualizer = VisualizerFlow(image_size=(256, 256), vis_type="arrows")
    window_manager = WindowManager(
        image_size=(256, 256),
        grid_size=(2, 2),
        window_names=["Ground Truth Flow", "Predicted Flow"],
        refresh_rate=0,
        save_images=False,
        save_dir=None,
    )
    gt_image = flow_visualizer.render_image(gt_flow, scale=2.0)
    pred_image = flow_visualizer.render_image(pred_flow, scale=2.0)
    window_manager.render(gt_image, pred_image)
