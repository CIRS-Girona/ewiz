import numpy as np
import h5py
import hdf5plugin

from ewiz.data.loaders import LoaderEvents, LoaderTime

from ewiz.renderers import WindowManager
from ewiz.renderers.visualizers import VisualizerEvents, VisualizerGray, VisualizerFlow


if __name__ == "__main__":
    data_dir = "/home/jad/datasets/carla/ewiz/260p/static/01_static_town2_backward_clear-sunset"
    data_loader = LoaderTime(
        data_dir=data_dir,
        data_stride=40,
        reader_mode="flow",
        inverse_flow=True
    )
    window_manager = WindowManager(
        image_size=(260, 346),
        grid_size=(3, 2),
        window_names=["Events", "Grayscale", "Flow"],
        refresh_rate=0
    )
    events_visualizer = VisualizerEvents(image_size=(260, 346))
    gray_visualizer = VisualizerGray(image_size=(260, 346))
    flow_visualizer = VisualizerFlow(image_size=(260, 346))

    # Iterate over data
    for events, gray_images, gray_time, flow in data_loader:
        events_image = events_visualizer.render_image(events=events)
        gray_image = gray_visualizer.render_image(gray_image=gray_images[0])
        flow_image = flow_visualizer.render_image(flow=flow)
        # Render frame
        window_manager.render(events_image, gray_image, flow_image)
