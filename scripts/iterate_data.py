import numpy as np
import h5py
import hdf5plugin

from ewiz.data.loaders import LoaderEvents, LoaderTime

from ewiz.renderers import WindowManager
from ewiz.renderers.visualizers import VisualizerEvents, VisualizerFlow


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1.bak2"
    data_loader = LoaderTime(
        data_dir=data_dir,
        reader_mode="flow"
    )
    window_manager = WindowManager(
        image_size=(260, 346),
        grid_size=(2, 2),
        window_names=["Events", "Flow"],
        refresh_rate=2
    )
    events_visualizer = VisualizerEvents(image_size=(260, 346))
    flow_visualizer = VisualizerFlow(image_size=(260, 346))

    # Iterate over data
    for events, gray_images, gray_time, flow in data_loader:
        events_image = events_visualizer.render_image(events=events)
        flow_image = flow_visualizer.render_image(flow=flow)
        # Render frame
        window_manager.render(events_image, flow_image)
