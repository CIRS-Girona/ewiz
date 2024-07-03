import numpy as np
import h5py
import hdf5plugin

from ewiz.data.readers import ReaderFlow

from ewiz.renderers import WindowManager
from ewiz.renderers.visualizers import VisualizerEvents


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1"
    clip_mode = "images"
    data = ReaderFlow(data_dir=data_dir, clip_mode=clip_mode)
    events, gray_images, gray_time, flow = data[1000:1010]

    window_manager = WindowManager(
        image_size=(260, 346),
        grid_size=(2, 2),
        window_names=["Events"]
    )
    events_visualizer = VisualizerEvents(image_size=(260, 346))
    image = events_visualizer.render_image(events=events)

    # Render frame
    window_manager.render(image)
