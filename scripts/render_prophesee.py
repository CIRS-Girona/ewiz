import numpy as np
import h5py
import hdf5plugin

from ewiz.data.readers import ReaderBase

from ewiz.renderers import WindowManager
from ewiz.renderers.visualizers import VisualizerEvents, VisualizerFlow


if __name__ == "__main__":
    data_dir = "/home/jad/datasets/prophesee/ewiz_converted/test1"
    clip_mode = "time"
    data = ReaderBase(data_dir=data_dir, clip_mode=clip_mode)
    events, gray_images, gray_time = data[0:100]

    window_manager = WindowManager(
        image_size=(720, 1280),
        grid_size=(2, 2),
        window_names=["Events"]
    )
    events_visualizer = VisualizerEvents(image_size=(720, 1280))
    events_image = events_visualizer.render_image(events=events)

    # Render frame
    window_manager.render(events_image)
