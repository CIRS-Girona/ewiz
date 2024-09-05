import numpy as np
import h5py
import hdf5plugin

from ewiz.data.loaders import LoaderEvents, LoaderTime

from ewiz.renderers import WindowManager
from ewiz.renderers.visualizers import VisualizerEvents, VisualizerGray


if __name__ == "__main__":
    data_dir = "/home/jad/datasets/prophesee/ewiz_converted/test1"
    data_loader = LoaderTime(
        data_dir=data_dir,
        data_stride=50,
        reader_mode="base"
    )
    window_manager = WindowManager(
        image_size=(720, 1280),
        grid_size=(2, 2),
        window_names=["Events"],
        refresh_rate=1
    )
    events_visualizer = VisualizerEvents(image_size=(720, 1280))

    # Iterate over data
    for events, gray_images, gray_time in data_loader:
        events_image = events_visualizer.render_image(events=events)
        # Render frame
        window_manager.render(events_image)
