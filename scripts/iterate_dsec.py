import numpy as np
import h5py
import hdf5plugin

from ewiz.data.iterators import IteratorEvents, IteratorTime

from ewiz.renderers import WindowManager
from ewiz.renderers.visualizers import VisualizerEvents, VisualizerGray


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/converted/dsec/interlaken_00_a.bak0"
    data_loader = IteratorEvents(
        data_dir=data_dir,
        data_stride=5e5,
        reader_mode="base"
    )
    window_manager = WindowManager(
        image_size=(480, 640),
        grid_size=(2, 2),
        window_names=["Events", "Grayscale"],
        refresh_rate=1
    )
    events_visualizer = VisualizerEvents(image_size=(480, 640))
    gray_visualizer = VisualizerGray(image_size=(480, 640))

    # Iterate over data
    for events, gray_images, gray_time in data_loader:
        events_image = events_visualizer.render_image(events=events)
        gray_image = gray_visualizer.render_image(gray_image=gray_images[0])
        # Render frame
        window_manager.render(events_image, gray_image)
