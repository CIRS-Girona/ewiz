import numpy as np
import h5py
import hdf5plugin

from ewiz.renderers.videos import VideoRendererBase


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/stonefish/ewiz/stonefish_recording_16-10-2024_15-48-50"
    video_renderer = VideoRendererBase(data_dir, 100)
    video_renderer.play()
