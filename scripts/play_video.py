import numpy as np
import h5py
import hdf5plugin

from ewiz.renderers.videos import VideoRendererBase


if __name__ == "__main__":
    data_dir = "/home/jad/datasets/stonefish/ewiz/stonefish_recording_17-10-2024_10-26-10"
    video_renderer = VideoRendererBase(data_dir, 80)
    video_renderer.play()
