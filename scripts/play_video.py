import numpy as np
import h5py
import hdf5plugin

from ewiz.renderers.videos import VideoRendererBase


if __name__ == "__main__":
    data_dir = "/home/jad/datasets/carla/ewiz/test1"
    video_renderer = VideoRendererBase(data_dir, 10)
    video_renderer.play()
