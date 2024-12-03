import numpy as np
import h5py
import hdf5plugin

from ewiz.renderers.videos import VideoRendererBase


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/carla/ewiz/01_static_town2_backward_clear-sunset"
    save_dir = "/home/jad/Documents/datasets/carla/rendered_data/01_static_town2_backward_clear-sunset"
    video_renderer = VideoRendererBase(data_dir, 10, save_images=False, save_dir=save_dir)
    video_renderer.play()
