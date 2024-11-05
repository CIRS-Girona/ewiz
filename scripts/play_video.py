import numpy as np
import h5py
import hdf5plugin

from ewiz.renderers.videos import VideoRendererBase


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/carla/ewiz/town10_forward_clear-noon"
    save_dir = "/home/jad/Documents/datasets/carla/rendered_data/town10_forward_clear-noon"
    video_renderer = VideoRendererBase(data_dir, 80, save_images=True, save_dir=save_dir)
    video_renderer.play()
