import numpy as np
import h5py
import hdf5plugin

from ewiz.renderers.videos import VideoRendererBase


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/stonefish/data/val/estonefish_static_forward_reef"
    save_dir = None
    video_renderer = VideoRendererBase(data_dir, 10, save_images=False, save_dir=save_dir)
    video_renderer.play()
