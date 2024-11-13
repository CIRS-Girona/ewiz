import numpy as np
import h5py
import hdf5plugin

from ewiz.renderers.videos import VideoRendererBase


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/stonefish/ewiz/stonefish_recording_08-11-2024_10-05-53"
    save_dir = "/home/jad/Documents/datasets/stonefish/rendered_data/stonefish_recording_08-11-2024_10-05-53"
    video_renderer = VideoRendererBase(data_dir, 80, save_images=True, save_dir=save_dir)
    video_renderer.play()
