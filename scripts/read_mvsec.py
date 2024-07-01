import numpy as np
import h5py
import hdf5plugin

from ewiz.data.readers import ReaderBase


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1"
    clip_mode = "images"
    data = ReaderBase(data_dir=data_dir, clip_mode=clip_mode)
    events, gray_images, gray_time = data[100]
    print(events, gray_images, gray_time)
