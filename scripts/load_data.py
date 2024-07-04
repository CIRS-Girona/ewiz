import numpy as np
import h5py
import hdf5plugin

from ewiz.data.loaders import LoaderEvents


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1"
    data_loader = LoaderEvents(data_dir=data_dir)
