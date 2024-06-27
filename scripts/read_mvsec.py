import numpy as np
import h5py
import hdf5plugin

from ewiz.data.readers import ReaderBase


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1"
    data_reader = ReaderBase(data_dir=data_dir)
    print(data_reader[0])
