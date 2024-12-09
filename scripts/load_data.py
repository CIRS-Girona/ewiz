import numpy as np
import h5py
import hdf5plugin

from ewiz.data.iterators import IteratorEvents


if __name__ == "__main__":
    data_dir = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1"
    data_loader = IteratorEvents(data_dir=data_dir)
