import numpy as np
import h5py
import hdf5plugin


if __name__ == "__main__":
    events_path = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1/events.hdf5"
    events_file = h5py.File(events_path, "r")
    print(events_file["events"]["time"][50])
