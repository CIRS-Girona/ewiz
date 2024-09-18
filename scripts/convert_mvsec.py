import numpy as np

from ewiz.data.converters import ConvertMVSEC


if __name__ == "__main__":
    mvsec_dir = "/home/jad/Documents/datasets/mvsec/hdf5/indoor_flying1"
    out_dir = "/home/jad/Documents/datasets/mvsec/ewiz/indoor_flying1"
    mvsec_converter = ConvertMVSEC(mvsec_dir, out_dir)
    mvsec_converter.convert()
