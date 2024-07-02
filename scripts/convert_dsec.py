import numpy as np

from ewiz.data.converters import ConvertDSEC


if __name__ == "__main__":
    dsec_dir = "/home/jad/Documents/datasets/mvsec/indoor_flying1"
    out_dir = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1"
    dsec_converter = ConvertDSEC(dsec_dir, out_dir)
    dsec_converter.convert()
