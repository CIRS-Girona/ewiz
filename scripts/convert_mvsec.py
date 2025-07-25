import numpy as np

from ewiz.data.converters import ConvertMVSEC


if __name__ == "__main__":
    mvsec_dir = "/home/jad/datasets/mvsec/mvsec/indoor_flying1"
    out_dir = "/home/jad/datasets/mvsec/ewiz_format/indoor_flying1"
    mvsec_converter = ConvertMVSEC(mvsec_dir, out_dir)
    mvsec_converter.convert()
