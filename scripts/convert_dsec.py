import numpy as np

from ewiz.data.converters import ConvertDSEC


if __name__ == "__main__":
    dsec_dir = "/home/jad/Documents/datasets/dsec/interlaken_00_a"
    out_dir = "/home/jad/Documents/datasets/converted/dsec/interlaken_00_a"
    dsec_converter = ConvertDSEC(dsec_dir, out_dir)
    dsec_converter.convert()
