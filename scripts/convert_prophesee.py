import numpy as np

from ewiz.data.converters import ConvertProphesee


if __name__ == "__main__":
    prophesee_dir = "/home/jad/datasets/prophesee/hdf5/test4.hdf5"
    out_dir = "/home/jad/datasets/prophesee/ewiz/test4"
    prophesee_converter = ConvertProphesee(prophesee_dir, out_dir)
    prophesee_converter.convert()
