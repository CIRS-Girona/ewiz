import numpy as np

from ewiz.data.converters import ConvertStonefish


if __name__ == "__main__":
    stonefish_dir = "/home/jad/datasets/stonefish/bags/low_resolution_sequence.bag"
    out_dir = "/home/jad/Documents/datasets/stonefish/ewiz/low_resolution_sequence"
    stonefish_converter = ConvertStonefish(stonefish_dir, out_dir, sensor_size=(128, 128))
    stonefish_converter.convert()
