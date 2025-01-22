import numpy as np

from ewiz.data.converters import ConvertStonefish


if __name__ == "__main__":
    stonefish_dir = "/home/jad/Documents/datasets/stonefish/bags/"
    out_dir = "/home/jad/Documents/datasets/stonefish/ewiz/"
    stonefish_converter = ConvertStonefish(stonefish_dir, out_dir, sensor_size=(720, 1280))
    stonefish_converter.convert()
