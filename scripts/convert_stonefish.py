import numpy as np

from ewiz.data.converters import ConvertStonefish


if __name__ == "__main__":
    stonefish_dir = "/home/jad/datasets/stonefish/bags/test1.bag"
    out_dir = "/home/jad/datasets/stonefish/ewiz/test1"
    stonefish_converter = ConvertStonefish(stonefish_dir, out_dir, sensor_size=(400, 800))
    stonefish_converter.convert()
