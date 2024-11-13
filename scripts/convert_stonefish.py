import numpy as np

from ewiz.data.converters import ConvertStonefish


if __name__ == "__main__":
    stonefish_dir = "/home/jad/Documents/datasets/stonefish/bags/stonefish_recording_08-11-2024_10-05-53.bag"
    out_dir = "/home/jad/Documents/datasets/stonefish/ewiz/stonefish_recording_08-11-2024_10-05-53"
    stonefish_converter = ConvertStonefish(stonefish_dir, out_dir, sensor_size=(400, 800))
    stonefish_converter.convert()
