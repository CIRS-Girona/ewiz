import numpy as np

from ewiz.data.converters import ConvertBag


if __name__ == "__main__":
    bag_dir = "/home/jad/datasets/real/bag/marine_snow_random1.bag"
    out_dir = "/home/jad/datasets/real/ewiz_format/marine_snow_random1"
    bag_converter = ConvertBag(bag_dir, out_dir)
    bag_converter.convert()
