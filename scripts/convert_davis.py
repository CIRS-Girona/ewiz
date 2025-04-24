import numpy as np

from ewiz.data.converters import ConvertDAVIS


if __name__ == "__main__":
    bag_dir = "/home/jad/datasets/real/bag/amphora1.bag"
    out_dir = "/home/jad/datasets/real/ewiz_format/amphora1"
    bag_converter = ConvertDAVIS(bag_dir, out_dir)
    bag_converter.convert()
