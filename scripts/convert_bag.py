import numpy as np

from ewiz.data.converters import ConvertBag


if __name__ == "__main__":
    bag_dir = "/home/jad/Documents/datasets/mvsec/bags/indoor_flying1"
    out_dir = "/home/jad/Documents/datasets/converted/mvsec/indoor_flying1"
    bag_converter = ConvertBag(bag_dir, out_dir)
    bag_converter.convert()
