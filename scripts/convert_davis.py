import numpy as np

from ewiz.data.converters import ConvertDAVIS


if __name__ == "__main__":
    bag_dir = "/home/jad/datasets/papers/pool_datasets/bags/good_aperture/rov02.bag"
    out_dir = "/home/jad/datasets/papers/pool_datasets/ewiz_format/rov02"
    bag_converter = ConvertDAVIS(bag_dir, out_dir)
    bag_converter.convert()
