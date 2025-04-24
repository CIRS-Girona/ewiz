import numpy as np

from ewiz.data.converters import ConvertStonefish


if __name__ == "__main__":
    stonefish_dir = "/home/jad/datasets/papers/hops/bags_format/net_test.bag"
    out_dir = "/home/jad/datasets/papers/hops/ewiz_format/net_test"
    stonefish_converter = ConvertStonefish(
        stonefish_dir, out_dir, sensor_size=(512, 512)
    )
    stonefish_converter.convert()
