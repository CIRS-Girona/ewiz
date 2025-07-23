import numpy as np

from ewiz.data.converters import ConvertStonefish


if __name__ == "__main__":
    stonefish_dir = "/home/jad/datasets/estonefish-scenes/bags/dynamic/estonefish_dynamic_varying_rocky_down-looking.bag"
    out_dir = "/home/jad/datasets/estonefish-scenes/ewiz_format/dynamic/estonefish_dynamic_varying_rocky_down-looking"
    stonefish_converter = ConvertStonefish(
        stonefish_dir, out_dir, sensor_size=(260, 346), clip_size=True
    )
    stonefish_converter.convert()
