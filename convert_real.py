import os

from ewiz.data.converters import ConvertDAVIS


if __name__ == "__main__":
    in_dir = "/home/jad/datasets/papers/pool_datasets/bags/good_aperture"
    out_dir = "/home/jad/datasets/real/ewiz_format"

    for file_name in os.listdir(in_dir):
        if file_name.endswith(".bag"):
            bag_path = os.path.join(in_dir, file_name)
            print(bag_path)
            base_name = os.path.splitext(file_name)[0]
            out_folder = os.path.join(out_dir, base_name)
            os.makedirs(out_folder, exist_ok=True)
            bag_converter = ConvertDAVIS(bag_path, out_folder)
            bag_converter.convert()
