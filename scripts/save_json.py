"""A script showing how to run motion compensation.
"""
from ewiz.core.utils import save_json, read_json
from ewiz.losses import LossMotionCompensation


if __name__ == "__main__":
    mc_loss = LossMotionCompensation(
        image_size=(256, 256),
        losses=["gradient_magnitude"],
        weights=[1.0]
    )

    data = {
        "image_size": (256, 256),
        "options": {
            "resolution": (1920, 1080),
            "upscale_mode": "bilinear"
        }
    }
    file_path = "/home/jad/datasets/out.json"
    save_json(data, file_path)

    # Print output
    out = read_json(file_path)
    print(out)
