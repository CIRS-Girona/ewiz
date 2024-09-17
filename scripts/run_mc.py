"""A script showing how to run motion compensation.
"""
from ewiz.losses import LossMotionCompensation


if __name__ == "__main__":
    mc_loss = LossMotionCompensation(
        image_size=(256, 256),
        losses=["gradient_magnitude"],
        weights=[1.0]
    )
