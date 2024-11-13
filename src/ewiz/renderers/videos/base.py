import numpy as np
import cv2

from ewiz.core.props import import_props
from ewiz.data.loaders import LoaderTime
from ewiz.renderers import WindowManager
from ewiz.renderers.visualizers import VisualizerEvents, VisualizerGray, VisualizerFlow

from typing import Any, Dict, List, Tuple, Callable, Union


# TODO: Continue code
class VideoRendererBase():
    """Base video renderer.
    """
    def __init__(
        self,
        data_dir: str,
        data_stride: int = None,
        data_range: Tuple[int, int] = None,
        save_images: bool = False,
        save_dir: str = None
    ) -> None:
        self.data_dir = data_dir
        self.data_stride = data_stride
        self.data_range = data_range
        self.props = import_props(self.data_dir)
        self.save_images = save_images
        self.save_dir = save_dir
        self._init_video()

    def _init_video(self) -> None:
        """Initializes video.
        """
        # Initialize main modules
        self.data_loader = LoaderTime(
            data_dir=self.data_dir,
            data_stride=self.data_stride,
            data_range=self.data_range
        )
        # TODO: Change refresh rate
        self.window_manager = WindowManager(
            image_size=self.props["sensor_size"],
            grid_size=(1, 1),
            window_names=["eWiz"],
            refresh_rate=1,
            window_size=(720, 1080),
            save_images=self.save_images,
            save_dir=self.save_dir
        )

        # Initialize renderers
        self.events_visualizer = VisualizerEvents(self.props["sensor_size"])
        self.gray_visualizer = VisualizerGray(self.props["sensor_size"])

    def _create_time_text(self, events: np.ndarray) -> str:
        """Creates time text.
        """
        time_val = np.mean(events[:, 2])/1e6
        time_text = "t = " + "%.2f" % time_val + " s"
        return time_text

    def play(self, *args, **kwargs) -> None:
        """Plays video.
        """
        for events, gray_images, gray_time in self.data_loader:
            events_image = self.events_visualizer.render_image(events=events)
            if gray_images is not None:
                events_mask = self.events_visualizer.render_mask(events=events)
                events_mask = np.repeat(np.expand_dims(events_mask, axis=2), 3, axis=2)
                gray_image = self.gray_visualizer.render_image(gray_image=gray_images[0])
                gray_image = np.repeat(gray_image, 3, axis=2)
                rendered_image = np.where(events_mask, events_image, gray_image)
            else:
                rendered_image = events_image

            # TODO: Modify window manager format
            self.window_manager.render(
                rendered_image, texts=[self._create_time_text(events)], position=(15, 30)
            )
        self.window_manager.create_mp4()
