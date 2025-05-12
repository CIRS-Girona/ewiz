import numpy as np
import cv2

from ewiz.data.iterators import IteratorTime
from ewiz.renderers import WindowManager
from ewiz.renderers.visualizers import VisualizerEvents, VisualizerGray, VisualizerFlow

from .base import VideoRendererBase

from typing import Any, Dict, List, Tuple, Callable, Union

class VideoRendererFlow(VideoRendererBase):
    """Dataset video renderer. Allows you to read and render a sequence of events and their 
    corrsponding ground truth optical flow in the eWiz format.
    """

    def __init__(
        self,
        data_dir: str,
        data_stride: int = None,
        data_range: Tuple[int, int] = None,
        flow_vis_type: str = "colors",
        flow_scale: int = 100,
        save_images: bool = False,
        save_dir: str = None,
    ) -> None:
        """
        Args:
            data_dir (str): Dataset directory, should be using the eWiz format.
            data_stride (int, optional): Data stride. Defaults to None.
            data_range (Tuple[int, int], optional): Data range. Defaults to None.
            flow_vis_type (str, optional): Flow visualization method. Can be set to
                either "colors" or "arrows". Defaults to "colors". 
            flow_scale (int, optional): Flow scale. Scales flow displacements during
                visualization process if flow_vis_type = "arrows". Defaults to 100.
            save_images (bool, optional): Saves generated images. Defaults to False.
            save_dir (str, optional): Images save directory. Defaults to None.

        Examples:
            To render a dataset in the eWiz format, you just have to:

            >>> video_renderer = VideoRendererFlow(data_dir="/path/to/dataset")
            >>> video_renderer.play()
        """
        self.flow_vis_type = flow_vis_type
        self.flow_scale = flow_scale
        super().__init__(data_dir, data_stride, data_range, save_images, save_dir)
        

    def _init_video(self) -> None:
        """Initializes video."""
        # Initialize main modules
        self.data_loader = IteratorTime(
            data_dir=self.data_dir,
            data_stride=self.data_stride,
            data_range=self.data_range,
            reader_mode="flow",
        )
        # TODO: Change refresh rate
        self.window_manager = WindowManager(
            image_size=self.props["sensor_size"],
            grid_size=(1, 2),
            window_names=["Events", "Flow"],
            refresh_rate=1,
            window_size=(720, 840),
            save_images=self.save_images,
            save_dir=self.save_dir,
        )

        # Initialize renderers
        self.events_visualizer = VisualizerEvents(self.props["sensor_size"])
        self.gray_visualizer = VisualizerGray(self.props["sensor_size"])
        self.flow_visualizer = VisualizerFlow(self.props["sensor_size"], 
                                              vis_type=self.flow_vis_type)

    def play(self, *args, **kwargs) -> None:
        """Plays video."""
        for events, gray_images, gray_time, flow in self.data_loader:
            events_image = self.events_visualizer.render_image(events=events)
            if gray_images is not None:
                events_mask = self.events_visualizer.render_mask(events=events)
                events_mask = np.repeat(np.expand_dims(events_mask, axis=2), 3, axis=2)
                gray_image = self.gray_visualizer.render_image(
                    gray_image=gray_images[0]
                )
                gray_image = np.repeat(gray_image, 3, axis=2)
                rendered_image = np.where(events_mask, events_image, gray_image)
            else:
                rendered_image = events_image
            if flow is not None:
                flow_image = self.flow_visualizer.render_image(flow=flow, scale=self.flow_scale)
            # TODO: Modify window manager format
            self.window_manager.render(
                rendered_image,
                flow_image,
                texts=[self._create_time_text(events)]*2,
                position=(15, 30),
            )
        self.window_manager.create_mp4()