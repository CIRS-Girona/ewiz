import os
import h5py
import hdf5plugin
import numpy as np

import cv2
import roslib
import rosbag

from cv_bridge import CvBridge
from tqdm import tqdm

from .base import ConvertBase
from ..writers import WriterEvents, WriterGray, WriterFlow

from typing import Any, Dict, List, Tuple, Callable, Union


# TODO: Add in separate utilities file
def ros_message_to_cv_image(message: Any) -> np.ndarray:
    """Converts ROS message to OpenCV image.
    """
    image = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.width, -1
    )
    return image


# TODO: Requires testing
class ConvertStonefish(ConvertBase):
    """Stonefish bag to eWiz data converter.
    """
    def __init__(
        self,
        data_dir: str,
        out_dir: str,
        sensor_size: Tuple[int] = (260, 346),
        events_topic: str = "/bluerov2/sensors/events",
        rgb_topic: str = "/bluerov2/sensors/rgb/image_color",
        flow_topic: str = "/bluerov2/sensors/flow/image_raw"
    ) -> None:
        super().__init__(data_dir, out_dir, sensor_size)
        self.sensor_size = sensor_size
        self.events_topic = events_topic
        self.rgb_topic = rgb_topic
        self.flow_topic = flow_topic

        # OpenCV bridge
        self.bridge = CvBridge()

        self._init_events()
        self._init_images()
        self._init_flow()
        self._init_writers()
        self._get_min_time()

    def ros_flow_to_cv(self, message: Any) -> np.ndarray:
        """Converts ROS message to OpenCV image.
        """
        flow = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        return flow

    def _init_events(self) -> None:
        """Initializes events bag file.
        """
        self.bag_file = rosbag.Bag(self.data_dir)

        # TODO: Check removal
    def _init_images(self) -> None:
        """Initializes images file.
        """
        pass

    def _init_flow(self) -> None:
        """Initializes optical flow.
        """
        self.previous_time = None

    def _init_writers(self) -> None:
        """Initializes writers.
        """
        self.events_writer = WriterEvents(self.out_dir)
        self.gray_writer = WriterGray(self.out_dir)
        self.flow_writer = WriterFlow(self.out_dir)

    def _get_min_time(self) -> None:
        """Gets minimum timestamp.
        """
        # Get minimum events timestamp
        events_messages = self.bag_file.read_messages(topics=self.events_topic)
        for events_data in events_messages:
            _, message, _ = events_data
            # TODO: Check why empty arrays
            if len(message.events) > 1:
                events = self._extract_events(message)
                self.min_time = int(events[0, 2])
                break
        # Get minimum grayscale timestamp
        rgb_messages = self.bag_file.read_messages(topics=self.rgb_topic)
        for rgb_data in rgb_messages:
            _, message, _ = rgb_data
            _, rgb_min_time = self._extract_rgb_to_gray(message)
            rgb_min_time = int(rgb_min_time)
            break
        # Get minimum flow timestamp
        flow_messages = self.bag_file.read_messages(topics=self.flow_topic)
        for flow_data in flow_messages:
            _, message, _ = flow_data
            _, flow_min_time = self._extract_flow(message)
            flow_min_time = int(flow_min_time)
            self.previous_time = None
            break
        # Get global minimum timestamp
        for time in (self.min_time, rgb_min_time, flow_min_time):
            if time < self.min_time:
                self.min_time = time

    def convert(self) -> None:
        """Converts bag data.
        """
        print("# === Converting Bag Data === #")
        print("# === Converting Events === #")
        events_messages = self.bag_file.read_messages(topics=self.events_topic)
        for events_data in tqdm(events_messages):
            _, message, _ = events_data
            # TODO: Check why empty arrays
            if len(message.events) >= 1:
                events = self._extract_events(message)
                events = events.astype(np.int64)
                events[:, 2] -= self.min_time
                # TODO: Should be done in Stonefish, sort the events based on timestamps
                events = events[events[:, 2].argsort()]
                self.events_writer.write(events=events)
        # Map time to events
        self.events_writer.map_time_to_events()

        print("# === Converting Grayscale Images === #")
        rgb_messages = self.bag_file.read_messages(topics=self.rgb_topic)
        for rgb_data in tqdm(rgb_messages):
            _, message, _ = rgb_data
            gray_image, time = self._extract_rgb_to_gray(message)
            time -= self.min_time
            self.gray_writer.write(gray_image, time)
        # Map time to grayscale images
        self.gray_writer.map_time_to_gray()
        self.gray_writer.map_gray_to_events()

        print("# === Converting Optical Flow === #")
        flow_messages = self.bag_file.read_messages(topics=self.flow_topic)
        for flow_data in tqdm(flow_messages):
            _, message, _ = flow_data
            flow_image, time = self._extract_flow(message)
            time -= self.min_time
            if self.previous_time is not None:
                self.flow_writer.write(flow_image, time)
        # Map time to flow
        self.flow_writer.map_time_to_flow()
        self.flow_writer.map_flow_to_events()

    # TODO: Timestamps require modifications
    # ======================= #
    # --- Data Extractors --- #
    # ======================= #
    def _extract_events(self, message: Any) -> np.ndarray:
        """Extracts events from ROS message.
        """
        events = message.events
        pos = np.array([(event.x, event.y) for event in events], dtype=np.float64)
        time = np.array([(event.ts.secs + event.ts.nsecs*1e-9)*1e6 for event in events], dtype=np.float64)
        # TODO: Optimize polarity extraction
        pol = np.where([event.polarity for event in events], 1, 0).astype(np.float64)
        events = np.empty((len(events), 4), dtype=np.float64)
        events[:, :2] = pos
        # TODO: Fix orientation in Stonefish
        events[:, 1] = (self.sensor_size[0] - 1) - events[:, 1]
        events[:, 2] = time
        events[:, 3] = pol
        return events

    def _extract_gray(self, message: Any) -> Tuple[np.ndarray, int]:
        """Extracts grayscale data from ROS message.
        """
        time = getattr(message, "header").stamp
        time = int((time.secs + time.nsecs*1e-9)*1e6)
        # TODO: Add missing pixels check
        gray_image = ros_message_to_cv_image(message)
        gray_image = gray_image[..., 0]
        return gray_image, time

    def _extract_rgb_to_gray(self, message: Any) -> Tuple[np.ndarray, int]:
        """Extracts RGB data from ROS message.
        """
        time = getattr(message, "header").stamp
        time = int((time.secs + time.nsecs*1e-9)*1e6)
        # TODO: Add missing pixels check
        rgb_image = ros_message_to_cv_image(message)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        return gray_image, time

    def _extract_flow(self, message: Any) -> Tuple[np.ndarray, int]:
        """Extracts flow data from ROS message.
        """
        time = getattr(message, "header").stamp
        time = int((time.secs + time.nsecs*1e-9)*1e6)
        flow_image = self.ros_flow_to_cv(message)
        flow_image = np.transpose(flow_image, axes=(2, 0, 1))
        if self.previous_time is not None:
            delta_time = time - self.previous_time
            flow_image = flow_image*(delta_time*1e-6)
        # Save previous time
        self.previous_time = time
        return flow_image, time
