import os
import h5py
import hdf5plugin
import numpy as np

import roslib
import rosbag

from tqdm import tqdm

from .base import ConvertBase
from ..writers import WriterEvents, WriterGray

from typing import Any, Dict, List, Tuple, Callable, Union


# TODO: Add in separate utilities file
def ros_message_to_cv_image(message: Any) -> np.ndarray:
    """Converts ROS message to OpenCV image.
    """
    image = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.width, -1
    )
    return image


class ConvertBag(ConvertBase):
    """DAVIS bag to eWiz data converter.
    """
    def __init__(
        self,
        data_dir: str,
        out_dir: str,
        sensor_size: Tuple[int] = (260, 346),
        events_topic: str = "/davis/left/events",
        gray_topic: str = "/davis/left/image_raw"
    ) -> None:
        super().__init__(data_dir, out_dir)
        self.sensor_size = sensor_size
        self.events_topic = events_topic
        self.gray_topic = gray_topic
        self._init_writers()
        self._get_min_time()

    def _init_events(self) -> None:
        """Initializes events bag file.
        """
        for dir, subdirs, files in os.walk(self.data_dir):
            for file in files:
                if "_data.bag" in file:
                    self.bag_path = os.path.join(dir, file)
                    self.bag_file = rosbag.Bag(self.bag_path)
                    break

    # TODO: Check removal
    def _init_images(self) -> None:
        """Initializes images file.
        """
        pass

    def _init_writers(self) -> None:
        """Initializes writers.
        """
        self.events_writer = WriterEvents(self.out_dir)
        self.gray_writer = WriterGray(self.out_dir)

    def _get_min_time(self) -> None:
        """Gets minimum timestamp.
        """
        # Get minimum events timestamp
        events_messages = self.bag_file.read_messages(topics=self.events_topic)
        for events_data in events_messages:
            _, message, _ = events_data
            events = self._extract_events(message)
            self.min_time = int(events[0, 2])
            break
        # Get minimum grayscale timestamp
        gray_messages = self.bag_file.read_messages(topics=self.gray_topic)
        for gray_data in gray_messages:
            _, message, _ = gray_data
            _, gray_min_time = self._extract_gray(message)
            gray_min_time = int(gray_min_time)
            break
        # Get global minimum timestamp
        if gray_min_time < self.min_time:
            self.min_time = gray_min_time

    def convert(self) -> None:
        """Converts bag data.
        """
        print("# === Converting Bag Data === #")
        print("# === Converting Events === #")
        events_messages = self.bag_file.read_messages(topics=self.events_topic)
        for events_data in tqdm(events_messages):
            _, message, _ = events_data
            events = self._extract_events(message)
            events = events.astype(np.int64)
            events[:, 2] -= self.min_time
            self.events_writer.write(events=events)
        # Map time to events
        self.events_writer.map_time_to_events()

        print("# === Converting Grayscale Images === #")
        gray_messages = self.bag_file.read_messages(topics=self.gray_topic)
        for gray_data in tqdm(gray_messages):
            _, message, _ = gray_data
            gray_image, time = self._extract_gray(message)
            time -= self.min_time
            self.gray_writer.write(gray_image, time)
        # Map time to grayscale images
        self.gray_writer.map_time_to_gray()
        self.gray_writer.map_gray_to_events()

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
