<h1 align="center">
    eWiz
</h1>

<h4 align="center">
    All-in-one Event-based Data Manipulatio
</h4>

<div align="center">

<!-- Add badges here -->
[Introduction](#introduction) •
[Getting Started](#getting-started) •
[Usage](#usage) •
[Data Format](#data-format) •
[Citation](#citation) •
[Acknowledgements](#acknowledgements) •
[Related Projects](#related-projects) •
[Licensing](#licensing)

</div>

## Introduction
eWiz is a Python library designed for efficient manipulation, visualization, and processing of event-based data. Whether you're working with event-based sensors like DAVIS, generating synthetic datasets, or building spiking neural networks (SNNs), eWiz provides the essential tools and utilities to streamline your workflow.

Event-based data, characterized by its high temporal resolution and asynchronous nature, poses unique challenges for traditional data processing tools. eWiz addresses these challenges with an optimized, modular design that supports seamless storage, retrieval, and processing of event streams, grayscale images, and optical flow data.

### Key Features
* **Multi-modal Data:** Support for multi-modal data, such as events, grayscale images, and optical flow data.
* **Optimized Storage:** Utilizes the HDF5 format coupled with BLOSC compression to minimize disk usage while maintaining fast access.
* **Dataset Support:** Supports popular datasets in the literature (e.g., MVSEC, DSEC).

### Why eWiz?
1. **Efficiency:** Designed to handle large event-based datasets without overwhelming memory or disk requirements.
2. **Flexibility:** Works with real-world sensors (e.g., DAVIS), popular datasets (e.g., MVSEC), and synthetically generated datasets.
3. **Ease of Use:** Intuitive APIs make event-based data processing straightforward.
4. **Research-ready:** Perfect for applications in event-based vision, spiking neural networks, and neuromorphic computing.

## Getting Started
### Installation
The eWiz library is mainly compatible with Python 3.8 and requires PyTorch 2.0.1 and above to run. eWiz makes use of PyTorch and CUDA to run its optimization algorithms, more specifically motion compensation.

> **Note:** It is recommended to use eWiz with Ubuntu` 20.04 in a separate virtual environment with Python 3.8 installed (using Anaconda for example).
> Check that your graphic card drivers are correctly installed on your system as eWiz requires CUDA 11.8 and above.

You can install eWiz directly using PyPi along all its dependencies. However, you need to have PyTorch 2.0.1 or above already installed in your Python environment. Activate your Anaconda environment and start by installing your preferred version of PyTorch and Torchvision. Don't forget to install the CUDA supported version.

An example PyTorch installation command can be found below:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
This should install PyTorch along-side CUDA on your system. Afterwards, you can directly install eWiz with PyPi inside your environment:
```bash
pip install ewiz
```
You now have eWiz installed, along all its dependencies. Check out our [usage](#usage) section to learn how to run some example scripts.

## Usage
The eWiz library provides powerful tools for working with event-based data, including writing, reading, and loading multi-modal datasets. To help you get started, we’ve included detailed examples in the `scripts` folder. These scripts demonstrate how to use the core components of the library in practical scenarios.

### Examples
The following examples are included in the `scripts` folder:
1. **Data Writers:** Write event-based data, grayscale images, and optical flow using the eWiz format.
2. **Data Readers:** Read and slice data, based on event indices, timestamps, or grayscale indices.
3. **Data Loaders:** Use PyTorch-style data loaders to preprocess and load the multi-modal data sequentially, by striding over event indices, timestamps, or grayscale indices.
4. **Data Converters:** Convert open-source datasets, such as MVSEC, DSEC to the eWiz format.

#### Data Writers
Use the data writers module to save event-based, grayscale, and optical flow data in the eWiz format.
```python
from ewiz.data.writers import WriterEvents, WriterGray, WriterFlow

# Initialize data writers
out_dir = "/path/to/output"
events_writer = WriterEvents(out_dir=out_dir)
gray_writer = WriterGray(out_dir=out_dir)
flow_writer = WriterFlow(out_dir=out_dir)

# Example data
events_data = [...]
gray_data = [...]
flow_data = [...]
timestamp = 0.05

# Save data
events_writer.write(events=events_data)
gray_writer.write(gray_image=gray_data, time=timestamp)
flow_writer.write(flow=flow_data, time=timestamp)

# Generate time mappings
events_writer.map_time_to_events()
gray_writer.map_time_to_gray()
gray_writer.map_gray_to_events()
flow_writer.map_time_to_flow()
flow_writer.map_flow_to_events()
```

#### Data Readers
The data readers allow you to efficiently read and slice datasets using event indices, timestamps, or grayscale indices.
```python
from ewiz.data.readers import ReaderFlow

# Initialize data reader
data_dir = "/path/to/dataset"
reader = ReaderFlow(data_dir=data_dir, clip_mode="time")

# Clip data with timestamps
start, end = 100, 140
events, gray_images, gray_time, flow = reader[start:end]
```
#### Data Loaders
The data loaders module allows for easy sequential data loading. In this example, we load the multi-modal data, and stride over it with a time interval of 20 ms.
```python
from ewiz.data.loaders import LoaderTime

# Initialize data loader
data_dir = "/path/to/dataset"
loader = LoaderTime(data_dir=data_dir, data_stride=20)

# Iterate over data
for data in loader:
    events, gray_images, gray_time, flow = data
```

#### Data Converters
Convert datasets like MVSEC, and DSEC to the eWiz format for seamless integration. An example for the MVSEC dataset is shown below:
```python
from ewiz.data.converters import ConvertMVSEC

mvsec_dir = "/path/to/MVSEC/dataset"
out_dir = "/path/to/save/converted/dataset"
mvsec_converter = ConvertMVSEC(mvsec_dir, out_dir)
mvsec_converter.convert()
```

More examples can be found in the `scripts` folder. For example, `play_video.py` showcases how you can visualize datasets converted to the eWiz format. The `render_events.py` shows how to use the visualizers modules to visualize event-based data, grayscale images, and optical flow data side by side.

## Data Format
eWiz makes use of a compressed form of the HDF5 file format to save all data. Moreover, to avoid the use of time consuming sorted search algorithms, we use saved look-up arrays that map timestamps to data properties. Currently, eWiz saves the following data:
* **Events:** Event-based data, which includes x and y-coordinates, timestamps (in &micro;s), and polarities.
* **Grayscale Images:** The grayscale images captured by the camera in case of a hybrid sensor.
* **Optical Flow:** Ground truth optical flow data, which can also be inverted if desired.

### General Structure
eWiz uses the BLOSC compression format for improved memory management and efficient data storage. All data is saved in a single folder containing multiple HDF5 files and one JSON file. Each HDF5 file contains a data type, whether events, grayscale images, or optical flow. In summary, the general data format is as follows:
* A properties file, called `props.json`, it includes general properties about the dataset. Currently, we only save the `sensor_size` but we aim to add more properties.
* A compilation of HDF5 files, containing the different components of the dataset. Currently, eWiz supports saving events, grayscale images, and optical flow data. They are saved in `events.hdf5`, `gray.hdf5`, and `flow.hdf5` respectively.

> **Note:** Due to compression, you might not be able to read the data with a simple HDF5 viewer.

All data files contain the `time_offset` HDF5 group. It is composed of a single value (in &micro;s) which indicates the starting timestamp of each sequence. This starting timestamp may not be the same for all data types, as the grayscale images for example have a different sampling rate than that of the events. For all data types to be synchronized, it is just a matter of adding the `time_offset` value to the timestamps of the desired data type.

### Events
The `events.hdf5` file contains the following data:
* The `events` group, containing the events of the sequence, separated into 4 distinct datasets:
  * The **x-coordinates**, inside the `x` dataset, of data type `uint16`, which contains the x-coordinates of the events in a 1D array, with values ranging from 0 to the width size of the sensor.
  * The **y-coordinates**, inside the `y` dataset, of data type `uint16`, which contains the y-coordinates of the events in a 1D array, with values ranging from 0 to the height size of the sensor.
  * The **timestamps**, inside the `time` dataset, of data type `int64`, which contains the timestamps (in &micro;s) of the events in a 1D array.
  * The **polarities**, inside the `polarity` dataset, of data type `bool`, which contains the polarities of the events in a 1D array, with values ranging from 0 to the height size of the sensor.
* The **time to event indices** mapping, inside the `time_to_events` dataset, which contains the time mappings in a 1D array where the indices are the timestamps (in ms) and the corresponding values are the event indices.

### Grayscale Images
The `gray.hdf5` file contains the following data:
* The **grayscale images**, inside the `gray_images` dataset, of data type `uint8`, which contains the grayscale images in an array of *(number of images x height x width)*.
* The **timestamps**, inside the `time` dataset, of data type `int64`, which contains the timestamps (in &micro;s) of the images in a 1D array.
* The **grayscale to event indices** mapping, inside the `gray_to_events` dataset, which contains the grayscale mappings in a 1D array where the indices are the grayscale indices and the corresponding values are the event indices.
* The **time to grayscale indices** mapping, inside the `time_to_gray` dataset, which contains the time mappings in a 1D array where the indices are the timestamps (in ms) and the corresponding values are the grayscale indices.

### Optical Flow
The `flow.hdf5` file contains the following data:
* The **optical flow**, inside the `flows` dataset, of data type `int64`, which contains the optical flows in an array of *(number of flows x 2 x height x width)*.
* The **timestamps**, inside the `time` dataset, of data type `int64`, which contains the timestamps (in &micro;s) of the flows in a 1D array.
* The **flow to event indices** mapping, inside the `flow_to_events` dataset, which contains the flow mappings in a 1D array where the indices are the flow indices and the corresponding values are the event indices.
* The **time to flow indices** mapping, inside the `time_to_flow` dataset, which contains the time mappings in a 1D array where the indices are the timestamps (in ms) and the corresponding values are the flow indices.

> **Note:** When you are using the library you do not have to know the data format in detail. The included data readers and loaders automatically read the data and synchronize it internally.

## Citation
This repository is related to the paper below. If you find this repository please do not hesitate to give it a star :star2:!
```bibtex
@misc{mansourCarla2024,
    title={eCARLA-scenes: A synthetically generated dataset for event-based optical flow prediction},
    author={Jad Mansour and Hayat Rajani and Rafael Garcia and Nuno Gracias},
    year={2024},
    eprint={2412.09209},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.09209},
}
```
Also, if you have any questions do not hesitate to contact me at [jad.mansour@udg.edu](mailto:jad.mansour@udg.edu).

## Acknowledgements
Jad Mansour was supported by the Joan Oró Grant no. 2024 FI-2 00762. The study was also supported in part by the SIREC project, funded by the Ministerio de Ciencia e Innovación, Gobierno de España under agreement no. PID2020-116736RB-IOO.

This work was also inspired by the following repositories:
* [Secrets of Event-based Optical Flow (T-PAMI 2024, ECCV 2022)](https://github.com/tub-rip/event_based_optical_flow)

## Related Projects
Related projects to this work are:

* [eCARLA-Scenes: A Synthetic Event-based Optical Flow Dataset for Autonomous Field Vehicles](https://github.com/CIRS-Girona/ecarla-scenes)

## Licensing
This project is licensed under the **GNU General Public License v3.0**.

You are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. However, any derived work must also be open-source and distributed under the same license.

For more details, see the [LICENSE](LICENSE) file in the repository or visit the official GPL-3.0 license page: [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

> 2024, Jad Mansour, University of Girona. All rights reserved.
