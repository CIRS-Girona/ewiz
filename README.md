<h1 align="center">
    eWiz
</h1>

<h4 align="center">
    All-in-one Event-based Data Manipulation
</h4>

<div align="center">

<!-- Add badges here -->
[Introduction](#introduction) •
[Getting Started](#getting-started) •
[Usage](#usage) •
[Citation](#citation) •
[Acknowledgements](#acknowledgements) •
[Related Projects](#related-projects)

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
    pass
```
