.. _writers:

Writers
=======

Overview
--------
The **Writers** module is a foundational component of the **eWiz** library, designed
for efficient storage and organization of event-based, grayscale, and optical flow data.
It ensures optimized data management by employing compression and time mapping,
enabling seamless integration into data generation pipelines.

### Key Features
- **Optimized Storage**: Uses the **BLOSC** compression algorithm to minimize
  dataset size while maintaining fast access.
- **Efficient Memory Usage**: Writes data directly to disk to prevent memory overload.
- **Time Mapping**: Automatically establishes relationships between different data
  modalities. This includes:
  - Events Indices ↔ Timestamps
  - Events Indices ↔ Grayscale Images Indices
  - Grayscale Images Indices ↔ Timestamps
  - Optical Flow Indices ↔ Timestamps
  - Optical Flow Indices ↔ Events Indices
- **eWiz Format**: Saves data in HDF5 format with a unified structure for
  compatibility and scalability.

.. tip::

   The **Writers** module is ideal for large-scale data generation workflows,
   particularly when dealing with event-based data and multi-modal sensor streams.

---

Initialization and Arguments
----------------------------
The **Writers** module provides three primary classes: `WriterEvents`, `WriterGray`,
and `WriterFlow`. These classes share a common argument structure:

### Arguments
- **out_dir** (*str*):
  The output directory where data files will be saved. The directory structure
  is automatically created if it does not already exist. Each data type (events,
  grayscale, optical flow) is stored in a separate HDF5 file, and camera properties
  are saved in a JSON file.

.. note::

   Ensure that the `out_dir` path has write permissions. The module will create
   any necessary subdirectories automatically.

---

Key Classes
-----------
### WriterEvents
Handles the storage of event-based data and mapping of timestamps to event
indices.
- **Methods**:
  - ``write(events: np.ndarray)``: Saves event data to disk in compressed format.
  - ``map_time_to_events()``: Establishes a mapping between timestamps and event indices.

### WriterGray
Saves grayscale image data along with timestamps and maps their relationships to
event indices.
- **Methods**:
  - ``write(gray_image: np.ndarray, time: float)``: Writes a grayscale image and
  its associated timestamp.
  - ``map_time_to_gray()``: Maps timestamps to grayscale images.
  - ``map_gray_to_events()``: Maps grayscale images to event indices.

### WriterFlow
Stores optical flow data with timestamps and maps their relationships to event indices.
- **Methods**:
  - ``write(flow: np.ndarray, time: float)``: Writes optical flow data with its
  associated timestamp.
  - ``map_time_to_flow()``: Maps timestamps to optical flow data.
  - ``map_flow_to_events()``: Maps optical flow data to event indices.

---

Accessing the Writers
---------------------
The **Writers** module is straightforward to use. Once initialized, the writers
automatically handle file creation, data compression, and mapping.

### Example Usage
```python
from ewiz.data.writers import WriterEvents, WriterGray, WriterFlow

# Define output directory
out_path = "/path/to/output"

# Initialize writers
events_writer = WriterEvents(out_dir=out_path)
gray_writer = WriterGray(out_dir=out_path)
flow_writer = WriterFlow(out_dir=out_path)

# Example data
events_data = [...]
gray_data = [...]
flow_data = [...]
timestamp = 0.05

# Save data
events_writer.write(events=events_data)
gray_writer.write(gray_image=gray_data, time=timestamp)
flow_writer.write(flow=flow_data, time=timestamp)

# Generate mappings
events_writer.map_time_to_events()
gray_writer.map_time_to_gray()
gray_writer.map_gray_to_events()
flow_writer.map_time_to_flow()
flow_writer.map_flow_to_events()
```
