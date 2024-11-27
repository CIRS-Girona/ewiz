Data Format
===========
eWiz makes use of a compressed form of the `HDF5`_ file format to save all data.
Moreover, to avoid the use of time consuming sorted search algorithms, we use
saved look-up arrays that map timestamps to different data properties. Currently,
eWiz saves the following data:

* **Events:** Event-based data, which includes x and y-coordinates, timestamps,
  and polarities.
* **Grayscale Images:** The grayscale images captured by the event-based camera
  in case the sensor is hybrid, such as the likes of the `DAVIS`_ sensor.

  .. TODO: Add references
* **Optical Flow:** Ground truth optical flow data. Optical flow data format depends
  on the dataset that have been converted.

Data Structure
--------------
eWiz uses the `BLOSC`_ compression format for improved memory management and
efficient data storage. All data is saved in a single folder containing multiple
multiple `HDF5`_ files and one *JSON* file. Each `HDF5`_ file contains a data
type, whether events, grayscale images, or optical flow. In summary, the General
data format is as follows:

* A properties file, called ``props.json``, it includes general properties about
  the dataset. Currently, we only save the ``sensor size`` but we aim to add more
  properties, the latter can be used to automatically obtain the image size during
  visualization.
* A compilation of `HDF5`_ files, called ``data.hdf5``, containing the different
  components of the dataset. Currently, eWiz supports saving events, grayscale
  images, and optical flow data.

.. note::
  The `HDF5`_ datasets are compressed using the `BLOSC`_ format and as a result,
  you might not be able to read the data with a simple HDF5 viewer.

In the sections below, we explain the data format in details for each data type.

All data files contain the ``time_offset`` HDF5 group. It is composed of a single
value which indicates the starting timestamp of the sequence. This starting
timestamp may not be the same for all data types, as the grayscale images for example
have a different sampling rate than that of the events. For all data types to be
synchronized, it is just a matter of adding the ``time_offset`` value to the
corresponding timestamps. The ``time_offset`` value is saved in *μs*.

Events
``````
eWiz's events data is saved in the ``events.hdf5`` data file. It consists of the
following HDF5 groups:

* The ``events`` group contains the main event-based data, divided as follows:

  * **The x-coordinates**, present in the ``x`` data group. Contains the x-coordinates
    of the events in a 1D array with values ranging from 0 to the width of the sensor.
    The data type is ``uint16``.
  * **The y-coordinates**, present in the ``y`` data group. Contains the y-coordinates
    of the events in a 1D array with values ranging from 0 to the height of the sensor.
    The data type is ``uint16``.
  * **The events' timestamps**, present in the ``time`` data group. Contains the events'
    timestamps in *μs* in a 1D array. The data type is ``int64``.
  * **The events' polarities**, present in the ``polarity`` data group. Contains the
    event's polarities in a 1D array, in which ``False`` is negative and ``True``
    is positive. The data type is ``bool``.

* The ``time_to_events`` group consists of a 1D array that saves a time mapping
  between events and timestamps. We use such time mappings to avoid doing binary
  searches online, which require a lot of computational time. The indices of the
  1D array are the timestamps (after adding the time offset), in *ms*, while
  the value itself is the corresponding event's index.
* The ``time_offset`` group indicates the starting timestamp of the sequence.

Grayscale Images
````````````````
eWiz's grayscale image data is saved in the ``gray.hdf5`` data file. It consists
of the following HDF5 groups:

* The ``gray_images`` group contains all the grayscale images of the sequence.
  It consists of a 3D array of size, *number of images* by *image height* by
  *image width*. The data type is ``uint8``.
* The ``time`` group contains the timestamp of each grayscale image in *μs*. It
  is a 1D array of type ``int64``. Each index corresponds to the image, while
  each value corresponds to its timestamp.
* The ``time_offset`` group indicates the starting timestamp of the sequence.
* The ``gray_to_events`` group consists of a 1D array that saves a grayscale mapping
  between events and images. The indices of the 1D array are the images, while
  the value itself is the corresponding event's index.
* The ``time_to_gray`` group consists of a 1D array that saves a time mapping
  between images and timestamps. The indices of the 1D array are the timestamps
  (after adding the time offset), in *ms*, while the value itself is the corresponding
  image index.

Optical Flow
````````````
eWiz's optical flow data is saved in the ``flow.hdf5`` data file. It consists
of the following HDF5 groups:

* The ``flows`` group contains all the optical flow values of the sequence.
  It consists of a 3D array of size, *number of flows* by *image height* by
  *image width*. The data type is ``int64``.
* The ``time`` group contains the timestamp of each optical flow in *μs*. It
  is a 1D array of type ``int64``. Each index corresponds to the flow, while
  each value corresponds to its timestamp.
* The ``time_offset`` group indicates the starting timestamp of the sequence.
* The ``flow_to_events`` group consists of a 1D array that saves a flow mapping
  between events and flows. The indices of the 1D array are the flows, while
  the value itself is the corresponding event's index.
* The ``time_to_flow`` group consists of a 1D array that saves a time mapping
  between flows and timestamps. The indices of the 1D array are the timestamps
  (after adding the time offset), in *ms*, while the value itself is the corresponding
  flow index.

Data Reading Examples
---------------------
Data under the eWiz format can either be read `manually <raw-data-reading_>`_,
or using the integrated `data reader <reading-with-ewiz_>`_. Reading the data
manually involves using any supported HDF5 reading library and accessing the data
based on the groups discussed above. Such method however, is not required as eWiz
has an integrated reader that takes care of the latter.

.. _raw-data-reading:

Raw Data Reading
````````````````
In this section, we explain the possibility of reading the data with the ``h5py``
Python library. We will be reading the events HDF5 file:

First, let us import the required libraries:

.. code-block:: python

  import h5py
  import hdf5plugin
  import numpy as np

.. warning::
  It is important to import ``hdf5plugin``, otherwise data reading will throw
  an error. ``hdf5plugin`` contains the BLOSC-based decompression utility.

Now, we can read the **x-coordinates** of the events as follows:

.. code-block:: python

  # Insert events HDF5 file path here
  hdf5_path = ""
  events_file = h5py.File(hdf5_path, "a")
  x_coords = events_file["events"]["x"]

  # Read x-coordinates between event 20 and 60, and load them in memory
  events_chunk = x_coords[20:60]

Congratulations! You read your first event-based data chunk using the eWiz format.
While this method is feasible, it is recommended to use the eWiz reader, as it
automates many of the tasks required to manipulate the data, especially for
multi-modal data.

.. _reading-with-ewiz:

Reading with eWiz
`````````````````
eWiz includes a data reader that automatically reads and combines multi-modal
data, whether events, grayscale images, or optical flow. In case one of the three
data types is missing, eWiz ignores that data type and returns the data available.

Data Reader
:::::::::::
To read with eWiz do the following:

First, let us import the required modules:

.. code-block:: python

  import numpy as np
  from ewiz.data.readers import ReaderFlow

Now, we can read events, grayscale images, grayscale timestamps, and optical flow:

.. code-block:: python

  # Insert data folder path here
  data_path = ""
  # Initialize data reader
  data_reader = ReaderFlow(data_path, clip_mode="time")

Clip mode refers to the method with which we want to index the events, we have
3 clip modes implemented:

* ``events``: The indices given to the data reader correspond to the event indices.
  For example, indexing from 20 to 60 returns events and images from event 20 to
  event 60 in the sequence.
* ``time``: The indices given to the data reader correspond to the timestamps.
  For example, indexing from 20 to 60 returns events and images from time *20 ms*
  till time *60 ms* in the sequence.
* ``gray``: The indices given to the data reader correspond to the grayscale images.
  For example, indexing from 20 to 60 returns events and images between image 20
  till image 60 in the sequence.

In this example, we index with timestamps as follows:

.. code-block:: python

  events, gray_images, gray_time, flow = data_reader[20:60]

.. note::
  In case no grayscale images are available, the returned ``gray_images`` and
  ``gray_time`` will be ``None``. Moreover, to only return images and events
  without the optical flow, you could use the ``ReaderBase`` class instead.

Congratulations! You used the data reader of eWiz, you can go through the API
documentation to learn more about the data reading process.

Converters
::::::::::
eWiz also includes data converters for different event-based sensors. Currently
supported sensors and datasets are: *PROPHESEE*, *DAVIS346*, *MVSEC*, and *DSEC*.

Just important the required module, and convert the data accordingly. Here is an
example for the *MVSEC* dataset:

.. code-block:: python

  # Imports
  from ewiz.data.converters import ConvertMVSEC

  # Insert MVSEC dataset directory here
  mvsec_dir = ""
  # Converted eWiz data directory
  converted_dir = ""

  # Initialize and convert
  mvsec_converter = ConvertMVSEC(mvsec_dir, converted_dir)
  mvsec_converter.convert()

Congratulations! You successfully converted the MVSEC dataset to eWiz format.
For more information, check the API documentation about data converter.


.. General References
.. _HDF5: https://www.hdfgroup.org/
.. _DAVIS: https://inivation.com/
.. _BLOSC: https://www.blosc.org/
