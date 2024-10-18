Features
========
eWiz is a Python-based library that can be installed with PyPi.
eWiz works hand-in-hand with deep learning libraries, such as PyTorch and Tonic,
making integration easier in deep learning-oriented pipelines.

Data Manipulation
-----------------
Event-based data manipulation has been made easier with eWiz, through optimized
data reading and writing techniques. All data manipulation algorithms are encapsulated
inside a ``data`` module, which contains data converters, encoders, loaders, readers,
writers, and transforms. These modules also support grayscale images and optical
flow data in case ground truth is provided as input. Each module is explained in
detail in the following.

Writers
```````
eWiz's data writers are at the library's core since they are used to save
event-based, grayscale, and optical flow data. These writers were implemented with
optimized data management in mind, as event-based data requires a big amount of
storage space. To avoid huge dataset sizes, the data uses the BLOSC compression
format. Moreover, we avoid overloading the memory by directly saving the input
data on disk. eWiz relies on the ``h5py`` library and saves data as ``.hdf5``.

The data writer module can be used for data conversion, for any event-based
sensor, or for data saving from simulation. Data conversion is showcased inside
the ``converters`` module, while data saving from simulation is shown inside the
*eStonefish* and *eCARLA* data gathering pipelines.

eWiz also uses time mapping techniques to avoid time consuming search algorithms
when indexing with timestamps. Similar techniques are applied for the grayscale
and optical flow data. In addition, each image, whether grayscale or optical flow,
is directly mapped to its corresponding event index. To learn more about the data
format eWiz uses, check the *Data Format* section.

Readers
```````
eWiz's data readers are designed to easily read event-based, grayscale, and
optical flow data. The ``reader`` module is compatible with the eWiz format, which
contains can include up to 3 HDF5 files (one file for each data type), and a ``.json``
file that contains camera properties (resolution). The readers make use of the saved
time mappings to index and slice the data, whether with event indices, timestamps,
or grayscale images. Also, optical flow data is automatically synchronized through
interpolating and accumulating flow displacements between the two desired timestamps.

The data readers do not require a high amount of RAM, as the data is read in chunks
from storage. However, the use of solid state drives is recommended as some event
data chunks are relatively big to write on hard-disk drives. You can learn more
about eWiz's data readers by checking the *reader API documentation*.

Loaders
```````
The data loaders of eWiz are based on the readers. Their functionality is similar
to PyTorch's dataset module, in the sense that they are iterator objects which iterate
through the data. The user can choose a data stride, whether event indices, timestamps,
or grayscale images, and load the desired data sequentially. To learn more
about eWiz's data loaders, check the *loader API documentation*.

Converters
``````````
Add text here.
