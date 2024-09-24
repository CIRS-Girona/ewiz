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
Add text here.

Events
``````
Add text here.

Grayscale Images
````````````````
Add text here.

Optical Flow
````````````
Add text here.

Data Reading Examples
---------------------
Add text here.

Raw Data Reading
````````````````
Add text here.

Reading with eWiz
`````````````````
Add text here.

Converters
::::::::::
Add text here.

Data Reader
:::::::::::
Add text here.


.. General References
.. _HDF5: https://www.hdfgroup.org/
.. _DAVIS: https://inivation.com/
