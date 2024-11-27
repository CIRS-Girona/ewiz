Modules
=======
Explore the core modules of eWiz, each designed to handle specific aspects of
event-based data processing. From efficient data writing and reading to conversions
and augmentations, these modules form the backbone of the library.

.. toctree::
   :maxdepth: 2

   writers
   readers
   converters
   transforms
   encoders

---

Available Modules
-----------------
1. **Data Writers**: Save event-based, grayscale, and optical flow data in a highly
   optimized format.
   - :doc:`writers`

2. **Data Readers**: Efficiently read and slice multi-modal data using indices,
   timestamps, or grayscale image frames.
   - :doc:`readers`

3. **Data Converters**: Convert datasets from various formats (e.g., MVSEC, DSEC)
   to the eWiz format.
   - :doc:`converters`

4. **Data Augmentations**: Apply temporal and spatial augmentations to event-based
   data streams. *(Planned for future releases)*
   - :doc:`transforms`

5. **Data Encoders**: Encode event streams into image formats compatible with
   convolutional neural networks (CNNs). *(Planned for future releases)*
   - :doc:`encoders`
