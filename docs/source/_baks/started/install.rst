Installation
------------
The eWiz library is mainly compatible with `Python`_ 3.8 and requires `PyTorch`_
2.0.1 and above to run. eWiz makes use of `PyTorch`_ and `CUDA`_ to run its
optimization algorithms, more specifically, :doc:`motion compensation <../mc/index>`.

.. note::
    It is recommended to use eWiz with `Ubuntu`_ 20.04 in a separate virtual
    environment with `Python`_ 3.8 installed (using `Anaconda`_ for example).

    Check that your graphic card drivers are correctly installed on your system,
    as eWiz requires `CUDA`_ 11.8 and above.


You can install eWiz directly using `PyPi`_ along all its dependencies. However,
you need to have `PyTorch`_ 2.0.1 or above already installed in your `Python`_ environment.
Follow the steps below to install eWiz:

Activate your `Anaconda`_ environment and start by installing your preferred
version of `PyTorch`_ and `Torchvision`_. Don't forget to install the `CUDA`_
supported version.

.. highlight:: console

An example `PyTorch`_ installation command can be found below: ::

    $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

This should install `PyTorch`_ along-side `CUDA`_ on your system.


.. highlight:: console

Afterwards, you can directly install eWiz with `PyPi`_ inside your environment: ::

    $ pip install ewiz

You now have eWiz installed, along all its dependencies. Check out our
:doc:`start guide <start>` to learn how to run some example scripts.


.. Page References
.. _Python: https://www.python.org/
.. _PyTorch: https://pytorch.org/
.. _Torchvision: https://pytorch.org/vision/stable/index.html
.. _Anaconda: https://www.anaconda.com/
.. _Ubuntu: https://ubuntu.com/
.. _PyPi: https://pypi.org/
.. _CUDA: https://developer.nvidia.com/cuda-toolkit/
