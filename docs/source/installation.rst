Installation
============

System Requirements
-------------------
- **Python**: 3.8
- **PyTorch**: 2.0.1 or higher
- **CUDA**: 11.8 or higher for GPU support
- **Operating System**: Ubuntu 20.04

---

Steps for Installation
----------------------
1. Create a virtual environment.
2. Install PyTorch and TorchVision with CUDA support:

   .. code-block:: bash

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. Install eWiz using PyPi:

   .. code-block:: bash

    pip install ewiz

For detailed system requirements, check:

- `Python <https://www.python.org/>`_
- `PyTorch <https://pytorch.org/>`_
- `CUDA <https://developer.nvidia.com/cuda-toolkit/>`_

---

.. note::

   Ensure that your GPU drivers are updated and compatible with the required
   CUDA version.
