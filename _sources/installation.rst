Installation
============

Requirements
------------

torch-diffsim requires Python 3.8 or later and the following dependencies:

* PyTorch >= 2.0.0
* NumPy >= 1.20.0
* SciPy >= 1.7.0
* meshio >= 5.0.0
* polyscope >= 1.3.0
* matplotlib >= 3.5.0

Install from PyPI
-----------------

The easiest way to install torch-diffsim is via pip ![PyPI](https://img.shields.io/pypi/v/torch-diffsim?style=flat-square)

.. code-block:: bash

   pip install torch-diffsim

This will automatically install all required dependencies.

Install from Source
-------------------

For development or to get the latest features, you can install from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Rishit-dagli/torch-diffsim
   cd torch-diffsim
   
   # Install in editable mode
   pip install -e .

Development Installation
------------------------

If you want to contribute to torch-diffsim, install the development dependencies:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Rishit-dagli/torch-diffsim
   cd torch-diffsim
   
   # Install with development dependencies
   pip install -e ".[dev]"

This will install additional tools for testing and code formatting:

* pytest >= 7.0.0
* black >= 22.0.0
* ruff >= 0.1.0

Verifying Installation
----------------------

You can verify your installation by running:

.. code-block:: python

   import diffsim
   print(diffsim.__version__)

This should print the version number (e.g., ``0.1.0``).

GPU Support
-----------

torch-diffsim uses PyTorch and will automatically use GPU acceleration if CUDA is available. To check if GPU support is enabled:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")

For optimal performance, we recommend using a CUDA-enabled GPU, especially for large-scale simulations or optimization problems.

Troubleshooting
---------------

**Import errors**: If you encounter import errors, ensure that all dependencies are installed correctly:

.. code-block:: bash

   pip install torch numpy scipy meshio polyscope matplotlib

**PyTorch installation**: For specific PyTorch installation instructions (e.g., with specific CUDA versions), visit the `official PyTorch website <https://pytorch.org/get-started/locally/>`_.

**Polyscope rendering issues**: If you encounter issues with polyscope visualization, ensure you have proper OpenGL drivers installed on your system.

