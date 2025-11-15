..
  # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  # http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.

.. _getting_started:

Getting Started
===============

Welcome to CV-CUDA! This guide will help you get up and running with GPU-accelerated computer vision.

What is CV-CUDA?
----------------

CV-CUDA is a library of GPU-accelerated computer vision operators optimized for AI workflows. It enables you to:

* **Accelerate image pre- and post-processing** for CV AI models on NVIDIA GPUs
* **Keep data on GPU** throughout your entire pipeline (zero CPU-GPU copies)
* **Batch operations efficiently** for maximum throughput
* **Integrate seamlessly** with PyTorch, TensorRT, and other GPU libraries

Prerequisites
-------------

Before diving into CV-CUDA, make sure you have the necessary hardware and software.

See the :ref:`Prerequisites Information <prerequisites>` for complete hardware and software requirements.

Quick Start (5 Minutes)
------------------------

**1. Install Dependencies**

For CUDA 12:

.. code-block:: bash

   python3 -m venv venv_samples
   source venv_samples/bin/activate
   python3 -m pip install -r samples/requirements_hello_world_cu12.txt

For CUDA 13:

.. code-block:: bash

   python3 -m venv venv_samples
   source venv_samples/bin/activate
   python3 -m pip install -r samples/requirements_hello_world_cu13.txt

This installs minimal dependencies (CV-CUDA, NumPy, nvImageCodec) needed for the hello_world sample.

**2. Run Your First Sample**

.. code-block:: bash

   python3 samples/applications/hello_world.py

**3. See Results**

Check ``cvcuda/.cache/cat_hw.jpg`` - you just processed an image entirely on GPU!

.. note::

   The ``requirements_hello_world_cu12.txt`` and ``requirements_hello_world_cu13.txt`` files are minimal (only 4 packages) for quick testing.
   For other samples (operators, applications, interoperability), use the full installation script:

   .. code-block:: bash

      cd samples
      ./install_samples_dependencies.sh

**What's Next?** Continue below to learn the prerequisites and explore more samples.

Samples
-------

The samples are the best way to learn CV-CUDA. They demonstrate everything from basic operations to complete deep learning pipelines.

**What's in the Samples:**

* **Hello World** - Your introduction to CV-CUDA (load, resize, blur, save)
* **Operators** - Learn individual CV-CUDA operations (resize, blur, reformat, etc.)
* **Applications** - Complete pipelines (classification, detection, segmentation)

**View the Samples Documentation:**

See the :doc:`Samples Documentation <samples>` for a guided tour of all available examples.

Interoperability
----------------

See the :doc:`Interoperability <interoperability>` for information on how to use CV-CUDA with other libraries.

Advanced Topics
---------------

Once you're comfortable with the basics, explore advanced features:

* :doc:`Object Cache <advanced/object_cache>` - Learn about CV-CUDA's memory caching system
* :doc:`Make Operator Tool <advanced/make_operator>` - Create custom CV-CUDA operators

Additional Resources
--------------------

* **API Reference**: :ref:`Python API Documentation <python_api>`
* **Installation Guide**: :doc:`Detailed Installation <installation>`
* **GitHub**: `CV-CUDA Repository <https://github.com/CVCUDA/CV-CUDA>`_
* **Discussions**: `Ask Questions <https://github.com/CVCUDA/CV-CUDA/discussions>`_

Need Help?
----------

* Check the :doc:`Samples <samples>` for code examples
* Review the :ref:`Python API <python_api>` documentation
* Search `GitHub Issues <https://github.com/CVCUDA/CV-CUDA/issues>`_
* Ask on the `discussion forum <https://github.com/CVCUDA/CV-CUDA/discussions>`_
