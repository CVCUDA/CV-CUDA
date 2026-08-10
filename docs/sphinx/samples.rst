..
   # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _samples:

CV-CUDA Samples
===============

Welcome to the CV-CUDA samples documentation! These samples demonstrate how to use CV-CUDA for GPU-accelerated computer vision and deep learning workflows.

Overview
--------

CV-CUDA samples showcase the usage of CV-CUDA operators for GPU-accelerated computer vision workflows via simple single-operator examples or complete end-to-end deep learning pipelines.

Sample Categories
^^^^^^^^^^^^^^^^^

**Operators**
  Focused examples of individual CV-CUDA operations. Great for learning specific functionality and experimenting with parameters.

**Applications**
  Complete end-to-end pipelines combining preprocessing, inference, and post-processing.

**Interoperability**
  Examples demonstrating how CV-CUDA works with other GPU frameworks (PyTorch, CuPy, PyCUDA, etc.) through zero-copy data exchange.

Walkthrough Guide
-----------------

.. _samples_venv_installation:

Installation
^^^^^^^^^^^^

The easiest way to get started is to use the installation script that automatically detects your CUDA version
and installs the appropriate dependencies (including CV-CUDA).

**Option 1: Using the Installation Script (Recommended)**

.. code-block:: bash

   cd samples
   ./install_samples_dependencies.sh

This script will:

- Detect your CUDA version (12 or 13)
- Create a virtual environment at ``venv_samples``
- Install all required dependencies including CV-CUDA, PyTorch, NumPy, and sample-specific packages

After installation, activate the virtual environment:

.. code-block:: bash

   source venv_samples/bin/activate

**Option 2: Build from Source**

Alternatively, you can build CV-CUDA from source and install the remaining dependencies.
Follow the :ref:`installation guide <installation>`, then use the installation script which will
automatically use your local build:

.. code-block:: bash

   cd samples
   ./install_samples_dependencies.sh
   # Optionally install your local wheel over the PyPI version
   source venv_samples/bin/activate
   python3 -m pip install --force-reinstall ../build-rel/python3/repaired_wheels/cvcuda-*.whl

CV-CUDA Hello World
^^^^^^^^^^^^^^^^^^^

Once you have installed the dependencies, run the Hello World sample:

.. code-block:: bash

   python3 samples/applications/hello_world.py

This simple example demonstrates the fundamental CV-CUDA workflow:

* **Reading from disk straight to GPU** (no CPU-GPU copies)
* **Resizing and batching images** for parallel processing
* **Applying operations** (Gaussian blur) on the entire batch
* **Writing to disk from GPU** (no CPU-GPU copies)

**What You'll See:**

The sample loads an image, resizes it to 224×224, applies a Gaussian blur, and saves the result to ``.cache/cat_hw.jpg``.

**Try It with Your Own Image:**

.. code-block:: bash

   python3 samples/applications/hello_world.py -i your_image.jpg -o output.jpg

**Want to Learn More?**

See the complete :ref:`Hello World documentation <sample_hello_world>` for detailed explanations of each step.

.. _running_the_samples:

Running Samples
^^^^^^^^^^^^^^^

To test all samples at once:

.. code-block:: bash

   ./samples/run_samples.sh

This script runs every sample with default parameters.
Next Steps
^^^^^^^^^^

Now that you've explored the basics:

1. **Try More Samples**: Experiment with different operators and applications
2. **Modify for Your Use Case**: Adapt samples for your specific needs
3. **Read the API Documentation**: Explore the full :ref:`Python API <python_api>`
4. **Build Your Pipelines**: Use sample patterns in your applications

.. _sample_index:

Sample Index
------------

Quick access to all CV-CUDA sample documentation.

Applications
^^^^^^^^^^^^

See :ref:`Applications Overview <sample_applications>` for an overview of the application samples.

.. toctree::
   :maxdepth: 1

   samples/applications
   samples/applications/hello_world
   samples/applications/classification
   samples/applications/object_detection
   samples/applications/segmentation

Operators
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   adaptivethreshold <samples/operators/adaptivethreshold>
   advcvtcolor <samples/operators/advcvtcolor>
   averageblur <samples/operators/averageblur>
   bilateral_filter <samples/operators/bilateral_filter>
   bndbox <samples/operators/bndbox>
   boxblur <samples/operators/boxblur>
   brightness_contrast <samples/operators/brightness_contrast>
   center_crop <samples/operators/center_crop>
   channelreorder <samples/operators/channelreorder>
   clahe <samples/operators/clahe>
   color_twist <samples/operators/color_twist>
   composite <samples/operators/composite>
   conv2d <samples/operators/conv2d>
   convertto <samples/operators/convertto>
   copymakeborder <samples/operators/copymakeborder>
   crop_flip_normalize_reformat <samples/operators/crop_flip_normalize_reformat>
   customcrop <samples/operators/customcrop>
   cvtcolor <samples/operators/cvtcolor>
   erase <samples/operators/erase>
   flip <samples/operators/flip>
   gamma_contrast <samples/operators/gamma_contrast>
   gaussian <samples/operators/gaussian>
   gaussiannoise <samples/operators/gaussiannoise>
   histogrameq <samples/operators/histogrameq>
   hq_resize <samples/operators/hq_resize>
   inpaint <samples/operators/inpaint>
   joint_bilateral_filter <samples/operators/joint_bilateral_filter>
   label <samples/operators/label>
   laplacian <samples/operators/laplacian>
   median_blur <samples/operators/median_blur>
   morphology <samples/operators/morphology>
   normalize <samples/operators/normalize>
   osd <samples/operators/osd>
   pillowresize <samples/operators/pillowresize>
   random_resized_crop <samples/operators/random_resized_crop>
   reformat <samples/operators/reformat>
   remap <samples/operators/remap>
   resize <samples/operators/resize>
   resize_crop_convert_reformat <samples/operators/resize_crop_convert_reformat>
   rotate <samples/operators/rotate>
   stack <samples/operators/stack>
   threshold <samples/operators/threshold>
   warp_affine <samples/operators/warp_affine>
   warp_perspective <samples/operators/warp_perspective>

Interoperability
^^^^^^^^^^^^^^^^

See :ref:`Interoperability Overview <sample_interoperability>` for an overview of the interoperability samples.

.. toctree::
   :maxdepth: 1

   samples/interoperability
   samples/interoperability/pytorch
   samples/interoperability/cuda_python
   samples/interoperability/numpy
   samples/interoperability/nvimgcodec
   samples/interoperability/pynvvideocodec
   samples/interoperability/cupy
   samples/interoperability/pycuda

Common Utilities
----------------

.. toctree::
   :maxdepth: 1

   samples/common

Additional Resources
--------------------

* :ref:`Common Utilities <sample_common>` - Shared helper functions reference
* :ref:`Python API <python_api>` - Core API reference
* :ref:`Installation Guide <installation>` - Build and setup instructions
* `GitHub Repository <https://github.com/CVCUDA/CV-CUDA>`_ - Source code and issue tracker
* `Discussions <https://github.com/CVCUDA/CV-CUDA/discussions>`_ - Ask questions and share use cases
