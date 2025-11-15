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

.. _sample_operators:

Operators
=========

Individual operator samples demonstrating specific CV-CUDA operations.

Overview
--------

The operator samples show focused functionality for understanding specific operations:

* **Gaussian** - Blur and smoothing with configurable kernel and sigma
* **Resize** - Image resizing with various interpolation methods (linear, cubic, area, nearest)
* **Reformat** - Tensor layout conversions (HWC, CHW, NHWC, NCHW)
* **Stack** - Batch creation from multiple tensors for parallel processing
* **Label** - Connected component labeling for region identification

These samples are perfect for:

* Learning individual operator behavior
* Understanding operator parameters
* Quick experimentation
* Building custom pipelines

Operator Samples
----------------

.. toctree::
   :maxdepth: 1

   Gaussian Blur <operators/gaussian>
   Resize <operators/resize>
   Reformat <operators/reformat>
   Stack <operators/stack>
   Connected Components Labeling <operators/label>

Common Usage Patterns
---------------------

Single Operator
^^^^^^^^^^^^^^^

Simple, focused operation:

.. code-block:: python

   import cvcuda
   from common import read_image, write_image

   image = read_image("input.jpg")
   result = cvcuda.gaussian(image, (5, 5), (1.0, 1.0))
   write_image(result, "output.jpg")

Chaining Operators
^^^^^^^^^^^^^^^^^^

Combine multiple operations:

.. code-block:: python

   image = read_image("input.jpg")
   resized = cvcuda.resize(image, (224, 224, 3))
   blurred = cvcuda.gaussian(resized, (5, 5), (1.0, 1.0))
   write_image(blurred, "output.jpg")

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   batch = cvcuda.stack([read_image(p) for p in paths])
   processed = cvcuda.gaussian(batch, (5, 5), (1.0, 1.0))

See Also
--------

* :ref:`Applications <sample_applications>` - End-to-end pipelines
* :ref:`Common Utilities <sample_common>` - Helper functions
* :ref:`Python API <python_api>` - Complete operator API documentation
