..
   # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _sample_adaptivethreshold:

Adaptive Threshold
==================

Overview
--------

The Adaptive Threshold sample demonstrates locally-adaptive binarisation of a grayscale image using
CV-CUDA's GPU-accelerated adaptive threshold operator.  Unlike a global threshold, adaptive
thresholding computes a per-pixel threshold from a local neighbourhood, making it robust to
uneven illumination.  The sample converts the colour input to grayscale, runs
:py:func:`cvcuda.adaptivethreshold`, then broadcasts the single-channel result back to RGB for
saving as a viewable JPEG.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply adaptive thresholding to the default cat image:

.. code-block:: bash

   python3 adaptivethreshold.py

Custom Input
^^^^^^^^^^^^

Supply your own image and output path:

.. code-block:: bash

   python3 adaptivethreshold.py -i input.jpg -o cat_adaptivethreshold.jpg

Command-Line Arguments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Argument
     - Short Form
     - Default
     - Description
   * - ``--input``
     - ``-i``
     - tabby_tiger_cat.jpg
     - Input image file path
   * - ``--output``
     - ``-o``
     - cvcuda/.cache/cat_adaptivethreshold.jpg
     - Output image file path

Implementation
--------------

Adaptive Threshold Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/adaptivethreshold.py
   :language: python
   :start-after: docs_tag: begin_adaptivethreshold
   :end-before: docs_tag: end_adaptivethreshold
   :dedent:

Key points:

1. **Single-channel input**: :py:func:`cvcuda.adaptivethreshold` requires a U8 single-channel (HWC with C=1 or NHWC with C=1) tensor; colour images must be converted to grayscale first.
2. **Adaptive method**: ``GAUSSIAN_C`` uses a Gaussian-weighted neighbourhood average; ``MEAN_C`` uses a plain mean — both then subtract the constant ``c`` to produce the local threshold.
3. **block_size**: Must be an odd integer ≥ 3; larger values consider a wider neighbourhood and produce smoother thresholds.
4. **c constant**: A positive ``c`` makes the threshold stricter (fewer pixels exceed it), producing a sparser binary result; negative values do the opposite.
5. **Viewable output**: The single-channel binary result is replicated to three channels on the host before writing so that standard JPEG viewers can display it correctly.

Expected Output
^^^^^^^^^^^^^^^

The output is a binary (black-and-white) image where pixel intensity reflects whether each
pixel exceeded its local Gaussian-weighted neighbourhood threshold:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_adaptivethreshold.jpg
          :width: 100%

          Output: Adaptive Threshold (GAUSSIAN_C, block=11, c=2)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.cvtcolor`
     - Convert RGB input to single-channel grayscale
   * - :py:func:`cvcuda.adaptivethreshold`
     - Apply locally-adaptive binarisation per pixel

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load input image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save thresholded result as JPEG
* ``cuda_memcpy_d2h`` / ``cuda_memcpy_h2d`` - Transfer binary result to host for channel replication, then back to device

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic image transformation
* :ref:`Common Utilities <sample_common>` - Helper functions
