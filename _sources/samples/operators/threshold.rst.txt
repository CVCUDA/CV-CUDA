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

.. _sample_threshold:

Threshold
=========

Overview
--------

The Threshold sample demonstrates pixel-level intensity thresholding using CV-CUDA's GPU-accelerated
threshold operator. A binary threshold is applied: every pixel whose value exceeds a configurable
threshold is set to a maximum value (255), and all others are set to 0, producing a clean binary mask.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply a binary threshold to an image (default threshold = 128):

.. code-block:: bash

   python3 threshold.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify a custom output file:

.. code-block:: bash

   python3 threshold.py -i input.jpg -o thresholded.jpg

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
     - cvcuda/.cache/cat_threshold.jpg
     - Output image file path

Implementation
--------------

Threshold Operator
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/threshold.py
   :language: python
   :start-after: docs_tag: begin_threshold
   :end-before: docs_tag: end_threshold
   :dedent:

Key points:

1. **Batch dimension required**: ``cvcuda.threshold`` expects NHWC layout, so a HWC image must be
   reshaped with a leading batch dimension before calling the operator.
2. **Per-image parameters**: ``thresh`` and ``maxval`` are GPU tensors of shape ``(N,)`` and dtype
   ``F64``, allowing each image in a batch to use a different threshold value.
3. **Upload via cuda_memcpy_h2d**: NumPy arrays holding the scalar parameters are copied to GPU
   memory using ``cuda_memcpy_h2d`` before the operator is called.
4. **BINARY type**: ``cvcuda.ThresholdType.BINARY`` sets pixels above the threshold to ``maxval``
   and all others to zero; other types (``BINARY_INV``, ``TRUNC``, ``TOZERO``, ``TOZERO_INV``,
   ``OTSU``, ``TRIANGLE``) are also available.
5. **Output shape preserved**: The operator returns a tensor with the same shape, layout, and dtype
   as the input, which is reshaped back to HWC before writing.

Expected Output
^^^^^^^^^^^^^^^

The output is a binary image where bright regions (pixel value > 128) appear white and dark regions
appear black:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_threshold.jpg
          :width: 100%

          Output: Binary Threshold (thresh=128, maxval=255)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.threshold`
     - Apply pixel-intensity thresholding with configurable per-image threshold and maxval

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save thresholded image
* ``cuda_memcpy_h2d`` - Upload per-image threshold and maxval scalars to GPU memory

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic GPU image resize
* :ref:`Label Operator <sample_label>` - Connected-components labeling using threshold as preprocessing
* :ref:`Common Utilities <sample_common>` - Helper functions
