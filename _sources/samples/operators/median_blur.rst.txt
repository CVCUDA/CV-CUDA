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

.. _sample_median_blur:

Median Blur
===========

Overview
--------

The Median Blur sample demonstrates GPU-accelerated median filtering using CV-CUDA's
``median_blur`` operator. Median blur replaces each pixel with the median value of its
neighborhood, making it highly effective for removing salt-and-pepper noise while
preserving edges better than a simple averaging blur.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply a 7×7 median blur to an image (default):

.. code-block:: bash

   python3 median_blur.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify a custom output path:

.. code-block:: bash

   python3 median_blur.py -i input.jpg -o blurred.jpg

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
     - cvcuda/.cache/cat_median_blur.jpg
     - Output image file path

Implementation
--------------

Median Blur Operation
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/median_blur.py
   :language: python
   :start-after: docs_tag: begin_median_blur
   :end-before: docs_tag: end_median_blur
   :dedent:

Key points:

1. **Kernel size**: ``ksize`` is a two-element list ``[kW, kH]`` where both values must be odd positive integers. Larger kernels produce stronger smoothing.
2. **Noise removal**: Median blur is especially effective for removing impulse (salt-and-pepper) noise because the median statistic is robust to outliers.
3. **Edge preservation**: Unlike mean blur, median blur preserves edges well since the median value is always drawn from actual pixel values in the neighborhood.
4. **Supported types**: The operator supports ``uint8``, ``uint16``, and ``float32`` data types in HWC or NHWC layout.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image with impulse noise suppressed and edges intact:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_median_blur.jpg
          :width: 100%

          Output: Median Blur (7×7 kernel)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.median_blur`
     - Apply median blur filter to remove noise while preserving edges

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save blurred image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images with interpolation
* :ref:`Common Utilities <sample_common>` - Helper functions
