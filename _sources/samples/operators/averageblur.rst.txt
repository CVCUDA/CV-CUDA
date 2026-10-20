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

.. _sample_averageblur:

Average Blur
============

Overview
--------

The Average Blur sample demonstrates GPU-accelerated box filtering using CV-CUDA's
``averageblur`` operator. Each output pixel is the arithmetic mean of the pixels
within a rectangular kernel, producing a smoothing (low-pass) effect that reduces
noise and fine detail.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply the default 7×7 average blur to an image:

.. code-block:: bash

   python3 averageblur.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify a different output file:

.. code-block:: bash

   python3 averageblur.py -i input.jpg -o cat_averageblur.jpg

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
     - cvcuda/.cache/cat_averageblur.jpg
     - Output image file path

Implementation
--------------

Average Blur Operator Call
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/averageblur.py
   :language: python
   :start-after: docs_tag: begin_averageblur
   :end-before: docs_tag: end_averageblur
   :dedent:

Key points:

1. **Kernel size**: ``[7, 7]`` specifies a 7-pixel-wide by 7-pixel-tall averaging window; larger kernels produce stronger blurring.
2. **Kernel anchor**: ``[-1, -1]`` automatically centers the anchor within the kernel, which is standard for symmetric filters.
3. **Border mode**: ``cvcuda.Border.REFLECT101`` mirrors pixels across the border without repeating the edge pixel, preventing visible seams at image boundaries.
4. **Supported dtypes**: U8, U16, S16, S32, and F32 are all supported, making the operator suitable for both display images and intermediate float feature maps.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image with a 7×7 box blur applied:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_averageblur.jpg
          :width: 100%

          Output: 7×7 Average Blur Applied

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.averageblur`
     - Apply a box (average) blur with a rectangular kernel to smooth the image

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save blurred image to disk

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images with GPU acceleration
* :ref:`Common Utilities <sample_common>` - Shared helper functions
