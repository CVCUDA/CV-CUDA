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

.. _sample_pillowresize:

Pillow Resize
=============

Overview
--------

The Pillow Resize sample demonstrates high-quality image resizing using CV-CUDA's
GPU-accelerated Pillow-style resize operator. Unlike a plain bilinear or nearest-neighbour
resize, ``pillowresize`` matches the resampling quality of Python's Pillow library by
supporting filters such as LANCZOS, HAMMING, and BOX that are especially well-suited for
downscaling images.

Usage
-----

Basic Usage
^^^^^^^^^^^

Resize an image to 224×224 (default) using the LANCZOS filter:

.. code-block:: bash

   python3 pillowresize.py -i input.jpg

Custom Dimensions
^^^^^^^^^^^^^^^^^

Specify target width and height:

.. code-block:: bash

   python3 pillowresize.py -i input.jpg -o cat_pillowresize.jpg --width 512 --height 512

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
     - cvcuda/.cache/cat_pillowresize.jpg
     - Output image file path
   * - ``--width``
     -
     - 224
     - Target width in pixels
   * - ``--height``
     -
     - 224
     - Target height in pixels

Implementation
--------------

Pillow-Quality Resize
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/pillowresize.py
   :language: python
   :start-after: docs_tag: begin_pillowresize
   :end-before: docs_tag: end_pillowresize
   :dedent:

Key points:

1. **Output Shape**: Must include channel count explicitly, e.g. ``(H, W, C)`` for HWC tensors.
2. **Format Parameter**: Tells the operator how to interpret channel ordering (e.g. ``cvcuda.Format.RGB8``).
3. **LANCZOS Filter**: Produces sharper edges than LINEAR and is the recommended choice for downscaling, matching Pillow's high-quality mode.
4. **uint8 Output**: The operator preserves the input dtype; reading a JPEG returns ``uint8``, so the result is directly viewable without rescaling.
5. **Interp Variants**: ``HAMMING`` and ``BOX`` are also available and offer different quality/speed trade-offs for downscaling.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image resized to the target dimensions (default 224×224) with
Pillow-quality LANCZOS interpolation:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_pillowresize.jpg
          :width: 100%

          Output: Pillow Resize to 224×224 (LANCZOS)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.pillowresize`
     - Resize images to target dimensions using Pillow-compatible high-quality filters

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save resized image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Standard GPU resize operator
* :ref:`Common Utilities <sample_common>` - Helper functions
