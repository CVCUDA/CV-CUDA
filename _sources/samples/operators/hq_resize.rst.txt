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

.. _sample_hq_resize:

HQ Resize
=========

Overview
--------

The HQ Resize sample demonstrates high-quality image resizing using CV-CUDA's GPU-accelerated HQ
Resize operator. Unlike the standard Resize operator, HQ Resize accepts separate interpolation
filters for downscaling (``min_interpolation``) and upscaling (``mag_interpolation``), and
optionally applies an antialiasing low-pass filter before downscaling to eliminate moiré patterns
and ringing artifacts.

Usage
-----

Basic Usage
^^^^^^^^^^^

HQ-resize an image to 224×224 (default):

.. code-block:: bash

   python3 hq_resize.py -i input.jpg

Custom Dimensions
^^^^^^^^^^^^^^^^^

Specify a target width and height with a custom output path:

.. code-block:: bash

   python3 hq_resize.py -i input.jpg -o cat_hq_resize.jpg --width 512 --height 512

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
     - cvcuda/.cache/cat_hq_resize.jpg
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

HQ Resize Operator Call
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/hq_resize.py
   :language: python
   :start-after: docs_tag: begin_hq_resize
   :end-before: docs_tag: end_hq_resize
   :dedent:

Key points:

1. **Dual interpolation filters**: ``min_interpolation`` governs downscaling and
   ``mag_interpolation`` governs upscaling, allowing the best filter to be chosen
   for each direction independently.
2. **LANCZOS for minification**: The Lanczos filter provides superior sharpness
   and suppresses aliasing compared to LINEAR or NEAREST when reducing image size.
3. **Antialiasing flag**: Setting ``antialias=True`` applies a low-pass filter
   before downscaling, which further reduces moiré and ringing in the output.
4. **out_size is (H, W)**: The target size is specified as a ``(height, width)``
   tuple — no channel dimension is included; the operator infers it from the input layout.
5. **U8 in/out, no conversion**: The output tensor inherits the data type and layout
   (HWC, uint8) of the input, so no additional type conversion is needed before saving.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image resized to the target dimensions (default 224×224):

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_hq_resize.jpg
          :width: 100%

          Output: HQ-Resized to 224×224

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.hq_resize`
     - High-quality resize with separate min/mag interpolation filters and optional antialiasing

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save the resized image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Standard (lower-overhead) resize operator
* :ref:`Common Utilities <sample_common>` - Helper functions
