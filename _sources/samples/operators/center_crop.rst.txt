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

.. _sample_center_crop:

Center Crop
===========

Overview
--------

The Center Crop sample demonstrates symmetric center-cropping of an image using
CV-CUDA's GPU-accelerated ``center_crop`` operator.  The operator extracts a
rectangular region from the geometric centre of the image without requiring the
caller to compute corner offsets manually.

Usage
-----

Basic Usage
^^^^^^^^^^^

Crop the default cat image to 224×224 pixels:

.. code-block:: bash

   python3 center_crop.py -i input.jpg

Custom Crop Size
^^^^^^^^^^^^^^^^

Specify a different crop width and height:

.. code-block:: bash

   python3 center_crop.py -i input.jpg -o cat_center_crop.jpg --width 320 --height 240

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
     - cvcuda/.cache/cat_center_crop.jpg
     - Output image file path
   * - ``--width``
     -
     - 224
     - Crop width in pixels (clamped to input width)
   * - ``--height``
     -
     - 224
     - Crop height in pixels (clamped to input height)

Implementation
--------------

Center Crop
^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/center_crop.py
   :language: python
   :start-after: docs_tag: begin_center_crop
   :end-before: docs_tag: end_center_crop
   :dedent:

Key points:

1. **No coordinate math**: ``cvcuda.center_crop`` computes the top-left corner
   internally, so the caller only needs to supply the desired ``[height, width]``.
2. **crop_size list**: The second argument is a two-element Python list
   ``[crop_height, crop_width]`` — not a tuple.
3. **Layout preserved**: The output tensor shares the same layout (HWC/NHWC) and
   dtype as the input; no conversion is needed before writing.
4. **Clamping**: The sample clamps the requested crop dimensions to the actual image
   size to avoid an out-of-bounds error when the crop is larger than the source.
5. **Single-image usage**: The sample operates on an HWC tensor directly; batched
   NHWC usage follows the same ``crop_size`` argument convention.

Expected Output
^^^^^^^^^^^^^^^

The output is the center portion of the input image at the requested dimensions
(default 224×224):

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_center_crop.jpg
          :width: 100%

          Output: Center-cropped to 224×224

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.center_crop`
     - Symmetrically crop a rectangular region from the centre of an image

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save cropped image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Scale images to arbitrary dimensions
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
