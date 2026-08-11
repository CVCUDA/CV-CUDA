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

.. _sample_resize_crop_convert_reformat:

Resize Crop Convert Reformat
=============================

Overview
--------

The Resize Crop Convert Reformat sample demonstrates a fused preprocessing pipeline using
CV-CUDA's GPU-accelerated :py:func:`cvcuda.resize_crop_convert_reformat` operator.  In a
single kernel the operator:

* Resizes the input image to a target dimension using linear or nearest-neighbor
  interpolation,
* Crops a rectangular region of interest from the resized result,
* Converts the pixel data type (e.g. ``uint8`` → ``float32``), and
* Reformats the memory layout (e.g. ``NHWC`` → ``NCHW``) and optionally reverses
  the channel order (BGR ↔ RGB).

This mirrors the standard ImageNet-style pre-processing pipeline that DL inference
frameworks apply before feeding images to a convolutional network.

Usage
-----

Basic Usage
^^^^^^^^^^^

Run with the default 224×224 resize / 224×224 crop:

.. code-block:: bash

   python3 resize_crop_convert_reformat.py -i input.jpg

Custom Dimensions
^^^^^^^^^^^^^^^^^

Specify a different resize and crop target (crop is always the full ``--width`` ×
``--height`` window starting at the top-left corner):

.. code-block:: bash

   python3 resize_crop_convert_reformat.py -i input.jpg -o result.jpg --width 512 --height 512

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
     - cvcuda/.cache/cat_resize_crop_convert_reformat.jpg
     - Output image file path
   * - ``--width``
     -
     - 224
     - Target crop width in pixels (also used as resize width)
   * - ``--height``
     -
     - 224
     - Target crop height in pixels (also used as resize height)

Implementation
--------------

Fused Pipeline
^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/resize_crop_convert_reformat.py
   :language: python
   :start-after: docs_tag: begin_resize_crop_convert_reformat
   :end-before: docs_tag: end_resize_crop_convert_reformat
   :dedent:

Key points:

1. **Single-kernel fusion**: resize, crop, type conversion, layout reformat, and optional
   channel reversal all happen in one GPU pass, avoiding intermediate allocations and
   memory bandwidth waste.
2. **``layout="NCHW"``**: the output tensor uses channel-first memory order, which is the
   format expected by most deep-learning inference runtimes (TensorRT, ONNX Runtime, etc.).
3. **``manip=cvcuda.ChannelManip.REVERSE``**: swaps BGR ↔ RGB in the same pass, which is
   needed when the codec reads BGR and the model was trained on RGB (or vice versa).
4. **``scale`` and ``offset``**: optional linear normalisation ``output = pixel / scale + offset``
   applied after type conversion; set ``scale=127.5, offset=-1`` for ``[-1, 1]``
   normalisation used by many classification and detection models.
5. **Host-side reformat for saving**: because the image encoder expects HWC uint8, the
   NCHW float32 result is transposed and clipped on the CPU before writing.

Expected Output
^^^^^^^^^^^^^^^

The output shows the input image resized to 224×224, cropped to 224×224, and returned
to a viewable HWC uint8 format for saving:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_resize_crop_convert_reformat.jpg
          :width: 100%

          Output: Resized, Cropped, and Reformatted

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.resize_crop_convert_reformat`
     - Fused resize → crop → type convert → layout reformat in a single GPU kernel

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save result image
* ``cuda_memcpy_d2h`` / ``cuda_memcpy_h2d`` - Transfer tensor data between GPU and CPU for host-side reformat

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Simple image resize
* :ref:`Common Utilities <sample_common>` - Helper functions
