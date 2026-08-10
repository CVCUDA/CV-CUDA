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

.. _sample_advcvtcolor:

Advanced Color Conversion
=========================

Overview
--------

The Advanced Color Conversion sample demonstrates GPU-accelerated color-space transformation
using CV-CUDA's ``advcvtcolor`` operator. Unlike the basic :ref:`CvtColor Operator <sample_cvtcolor>`,
``advcvtcolor`` accepts a :pydata:`cvcuda.ColorSpec` argument that selects the standardized
luma/chroma coefficients (BT.601, BT.709, or BT.2020) used during the conversion. This sample
converts an RGB image to YUV (BT.709) and then back to RGB, demonstrating the round-trip workflow
common in video-processing pipelines.

Usage
-----

Basic Usage
^^^^^^^^^^^

Run with the default tabby cat image:

.. code-block:: bash

   python3 advcvtcolor.py -i input.jpg

Custom Input and Output
^^^^^^^^^^^^^^^^^^^^^^^

Specify a custom input image and output path:

.. code-block:: bash

   python3 advcvtcolor.py -i image.jpg -o cat_advcvtcolor.jpg

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
     - cvcuda/.cache/cat_advcvtcolor.jpg
     - Output image file path

Implementation
--------------

RGB to YUV and Back
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/advcvtcolor.py
   :language: python
   :start-after: docs_tag: begin_advcvtcolor
   :end-before: docs_tag: end_advcvtcolor
   :dedent:

Key points:

1. **Color Specification**: The ``spec`` argument selects the luma/chroma coefficients standard
   (BT.601 for SD video, BT.709 for HDTV, BT.2020 for UHD/HDR). Mixing specifications between
   forward and inverse conversions will produce incorrect colors.
2. **Round-trip fidelity**: Converting RGB → YUV → RGB with the same ``ColorSpec`` closely
   reproduces the original image; any visible difference is due to quantization in uint8.
3. **Supported layouts**: Both ``HWC`` (single image) and ``NHWC`` (batch) layouts are accepted
   without any reshaping step.
4. **Output shape preserved**: The output tensor always has the same shape and dtype as the input,
   so no extra allocation or reshape is needed for 444 (3-channel) conversions.
5. **NV12/NV21 variants**: For semi-planar YUV (NV12/NV21) conversions the input height must be
   ``H * 3 / 2`` and channels must be 1; the 444 interleaved path used here keeps the standard
   ``(H, W, 3)`` shape.

Expected Output
^^^^^^^^^^^^^^^

After the RGB → YUV → RGB round-trip the image looks nearly identical to the original:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_advcvtcolor.jpg
          :width: 100%

          Output: RGB → YUV (BT.709) → RGB

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.advcvtcolor`
     - Convert between RGB and YUV color spaces with a selectable color specification

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save the color-converted image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - GPU-accelerated image resizing
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
