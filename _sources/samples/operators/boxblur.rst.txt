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

.. _sample_boxblur:

Box Blur
========

Overview
--------

The Box Blur sample demonstrates selective region blurring using CV-CUDA's GPU-accelerated
box blur operator.  Rather than blurring the entire image, the operator accepts a list of
``BlurBoxI`` rectangles per image in the batch and applies a mean (box) filter only inside
those regions.  Pixels outside the declared boxes are copied through unchanged, making the
operator ideal for privacy redaction, watermark concealment, and artistic effects.

Usage
-----

Basic Usage
^^^^^^^^^^^

Blur three rectangular regions of the default input image:

.. code-block:: bash

   python3 boxblur.py -i input.jpg

Custom Input and Output
^^^^^^^^^^^^^^^^^^^^^^^

Specify input and output paths explicitly:

.. code-block:: bash

   python3 boxblur.py -i image.jpg -o cat_boxblur.jpg

Command-Line Arguments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 20 45

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
     - cvcuda/.cache/cat_boxblur.jpg
     - Output image file path

Implementation
--------------

Box Blur on Selected Regions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/boxblur.py
   :language: python
   :start-after: docs_tag: begin_boxblur
   :end-before: docs_tag: end_boxblur
   :dedent:

Key points:

1. **Batch dimension required**: ``cvcuda.boxblur`` expects NHWC input; use
   ``cvcuda.stack([hwc_image])`` to add a leading batch dimension before calling the operator.
2. **BlurBoxesI structure**: One inner list of ``BlurBoxI`` objects per image in the batch;
   each box is ``(x, y, width, height)`` in pixel coordinates plus a ``kernelSize`` for the
   square mean filter.
3. **Selective blurring**: Only the pixels inside each declared rectangle are filtered; all
   other pixels are passed through untouched.
4. **Kernel size trade-off**: Larger ``kernelSize`` produces stronger, more noticeable blur
   at the cost of slightly more compute; the kernel must be odd and at least 1.
5. **Layout restoration**: After blurring, ``reshape(shape[1:], "HWC")`` strips the batch
   dimension so the result can be saved directly with ``write_image``.

Expected Output
^^^^^^^^^^^^^^^

The output image is identical to the input except for three blurred rectangles: a moderate
patch in the upper-left, a strong central blur, and a light blur in the lower-right corner.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_boxblur.jpg
          :width: 100%

          Output: Selective Box Blur Applied

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.boxblur`
     - Apply a mean (box) filter to user-defined rectangular regions within an image

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save the blurred result image
* :ref:`parse_image_args() <common_parse_image_args>` - Parse ``--input`` / ``--output`` CLI arguments

See Also
--------

* :ref:`Resize Operator <sample_resize>` - GPU-accelerated image resize
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
