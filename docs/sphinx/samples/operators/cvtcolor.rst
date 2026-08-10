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

.. _sample_cvtcolor:

Color Conversion
================

Overview
--------

The Color Conversion sample demonstrates GPU-accelerated color space conversion using CV-CUDA's
``cvtcolor`` operator.  The example reads an input image, converts it from RGB to BGR by swapping
the red and blue channels, and writes the result as a viewable uint8 JPEG.  The same operator
supports a wide range of conversions including grayscale, RGBA, HSV, and YUV formats — only the
``code`` argument needs to change.

Usage
-----

Basic Usage
^^^^^^^^^^^

Convert the default tabby-cat image (RGB to BGR):

.. code-block:: bash

   python3 cvtcolor.py

Custom Input and Output
^^^^^^^^^^^^^^^^^^^^^^^

Specify input and output paths explicitly:

.. code-block:: bash

   python3 cvtcolor.py -i input.jpg -o cat_cvtcolor.jpg

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
     - cvcuda/.cache/cat_cvtcolor.jpg
     - Output image file path

Implementation
--------------

Color Space Conversion
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/cvtcolor.py
   :language: python
   :start-after: docs_tag: begin_cvtcolor
   :end-before: docs_tag: end_cvtcolor
   :dedent:

Key points:

1. **Batched input**: ``cvcuda.cvtcolor`` requires an NHWC tensor; a single HWC image is
   promoted to a batch of one with ``cvcuda.stack``.
2. **ColorConversion enum**: The desired conversion is selected by passing a
   :pydata:`cvcuda.ColorConversion` member as the ``code`` keyword argument.
3. **Symmetric channel counts**: The source and destination channel counts must match the
   chosen conversion code (e.g. RGB2BGR keeps 3 channels; BGR2GRAY reduces to 1).
4. **Batch dimension removal**: After conversion the leading batch dimension is dropped with
   ``Tensor.reshape`` so the result is a plain HWC tensor that ``write_image`` can encode
   directly as JPEG.
5. **Supported dtypes**: The operator accepts ``uint8`` and ``uint16`` inputs; the default
   JPEG pipeline uses ``uint8``.

Expected Output
^^^^^^^^^^^^^^^

The output shows the input image with red and blue channels exchanged.  Warm-toned areas
(e.g. orange fur) appear cooler and vice versa:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_cvtcolor.jpg
          :width: 100%

          Output: RGB channels converted to BGR

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.cvtcolor`
     - Convert image between color spaces using a GPU-accelerated kernel

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save color-converted image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic single-image operator example
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
