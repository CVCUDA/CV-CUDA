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

.. _sample_copymakeborder:

Copy Make Border
================

Overview
--------

The Copy Make Border sample demonstrates how to pad an image with a border of configurable
width and fill style using CV-CUDA's GPU-accelerated ``copymakeborder`` operator.  The
operator supports multiple border modes (``CONSTANT``, ``REPLICATE``, ``REFLECT``,
``REFLECT101``, ``WRAP``) and operates directly on device tensors, making it well-suited
for preprocessing pipelines that need to pad images before inference.

Usage
-----

Basic Usage
^^^^^^^^^^^

Add an orange border to the default cat image:

.. code-block:: bash

   python3 copymakeborder.py

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify a custom output file:

.. code-block:: bash

   python3 copymakeborder.py -i input.jpg -o cat_copymakeborder.jpg

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
     - cvcuda/.cache/cat_copymakeborder.jpg
     - Output image file path

Implementation
--------------

Adding a Colored Border
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/copymakeborder.py
   :language: python
   :start-after: docs_tag: begin_copymakeborder
   :end-before: docs_tag: end_copymakeborder
   :dedent:

Key points:

1. **Border mode**: ``cvcuda.Border.CONSTANT`` fills the added region with a fixed color;
   other modes (``REPLICATE``, ``REFLECT``, ``REFLECT101``, ``WRAP``) derive fill values
   from existing image pixels instead.
2. **Border value**: A three-element list ``[R, G, B]`` supplies the fill color for
   ``CONSTANT`` mode; it is silently ignored for the other modes.
3. **Output shape**: The output tensor is automatically sized to
   ``(H + top + bottom, W + left + right, C)``; no pre-allocation is required.
4. **Asymmetric padding**: ``top``, ``bottom``, ``left``, and ``right`` are independent
   integers, enabling padding that differs on each side — useful for letterboxing.

Expected Output
^^^^^^^^^^^^^^^

The output shows the original image surrounded by a 30-pixel vertical and 60-pixel
horizontal orange border:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_copymakeborder.jpg
          :width: 100%

          Output: Image with orange border padding

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.copymakeborder`
     - Pad an image with a configurable border width and fill mode

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save bordered image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images to target dimensions
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
