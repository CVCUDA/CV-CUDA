..
   # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _sample_resize:

Resize
======

Overview
--------

The Resize sample demonstrates image resizing using CV-CUDA's GPU-accelerated resize operator.

Usage
-----

Basic Usage
^^^^^^^^^^^

Resize an image to 224×224 (default):

.. code-block:: bash

   python3 resize.py -i input.jpg

Custom Dimensions
^^^^^^^^^^^^^^^^^

Specify target width and height:

.. code-block:: bash

   python3 resize.py -i image.jpg -o resized.jpg --width 512 --height 512

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
     - cvcuda/.cache/cat_resized.jpg
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

Single Image Resize
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/resize.py
   :language: python
   :start-after: docs_tag: begin_resize
   :end-before: docs_tag: end_resize
   :dedent:

Key points:

1. **Output Shape**: Must match input dimensions (H, W, C)
2. **Interpolation**: Default is linear (bilinear)
3. **Aspect Ratio**: Not preserved by default

Interpolation Methods
^^^^^^^^^^^^^^^^^^^^^

Available interpolation methods: ``LINEAR`` (default, good balance), ``NEAREST`` (fastest), ``CUBIC`` (highest quality), ``AREA`` (best for downscaling). Specify with :pydata:`cvcuda.Interp.LINEAR`, etc.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image resized to the target dimensions (default 224×224):

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_resized.jpg
          :width: 100%

          Output: Resized to 224×224

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.resize`
     - Resize images to target dimensions with specified interpolation

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save resized image

See Also
--------

* :ref:`Hello World Sample <sample_hello_world>` - Uses resize in pipeline
* :ref:`Classification Sample <sample_classification>` - Resizes for model input
* :ref:`Reformat Operator <sample_reformat>` - Change tensor layouts
* :ref:`Common Utilities <sample_common>` - Helper functions
