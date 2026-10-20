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

.. _sample_customcrop:

Custom Crop
===========

Overview
--------

The Custom Crop sample demonstrates how to extract an off-center rectangular sub-region
from an image using CV-CUDA's GPU-accelerated custom crop operator.
The crop region is expressed as a :py:class:`cvcuda.RectI` (x, y, width, height) in
input-image pixel coordinates, making it straightforward to implement any region-of-interest
extraction pipeline.

Usage
-----

Basic Usage
^^^^^^^^^^^

Crop the default input image with an automatically computed off-center rectangle:

.. code-block:: bash

   python3 customcrop.py

Custom Input and Output
^^^^^^^^^^^^^^^^^^^^^^^

Specify a custom input file and output path:

.. code-block:: bash

   python3 customcrop.py -i input.jpg -o cat_customcrop.jpg

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
     - cvcuda/.cache/cat_customcrop.jpg
     - Output image file path

Implementation
--------------

Crop Region Setup
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/customcrop.py
   :language: python
   :start-after: docs_tag: begin_customcrop_setup
   :end-before: docs_tag: end_customcrop_setup
   :dedent:

Applying the Custom Crop
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/customcrop.py
   :language: python
   :start-after: docs_tag: begin_customcrop
   :end-before: docs_tag: end_customcrop
   :dedent:

Key points:

1. **RectI coordinates**: ``x`` and ``y`` are the top-left corner of the crop window in the input image; ``width`` and ``height`` define the output dimensions.
2. **Off-center crop**: Choosing ``x = img_w // 5`` and ``y = img_h // 5`` deliberately avoids a centered crop, which is typical for ROI extraction use cases.
3. **Output shape**: The output tensor shape is ``(crop_h, crop_w, channels)`` for an HWC input, matching exactly the rectangle dimensions.
4. **HWC layout preserved**: The operator preserves the input layout (HWC or NHWC), so the result can be passed directly to downstream ops or written with ``write_image``.
5. **Stream support**: An optional ``stream`` keyword argument enables asynchronous execution on a specific CUDA stream.

Expected Output
^^^^^^^^^^^^^^^

The output shows the central 60% of the image, shifted 20% from the top-left corner:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_customcrop.jpg
          :width: 100%

          Output: Off-center Cropped Region

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.customcrop`
     - Extract a rectangular region-of-interest from the input tensor

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save cropped image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images to target dimensions
* :ref:`Common Utilities <sample_common>` - Helper functions
