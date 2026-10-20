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

.. _sample_color_twist:

Color Twist
===========

Overview
--------

The Color Twist sample demonstrates per-channel affine color transformation using CV-CUDA's
GPU-accelerated ``color_twist`` operator. A 3×4 float matrix defines how each output channel is
computed as a linear combination of the input channels plus a bias, enabling operations such as
saturation adjustments, color temperature shifts, sepia toning, and general channel mixing.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply the default warm-tint color twist to an image:

.. code-block:: bash

   python3 color_twist.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify a custom output path:

.. code-block:: bash

   python3 color_twist.py -i input.jpg -o cat_color_twist.jpg

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
     - cvcuda/.cache/cat_color_twist.jpg
     - Output image file path

Implementation
--------------

Color Twist Transform
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/color_twist.py
   :language: python
   :start-after: docs_tag: begin_color_twist
   :end-before: docs_tag: end_color_twist
   :dedent:

Key points:

1. **Twist matrix layout**: The twist tensor has shape ``(3, 4)`` with ``"HW"`` layout. Row ``i``
   defines the output for channel ``i`` as ``twist[i,0]*R + twist[i,1]*G + twist[i,2]*B + twist[i,3]``.
2. **Offset column**: The fourth column acts as a per-channel bias (brightness shift), allowing
   independent control of each channel's black point.
3. **Automatic clipping**: The operator clips results back into the source dtype's representable
   range, so no explicit clamping is needed.
4. **Batch support**: Pass an ``NHWC`` tensor or an ``ImageBatchVarShape`` to process a whole batch
   in one GPU kernel launch.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image with a warm golden-hour tint (red boosted, blue slightly reduced):

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_color_twist.jpg
          :width: 100%

          Output: Warm Color Twist Applied

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.color_twist`
     - Apply a 3×4 per-channel affine color transform to every pixel

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save color-twisted image
* ``cuda_memcpy_h2d`` - Upload the twist matrix from host NumPy array to device tensor

See Also
--------

* :ref:`Resize Operator <sample_resize>` - GPU-accelerated image resizing
* :ref:`Common Utilities <sample_common>` - Helper functions used by all samples
