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

.. _sample_remap:

Remap
=====

Overview
--------

The Remap sample demonstrates GPU-accelerated pixel remapping using CV-CUDA's ``remap`` operator.
A coordinate map is built on the CPU with NumPy — a sinusoidal wave-distortion field — then
uploaded to the GPU and applied to the source image.  The result is a ripple-distorted version of
the input that is saved as a viewable JPEG.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply the default wave distortion to the built-in test image:

.. code-block:: bash

   python3 remap.py

Custom Input
^^^^^^^^^^^^

Remap a custom source image and write to a custom output path:

.. code-block:: bash

   python3 remap.py -i input.jpg -o cat_remap.jpg

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
     - cvcuda/.cache/cat_remap.jpg
     - Output image file path

Implementation
--------------

Building the Displacement Map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/remap.py
   :language: python
   :start-after: docs_tag: begin_remap_setup
   :end-before: docs_tag: end_remap_setup
   :dedent:

Applying the Remap Operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/remap.py
   :language: python
   :start-after: docs_tag: begin_remap
   :end-before: docs_tag: end_remap
   :dedent:

Key points:

1. **Map tensor shape**: The coordinate map uses shape ``(H, W, 2)`` with dtype ``F32`` — two
   float channels storing ``[src_x, src_y]`` absolute source coordinates per output pixel.
2. **Map type — ABSOLUTE**: ``cvcuda.Remap.ABSOLUTE`` means each map value is an un-normalized
   ``(x, y)`` pixel coordinate in the source image, giving full control over the displacement.
3. **Source interpolation**: ``src_interp=LINEAR`` smooths the sampled source values for a
   continuous displacement field; ``NEAREST`` is faster when sub-pixel accuracy is not needed.
4. **Border policy**: ``border=REPLICATE`` avoids black edges at the image boundary by repeating
   the nearest border pixel, keeping the output perceptually clean.
5. **Host-to-device upload**: The NumPy map array is transferred to a pre-allocated ``cvcuda.Tensor``
   via ``cuda_memcpy_h2d`` — the same pattern used by other samples that synthesize GPU inputs.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image with a sinusoidal wave distortion applied:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_remap.jpg
          :width: 100%

          Output: Wave-Distorted Image

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.remap`
     - Warp an image using an arbitrary (H, W, 2) coordinate map

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save remapped image
* ``cuda_memcpy_h2d`` - Upload the NumPy coordinate map to the GPU tensor

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Simple spatial scaling
* :ref:`Common Utilities <sample_common>` - Helper functions
