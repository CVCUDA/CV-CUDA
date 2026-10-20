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

.. _sample_composite:

Composite
=========

Overview
--------

The Composite sample demonstrates GPU-accelerated image compositing with CV-CUDA.
It blends a foreground image (tabby cat) over a background image (Weimaraner dog)
using a single-channel alpha mask.  A filled circle in the centre of the frame
keeps the cat visible while the dog shows through outside the circle, making the
blend immediately obvious in the output image.

Usage
-----

Basic Usage
^^^^^^^^^^^

Composite the default cat foreground over the Weimaraner background:

.. code-block:: bash

   python3 composite.py -i input.jpg

Custom Input
^^^^^^^^^^^^

Supply your own foreground image and redirect the output:

.. code-block:: bash

   python3 composite.py -i image.jpg -o cat_composite.jpg

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
     - Foreground input image file path
   * - ``--output``
     - ``-o``
     - cvcuda/.cache/cat_composite.jpg
     - Output composited image file path

Implementation
--------------

Composite Operation
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/composite.py
   :language: python
   :start-after: docs_tag: begin_composite
   :end-before: docs_tag: end_composite
   :dedent:

Key points:

1. **Mask shape**: The foreground mask must be single-channel (``HWC`` with ``C=1``), uint8.
2. **Spatial alignment**: Foreground, background, and mask must share the same ``(H, W)`` dimensions; the background is resized to match the foreground before compositing.
3. **outchannels parameter**: Pass ``3`` for an RGB output tensor or ``4`` for RGBA.
4. **Mask semantics**: Pixel values ``> 0`` select the foreground; ``0`` selects the background — effectively a hard binary blend.
5. **GPU upload**: The mask is constructed on the CPU with NumPy then transferred to the GPU via ``cuda_memcpy_h2d`` before the operator call.

Expected Output
^^^^^^^^^^^^^^^

The composited image shows the cat inside a circular region with the Weimaraner
dog visible outside it:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image (foreground)

     - .. figure:: ../../content/cat_composite.jpg
          :width: 100%

          Output: Cat composited over Weimaraner

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.composite`
     - Blend foreground and background images using an alpha mask
   * - :py:func:`cvcuda.resize`
     - Resize the background to match foreground spatial dimensions

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load images as CV-CUDA tensors
* :ref:`write_image() <common_write_image>` - Save the composited image
* :ref:`cuda_memcpy_h2d() <common_cuda_memcpy_h2d>` - Upload the NumPy mask to the GPU

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images to target dimensions
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
