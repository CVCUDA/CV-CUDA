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

.. _sample_erase:

Erase
=====

Overview
--------

The Erase sample demonstrates how to fill one or more rectangular regions of an image with
solid colours using CV-CUDA's GPU-accelerated erase operator.  Six rectangular regions
(red, green, blue, white, and black solid fills plus one green-channel-only tint) are
stamped onto the image entirely on the GPU — no round-trip to the CPU is needed for the
pixel data.

Usage
-----

Basic Usage
^^^^^^^^^^^

Erase six rectangular regions into the default tabby-cat image:

.. code-block:: bash

   python3 erase.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Save the result to a specific file:

.. code-block:: bash

   python3 erase.py -i input.jpg -o erased.jpg

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
     - cvcuda/.cache/cat_erase.jpg
     - Output image file path

Implementation
--------------

Parameter Tensor Setup
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/erase.py
   :language: python
   :start-after: docs_tag: begin_erase_setup
   :end-before: docs_tag: end_erase_setup
   :dedent:

Erase Call
^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/erase.py
   :language: python
   :start-after: docs_tag: begin_erase
   :end-before: docs_tag: end_erase
   :dedent:

Key points:

1. **Parameter tensors**: ``anchor``, ``erasing``, ``values``, and ``imgIdx`` are small
   1-D tensors built from NumPy arrays and uploaded to the GPU with ``cuda_memcpy_h2d``.
2. **Batch dimension**: The operator expects NHWC input, so a single HWC image is wrapped
   in a batch of size 1 via ``reshape``.
3. **anchor** holds ``(x, y)`` pixel coordinates of each rectangle's top-left corner
   (type ``_2S32`` — a pair of int32 per element).
4. **erasing** holds ``(width, height, flag)`` per rectangle; ``flag`` is a channel
   bitmask (bit0=R, bit1=G, bit2=B) selecting which channels are overwritten by the
   ``values`` fill.  ``flag=7`` (``0b111``) replaces all three channels for a solid fill,
   while ``flag=2`` (``0b010``) replaces only the green channel, leaving R and B intact
   for a tint.
5. **random mode**: Setting ``random=True`` ignores ``values`` and fills each rectangle
   with deterministic pseudo-random noise controlled by ``seed``.

Expected Output
^^^^^^^^^^^^^^^

The output image is identical to the input except for six erased regions — red, green,
blue, white, and black solid rectangles plus one green-channel-only tint:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_erase.jpg
          :width: 100%

          Output: Six rectangular regions erased from the image

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.erase`
     - Fill rectangular regions with solid colours or pseudo-random noise

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save erased image
* ``cuda_memcpy_h2d`` - Upload NumPy parameter arrays to the GPU

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic single-operator sample structure
* :ref:`Common Utilities <sample_common>` - Helper functions
