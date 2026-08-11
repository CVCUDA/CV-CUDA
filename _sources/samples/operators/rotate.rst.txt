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

.. _sample_rotate:

Rotate
======

Overview
--------

The Rotate sample demonstrates GPU-accelerated image rotation using CV-CUDA's rotate operator.
It reads an input image, computes a centring shift so the rotated content stays visible, and
writes the result as a standard uint8 JPEG.

Usage
-----

Basic Usage
^^^^^^^^^^^

Rotate the default tabby-cat image by 45 degrees:

.. code-block:: bash

   python3 rotate.py

Custom Input
^^^^^^^^^^^^

Rotate a specific image and save to a custom output path:

.. code-block:: bash

   python3 rotate.py -i input.jpg -o cat_rotate.jpg

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
     - cvcuda/.cache/cat_rotate.jpg
     - Output image file path

Implementation
--------------

Centred Rotation
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/rotate.py
   :language: python
   :start-after: docs_tag: begin_rotate
   :end-before: docs_tag: end_rotate
   :dedent:

Key points:

1. **Rotation origin**: ``cvcuda.rotate`` rotates around the top-left corner, so a compensating
   translation shift must be provided to keep the image content centred.
2. **Centring shift**: The shift ``(cx - cx*cos - cy*sin, cy - cy*cos + cx*sin)`` is derived from
   the standard 2-D rotation-about-centre formula.
3. **Interpolation**: ``cvcuda.Interp.LINEAR`` gives smooth results; ``NEAREST`` is faster and
   ``CUBIC`` provides higher quality at the cost of more computation.
4. **Output shape and dtype**: The output tensor has the same spatial dimensions and data type as
   the input — no host-side conversion is needed.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image rotated 45 degrees with the content kept centred:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_rotate.jpg
          :width: 100%

          Output: Rotated 45 degrees (centred)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.rotate`
     - Rotate an image by an arbitrary angle with configurable interpolation

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save rotated image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images to target dimensions
* :ref:`Warp Affine Operator <sample_warp_affine>` - General affine transformations
* :ref:`Common Utilities <sample_common>` - Helper functions
