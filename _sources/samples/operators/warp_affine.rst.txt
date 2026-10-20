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

.. _sample_warp_affine:

Warp Affine
===========

Overview
--------

The Warp Affine sample demonstrates GPU-accelerated affine image transformation using CV-CUDA's
``warp_affine`` operator. The sample builds a 2×3 float32 transformation matrix that rotates the
image 15 degrees counter-clockwise about its centre and shifts it slightly to the right, then
applies it with bilinear interpolation and constant-value border filling.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply the default rotation+translation warp to an image:

.. code-block:: bash

   python3 warp_affine.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Write the warped result to a specific location:

.. code-block:: bash

   python3 warp_affine.py -i input.jpg -o warped.jpg

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
     - cvcuda/.cache/cat_warp_affine.jpg
     - Output image file path

Implementation
--------------

Affine Matrix Setup
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/warp_affine.py
   :language: python
   :start-after: docs_tag: begin_warp_affine_setup
   :end-before: docs_tag: end_warp_affine_setup
   :dedent:

Warp Affine Call
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/warp_affine.py
   :language: python
   :start-after: docs_tag: begin_warp_affine
   :end-before: docs_tag: end_warp_affine
   :dedent:

Key points:

1. **2×3 Matrix**: ``xform`` is a ``np.float32`` array of shape ``(2, 3)`` that encodes the full
   affine map (rotation, scale, shear, translation) in one compact structure.
2. **Centre-relative rotation**: Translating to the image centre before rotating avoids the image
   drifting off-canvas; the standard formula embeds the centre correction directly in the
   translation column of the matrix.
3. **Interpolation flag**: ``cvcuda.Interp.LINEAR`` gives smooth bilinear interpolation;
   ``NEAREST`` is faster but produces aliasing artefacts on smooth gradients.
4. **Border handling**: ``cvcuda.Border.CONSTANT`` with ``border_value=[0]`` fills any pixels that
   map outside the source image with black — useful for preserving the original framing.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image rotated 15 degrees counter-clockwise with a small rightward translation:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_warp_affine.jpg
          :width: 100%

          Output: Rotated 15° and shifted right

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.warp_affine`
     - Apply a 2×3 affine transformation matrix to an image with configurable interpolation and border handling

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save warped image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - GPU-accelerated image resize
* :ref:`Common Utilities <sample_common>` - Helper functions
