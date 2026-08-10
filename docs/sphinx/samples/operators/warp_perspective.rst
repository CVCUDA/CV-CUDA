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

.. _sample_warp_perspective:

Warp Perspective
================

Overview
--------

The Warp Perspective sample demonstrates GPU-accelerated perspective transform using CV-CUDA's
``warp_perspective`` operator. A 3×3 homography matrix maps every destination pixel back to its
source location, enabling keystone correction, bird's-eye-view synthesis, and other projective
geometry tasks.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply a default mild-keystone perspective transform:

.. code-block:: bash

   python3 warp_perspective.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Save the warped result to a specific file:

.. code-block:: bash

   python3 warp_perspective.py -i input.jpg -o cat_warp_perspective.jpg

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
     - cvcuda/.cache/cat_warp_perspective.jpg
     - Output image file path

Implementation
--------------

Perspective Matrix Setup
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/warp_perspective.py
   :language: python
   :start-after: docs_tag: begin_warp_perspective_setup
   :end-before: docs_tag: end_warp_perspective_setup
   :dedent:

Warp Perspective Call
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/warp_perspective.py
   :language: python
   :start-after: docs_tag: begin_warp_perspective
   :end-before: docs_tag: end_warp_perspective
   :dedent:

Key points:

1. **3×3 float32 matrix**: ``warp_perspective`` expects a 3×3 homography matrix, either as a
   nested Python list or a float32 NumPy array. The matrix relates homogeneous destination
   coordinates to homogeneous source coordinates.
2. **WARP_INVERSE_MAP flag**: When this flag is combined with the interpolation mode the matrix
   is interpreted as a destination→source mapping, which is how the standard DLT construction
   works. Without the flag the operator inverts the matrix internally.
3. **Border mode**: ``cvcuda.Border.CONSTANT`` fills pixels that map outside the source image
   with the ``border_value``; ``REPLICATE`` and ``WRAP`` are also supported.
4. **Batch support**: Pass an ``ImageBatch`` and a ``(N, 9)`` float32 transform tensor to apply
   per-image perspective matrices in a single call.

Expected Output
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_warp_perspective.jpg
          :width: 100%

          Output: Perspective-warped image

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.warp_perspective`
     - Apply a 3×3 homography perspective transform to an image

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save perspective-warped image
* :ref:`parse_image_args() <common_parse_image_args>` - Parse ``--input`` / ``--output`` CLI arguments

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Simple geometric scaling
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
