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

.. _sample_laplacian:

Laplacian
=========

Overview
--------

The Laplacian sample demonstrates second-order edge detection using CV-CUDA's GPU-accelerated
Laplacian operator. The Laplacian highlights regions of rapid intensity change, making it a
classic tool for detecting edges and fine structures in images. After applying the operator the
sample stretches the response histogram to the full uint8 range so the edge map is immediately
viewable as a JPEG.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply the Laplacian operator with the default settings (ksize=3, scale=1.0):

.. code-block:: bash

   python3 laplacian.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify a custom output file path:

.. code-block:: bash

   python3 laplacian.py -i input.jpg -o cat_laplacian.jpg

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
     - cvcuda/.cache/cat_laplacian.jpg
     - Output image file path

Implementation
--------------

Laplacian Edge Detection
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/laplacian.py
   :language: python
   :start-after: docs_tag: begin_laplacian
   :end-before: docs_tag: end_laplacian
   :dedent:

Key points:

1. **Kernel size**: ``ksize=3`` selects the 3×3 discrete Laplacian aperture. The only other
   supported value is ``ksize=1``, which uses a simpler cross-shaped kernel.
2. **Scale factor**: ``scale=1.0`` applies a uniform multiplier to the computed Laplacian
   values before the result is saturated back to the input dtype. Increasing the scale
   amplifies weaker edges.
3. **Border handling**: ``cvcuda.Border.REPLICATE`` repeats the edge pixels outward,
   avoiding the zero-filled boundary artifacts that ``CONSTANT`` mode would introduce.
4. **Dtype preservation**: The operator returns a tensor with the same dtype and layout as
   the input (uint8 HWC here), so no format conversion is required.
5. **Histogram stretching**: The raw Laplacian response is typically concentrated in a
   narrow value range. A host-side min/max stretch makes the edge map clearly visible in
   the saved image without changing the operator's output semantics.

Expected Output
^^^^^^^^^^^^^^^

The output shows the Laplacian edge response of the input image, normalized for visibility:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_laplacian.jpg
          :width: 100%

          Output: Laplacian Edge Response

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.laplacian`
     - Apply second-order Laplacian edge-detection filter to the input image

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save the edge-response image
* ``cuda_memcpy_d2h`` / ``cuda_memcpy_h2d`` - Transfer tensor data to/from host for histogram stretching

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images with GPU acceleration
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
