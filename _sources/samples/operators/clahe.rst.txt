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

.. _sample_clahe:

CLAHE
=====

Overview
--------

The CLAHE sample demonstrates Contrast Limited Adaptive Histogram Equalization using CV-CUDA's
GPU-accelerated ``cvcuda.clahe`` operator. CLAHE improves local contrast by equalizing the
histogram of small contextual tiles independently, then clipping the amplification to a
user-supplied limit to suppress noise amplification. The result is a perceptually clearer image
without the over-saturation that can occur with global histogram equalization.

Because ``cvcuda.clahe`` requires a single-channel (grayscale) ``U8`` tensor, the sample first
converts the RGB input to grayscale, applies CLAHE, and then replicates the enhanced grayscale
channel across R, G, and B for a viewable output image.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply CLAHE to an image (default ``clip_limit=2.0``, ``tile_grid_size=(8, 8)``):

.. code-block:: bash

   python3 clahe.py -i input.jpg

Custom Example
^^^^^^^^^^^^^^

Specify a custom input and output path:

.. code-block:: bash

   python3 clahe.py -i input.jpg -o cat_clahe.jpg

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
     - cvcuda/.cache/cat_clahe.jpg
     - Output image file path

Implementation
--------------

CLAHE Operator
^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/clahe.py
   :language: python
   :start-after: docs_tag: begin_clahe
   :end-before: docs_tag: end_clahe
   :dedent:

Key points:

1. **Grayscale requirement**: ``cvcuda.clahe`` only accepts single-channel ``U8`` tensors; RGB
   inputs must first be converted with ``cvcuda.cvtcolor(src, cvcuda.ColorConversion.RGB2GRAY)``.
2. **clip_limit**: Values above 1.0 enable contrast limiting; the default of 2.0 provides
   moderate enhancement while suppressing noise. Setting it to 0.0 raises an exception.
3. **tile_grid_size**: The tuple ``(cols, rows)`` of contextual tiles; each tile must be at
   least 1×1. Larger grids produce more localised adaptation at the cost of extra computation.
4. **Stream support**: An optional ``stream`` keyword enables asynchronous GPU execution; call
   ``stream.sync()`` before reading results back to the host.
5. **Batch support**: The operator accepts both ``HWC`` (single image) and ``NHWC`` (batch)
   tensors as well as variable-shape image batches (``ImageBatchVarShape``).

Expected Output
^^^^^^^^^^^^^^^

The output shows the grayscale-enhanced image saved as a three-channel (RGB) JPEG for
viewer compatibility:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_clahe.jpg
          :width: 100%

          Output: CLAHE-enhanced grayscale (replicated to RGB)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.clahe`
     - Contrast Limited Adaptive Histogram Equalization on a grayscale tensor
   * - :py:func:`cvcuda.cvtcolor`
     - Convert RGB input image to single-channel grayscale before CLAHE
   * - :py:func:`cvcuda.stack`
     - Stack the HWC input tensor into an NHWC batch for ``cvtcolor``

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save CLAHE-enhanced image
* ``cuda_memcpy_d2h`` - Download CLAHE result to host for grayscale-to-RGB replication
* ``cuda_memcpy_h2d`` - Upload the replicated RGB array back to a CVCUDA tensor

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Simple spatial transformation example
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
