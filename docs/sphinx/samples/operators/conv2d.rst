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

.. _sample_conv2d:

Conv2D
======

Overview
--------

The Conv2D sample demonstrates GPU-accelerated 2-D convolution using CV-CUDA's
``conv2d`` operator.  The sample wraps a single RGB image in an
``ImageBatchVarShape``, constructs a 3×3 sharpening kernel as a float
``ImageBatchVarShape``, and runs the convolution on the GPU.  Per-image kernel
anchors are supplied via a small ``Tensor`` of shape ``(N, 2)``.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply the default sharpening filter to an image:

.. code-block:: bash

   python3 conv2d.py -i input.jpg

Custom Input and Output
^^^^^^^^^^^^^^^^^^^^^^^

Specify both input and output paths:

.. code-block:: bash

   python3 conv2d.py -i image.jpg -o cat_conv2d.jpg

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
     - cvcuda/.cache/cat_conv2d.jpg
     - Output image file path

Implementation
--------------

Batch and Kernel Setup
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/conv2d.py
   :language: python
   :start-after: docs_tag: begin_conv2d_setup
   :end-before: docs_tag: end_conv2d_setup
   :dedent:

Conv2D Operator Call
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/conv2d.py
   :language: python
   :start-after: docs_tag: begin_conv2d
   :end-before: docs_tag: end_conv2d
   :dedent:

Key points:

1. **ImageBatchVarShape input**: ``conv2d`` requires the source image(s) wrapped in an ``ImageBatchVarShape``; individual ``Tensor`` objects must be converted via ``cvcuda.as_image`` first.
2. **Float kernel**: The convolution kernel must use ``cvcuda.Format.F32``; integer kernels are not supported.
3. **Kernel anchor**: A ``Tensor`` of shape ``(N, 2)`` with layout ``"NC"`` provides the ``(x, y)`` anchor for each image; ``(-1, -1)`` selects the kernel centre automatically.
4. **Border mode**: ``REFLECT101`` avoids the dark halo at image edges that ``CONSTANT`` (zero) padding produces when sharpening.
5. **Result extraction**: The output ``ImageBatchVarShape`` is iterated to retrieve each result ``Image``, which is then wrapped back into an HWC ``Tensor`` for saving.

Expected Output
^^^^^^^^^^^^^^^

The output image shows the input with edges and fine detail enhanced by the
sharpening kernel:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_conv2d.jpg
          :width: 100%

          Output: Sharpened with 3×3 kernel

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.conv2d`
     - Apply a per-image 2-D convolution kernel over an ``ImageBatchVarShape``

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save convolved image
* ``cuda_memcpy_h2d`` - Upload NumPy kernel weights and anchor coordinates to the GPU

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Another spatial image operator
* :ref:`Common Utilities <sample_common>` - Helper functions used by all operator samples
