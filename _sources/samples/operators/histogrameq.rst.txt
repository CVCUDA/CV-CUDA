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

.. _sample_histogrameq:

Histogram Equalization
======================

Overview
--------

The Histogram Equalization sample demonstrates GPU-accelerated contrast enhancement using
CV-CUDA's ``histogrameq`` operator. The operator redistributes pixel intensities so the
cumulative histogram of the output image is approximately uniform, improving global contrast
without any parameter tuning.

Usage
-----

Basic Usage
^^^^^^^^^^^

Equalize an image with the default input:

.. code-block:: bash

   python3 histogrameq.py

Custom Input
^^^^^^^^^^^^

Specify a custom input and output path:

.. code-block:: bash

   python3 histogrameq.py -i input.jpg -o cat_histogrameq.jpg

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
     - cvcuda/.cache/cat_histogrameq.jpg
     - Output image file path

Implementation
--------------

Histogram Equalization
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/histogrameq.py
   :language: python
   :start-after: docs_tag: begin_histogrameq
   :end-before: docs_tag: end_histogrameq
   :dedent:

Key points:

1. **Grayscale conversion**: ``cvcuda.cvtcolor`` with ``RGB2GRAY`` is applied first because histogram equalization is most meaningful on a single luminance channel.
2. **Batched NHWC layout**: The HWC image is wrapped in a batch dimension via ``cvcuda.stack`` so the ``cvtcolor`` operator (which expects NHWC) can be used directly.
3. **dtype keyword**: ``cvcuda.histogrameq`` requires an explicit ``dtype`` argument when operating on a ``Tensor``; for image-batch inputs the argument is optional.
4. **Host-side channel replication**: The equalized single-channel output is downloaded, tiled to three channels on the CPU, and re-uploaded as an HWC tensor so ``write_image`` can encode a standard JPEG.
5. **Zero-copy back-path**: ``cuda_memcpy_h2d`` and ``cuda_memcpy_d2h`` avoid any Python-level buffer copies beyond the mandatory host round-trip needed for channel replication.

Expected Output
^^^^^^^^^^^^^^^

The output shows the original image converted to grayscale with equalized contrast:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_histogrameq.jpg
          :width: 100%

          Output: Histogram-Equalized Grayscale

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.histogrameq`
     - Equalize pixel-intensity histogram to enhance global contrast
   * - :py:func:`cvcuda.cvtcolor`
     - Convert RGB image to single-channel grayscale before equalization
   * - :py:func:`cvcuda.stack`
     - Wrap a single HWC tensor into an NHWC batch for ``cvtcolor``

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save equalized image
* ``cuda_memcpy_d2h`` - Download equalized tensor to NumPy for channel replication
* ``cuda_memcpy_h2d`` - Upload replicated RGB tensor back to GPU for encoding

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic image transformation example
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
