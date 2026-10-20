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

.. _sample_channelreorder:

Channel Reorder
===============

Overview
--------

The Channel Reorder sample demonstrates per-image channel permutation using CV-CUDA's
GPU-accelerated ``channelreorder`` operator.  It reads an RGB image, wraps it in an
``ImageBatchVarShape``, specifies a ``[2, 1, 0]`` channel-index order to swap R and B
(producing a BGR image), and writes the result back to disk.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply the default RGB → BGR channel swap:

.. code-block:: bash

   python3 channelreorder.py -i input.jpg

Custom Input and Output
^^^^^^^^^^^^^^^^^^^^^^^

Specify custom input and output paths:

.. code-block:: bash

   python3 channelreorder.py -i input.jpg -o cat_channelreorder.jpg

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
     - cvcuda/.cache/cat_channelreorder.jpg
     - Output image file path

Implementation
--------------

Setup: Wrapping the Input Tensor as an ImageBatchVarShape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/channelreorder.py
   :language: python
   :start-after: docs_tag: begin_channelreorder_setup
   :end-before: docs_tag: end_channelreorder_setup
   :dedent:

Channel Reorder Operator Call
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/channelreorder.py
   :language: python
   :start-after: docs_tag: begin_channelreorder
   :end-before: docs_tag: end_channelreorder
   :dedent:

Key points:

1. **Two input containers** — fixed-shape ``Tensor`` inputs take a host order sequence, while
   ``ImageBatchVarShape`` inputs take a device-resident per-image orders tensor.
2. **VarShape orders layout** — the ``orders`` tensor must have layout ``"NC"`` (num_images ×
   num_channels), where each row gives the input-channel index for each output channel.
3. **Tensor order sequence** — ``cvcuda.channelreorder(tensor, [2, 1, 0])`` performs the
   canonical RGB → BGR swap without allocating a device parameter tensor.
4. **Zero-copy wrapping** — ``cvcuda.as_image`` ties the source buffer lifetime to the
   ``Image`` object; no extra device copy is performed.
5. **Result extraction** — iterate over a variable-shape output and call
   ``cvcuda.as_tensor(out_image.cuda(), "HWC")`` to obtain a writable HWC tensor.

Expected Output
^^^^^^^^^^^^^^^

The output image has its red and blue channels swapped relative to the input:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image (RGB)

     - .. figure:: ../../content/cat_channelreorder.jpg
          :width: 100%

          Output: Channels Reordered to BGR

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.channelreorder`
     - Permute image channels according to a per-image index tensor

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save reordered image
* ``cuda_memcpy_h2d`` - Upload the host-side orders array to the GPU

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic GPU image resizing
* :ref:`Common Utilities <sample_common>` - Helper functions
