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

.. _sample_inpaint:

Inpaint
=======

Overview
--------

The Inpaint sample demonstrates image inpainting using CV-CUDA's GPU-accelerated inpaint operator.
Inpainting reconstructs the pixel values inside a user-supplied mask region by propagating colour
information from the surrounding unmasked pixels.  The sample simulates salt-and-pepper sensor
noise by randomly zeroing ~15 % of pixels, then uses inpainting to remove the noise and restore
the image.

Usage
-----

Basic Usage
^^^^^^^^^^^

Inpaint the default tabby-cat image:

.. code-block:: bash

   python3 inpaint.py -i input.jpg

Custom Input and Output
^^^^^^^^^^^^^^^^^^^^^^^

Specify a custom input image and output path:

.. code-block:: bash

   python3 inpaint.py -i image.jpg -o cat_inpaint.jpg

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
     - cvcuda/.cache/cat_inpaint.jpg
     - Output image file path

Implementation
--------------

Inpainting and Outside-Mask Restoration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/inpaint.py
   :language: python
   :start-after: docs_tag: begin_inpaint
   :end-before: docs_tag: end_inpaint
   :dedent:

Key points:

1. **Mask format**: The mask must be a single-channel (``NHWC`` with ``C=1``) ``U8`` tensor.
   Non-zero pixels mark the region to be reconstructed; zero pixels are left unchanged.
2. **Batched input**: ``cvcuda.inpaint`` requires an ``NHWC`` (batched) source tensor.
   A plain ``HWC`` image is reshaped to ``(1, H, W, C)`` before the call.
3. **inpaintRadius**: Controls the neighbourhood radius examined when reconstructing each
   masked pixel.  Larger values smooth over wider damaged areas at the cost of more
   computation.
4. **Outside-mask restoration**: The operator may alter pixels just outside the mask
   boundary, so the result is downloaded and the original content is restored everywhere
   outside the mask (via ``np.where``) before the final image is uploaded and saved.
5. **Synthetic mask via upload_tensor**: The mask is built as a NumPy array on the CPU and
   then uploaded to a pre-allocated GPU tensor with ``upload_tensor``, matching the pattern
   used whenever host-side parameter data must be passed as a tensor.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image with salt-and-pepper noise removed by inpainting:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/cat_inpaint_damaged.jpg
          :width: 100%

          Input: Image with simulated salt-and-pepper sensor noise (~15 % pixels zeroed)

     - .. figure:: ../../content/cat_inpaint.jpg
          :width: 100%

          Output: Noise removed by inpainting

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.inpaint`
     - Reconstruct masked pixel regions using surrounding colour information

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save inpainted image
* ``upload_tensor`` - Upload the CPU-built mask and result arrays to GPU tensors
* ``download_tensor`` - Download tensors to the CPU for damage synthesis and restoration

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Simple single-operator sample
* :ref:`Label Operator <sample_label>` - Another sample that synthesises inputs and uploads via cuda_memcpy_h2d
* :ref:`Common Utilities <sample_common>` - Helper functions
