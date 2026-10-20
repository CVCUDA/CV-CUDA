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

.. _sample_normalize:

Normalize
=========

Overview
--------

The Normalize sample demonstrates per-channel mean-and-standard-deviation normalization
using CV-CUDA's GPU-accelerated normalize operator.  The sample applies the standard
ImageNet statistics (mean ``[123.675, 116.28, 103.53]`` and std ``[58.395, 57.12, 57.375]``
expressed in [0, 255] space) to an RGB image, producing float32 normalized values.
Because the normalized output is not directly viewable as a JPEG, the sample linearly
rescales the result back to the [0, 255] uint8 range before saving.

Usage
-----

Basic Usage
^^^^^^^^^^^

Normalize an image using the default ImageNet statistics:

.. code-block:: bash

   python3 normalize.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify a custom output file:

.. code-block:: bash

   python3 normalize.py -i input.jpg -o my_normalized.jpg

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
     - cvcuda/.cache/cat_normalize.jpg
     - Output image file path

Implementation
--------------

Setup: Mean and Std Tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/normalize.py
   :language: python
   :start-after: docs_tag: begin_normalize_setup
   :end-before: docs_tag: end_normalize_setup
   :dedent:

Normalize Operator Call
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/normalize.py
   :language: python
   :start-after: docs_tag: begin_normalize
   :end-before: docs_tag: end_normalize
   :dedent:

Key points:

1. **base and scale tensors**: Broadcast-shaped ``(1, 1, 3)`` HWC tensors holding per-channel mean
   and standard deviation values; the operator broadcasts them across all pixels automatically.
2. **SCALE_IS_STDDEV flag**: Tells the operator that the ``scale`` argument is a standard deviation
   rather than a raw scaling factor, so it computes ``out = (src - base) / (scale + epsilon)``.
3. **Float32 input requirement**: Passing a float32 source keeps the output in float32 so the
   normalized values retain their signed range; a uint8 source would clamp the result back to uint8.
4. **epsilon**: A small regularizer added to the denominator, preventing division by zero when the
   standard deviation is near zero.
5. **Visualization rescaling**: The normalized output typically falls in ``[-2, 2]``.  The sample
   min-max rescales that range back to ``[0, 255]`` for JPEG encoding.

Expected Output
^^^^^^^^^^^^^^^

The output shows the pixel distribution shifted and scaled by the ImageNet statistics,
then remapped to uint8 for viewing:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_normalize.jpg
          :width: 100%

          Output: ImageNet-normalized (rescaled to uint8 for display)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.normalize`
     - Apply per-channel mean-std normalization to a tensor

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save normalized image
* ``cuda_memcpy_h2d`` / ``cuda_memcpy_d2h`` - Transfer base/scale parameters and results between host and device

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images with GPU acceleration
* :ref:`Common Utilities <sample_common>` - Helper functions used in this sample
