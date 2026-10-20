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

.. _sample_random_resized_crop:

Random Resized Crop
===================

Overview
--------

The Random Resized Crop sample demonstrates how to use CV-CUDA's GPU-accelerated
:py:func:`cvcuda.random_resized_crop` operator to randomly select a sub-region of an
image, resize it to a fixed output size, and write the result.  This operation is the
core augmentation used in standard ImageNet training pipelines (e.g., torchvision's
``RandomResizedCrop``): a crop whose area is a random fraction of the original image
area and whose aspect ratio is sampled from a configurable range.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply random resized crop with default 224×224 output:

.. code-block:: bash

   python3 random_resized_crop.py -i input.jpg

Custom Output Size
^^^^^^^^^^^^^^^^^^

Specify a different target resolution:

.. code-block:: bash

   python3 random_resized_crop.py -i input.jpg -o cropped.jpg --width 320 --height 320

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
     - cvcuda/.cache/cat_random_resized_crop.jpg
     - Output image file path
   * - ``--width``
     -
     - 224
     - Target output width in pixels
   * - ``--height``
     -
     - 224
     - Target output height in pixels

Implementation
--------------

Random Resized Crop
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/random_resized_crop.py
   :language: python
   :start-after: docs_tag: begin_random_resized_crop
   :end-before: docs_tag: end_random_resized_crop
   :dedent:

Key points:

1. **Batched NHWC input**: The operator expects an NHWC tensor (batch dimension first).
   A single HWC image is reshaped to ``(1, H, W, C)`` before the call and the result is
   reshaped back to ``(H, W, C)`` afterwards.
2. **Scale bounds**: ``min_scale`` and ``max_scale`` control what fraction of the
   original image area the random crop covers.  The defaults ``(0.08, 1.0)`` match
   standard ImageNet pre-processing.
3. **Ratio bounds**: ``min_ratio`` and ``max_ratio`` bound the width-to-height ratio of
   the crop region before it is scaled to the output size, letting the network see both
   tall and wide crops.
4. **Interpolation**: ``cvcuda.Interp.LINEAR`` (bilinear) gives a good quality/speed
   trade-off; ``NEAREST`` is faster, ``CUBIC`` provides higher fidelity.
5. **Reproducibility**: The ``seed`` parameter makes the crop deterministic, which is
   useful for debugging or ablation experiments.

Expected Output
^^^^^^^^^^^^^^^

The output shows a randomly selected and resized crop of the original image:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_random_resized_crop.jpg
          :width: 100%

          Output: Random Resized Crop to 224×224

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.random_resized_crop`
     - Randomly crop a sub-region of the image and resize it to the target dimensions

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save the cropped and resized image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Deterministic resize to fixed dimensions
* :ref:`Common Utilities <sample_common>` - Helper functions
