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

.. _sample_joint_bilateral_filter:

Joint Bilateral Filter
======================

Overview
--------

The Joint Bilateral Filter sample demonstrates edge-preserving image smoothing using CV-CUDA's
GPU-accelerated joint bilateral filter operator. Unlike the standard bilateral filter, the joint
(cross) variant uses a separate *guidance* image to steer the range kernel — edges detected in
the guidance image are preserved in the filtered output, making it well-suited for noise
reduction while retaining sharp structural boundaries.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply joint bilateral filtering to an image using the default input:

.. code-block:: bash

   python3 joint_bilateral_filter.py -i input.jpg

Custom Output
^^^^^^^^^^^^^

Specify a custom output path:

.. code-block:: bash

   python3 joint_bilateral_filter.py -i input.jpg -o cat_joint_bilateral_filter.jpg

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
     - cvcuda/.cache/cat_joint_bilateral_filter.jpg
     - Output image file path

Implementation
--------------

Joint Bilateral Filter
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/joint_bilateral_filter.py
   :language: python
   :start-after: docs_tag: begin_joint_bilateral_filter
   :end-before: docs_tag: end_joint_bilateral_filter
   :dedent:

Key points:

1. **Guidance image**: A grayscale-derived image is used as the guidance signal so that luminance
   edges govern which pixels are blended — colour-channel noise is reduced without crossing structural
   boundaries.
2. **Channel matching**: ``srcColor`` must have the same spatial size and channel count as ``src``;
   the grayscale result is converted back to 3-channel RGB before being passed as guidance.
3. **diameter**: Controls the neighbourhood size; larger values consider farther pixels but increase
   cost quadratically.
4. **sigma_color / sigma_space**: Larger values produce stronger smoothing; ``sigma_color`` governs
   how different colours can still be blended, ``sigma_space`` governs spatial reach.
5. **Batch dimension**: The HWC tensor read from disk is reshaped to NHWC for ``cvtcolor`` and the
   filter call, then reshaped back to HWC before writing the output.

Expected Output
^^^^^^^^^^^^^^^

The output is a smoothed version of the input with fine texture noise reduced while prominent
edges — fur boundaries, whiskers — remain sharp because the luminance guidance image keeps them
intact.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_joint_bilateral_filter.jpg
          :width: 100%

          Output: Joint Bilateral Filtered

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.joint_bilateral_filter`
     - Edge-preserving smoothing guided by a separate reference image
   * - :py:func:`cvcuda.cvtcolor`
     - Convert RGB to grayscale (and back) to build the guidance tensor

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save filtered image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic spatial transformation
* :ref:`Common Utilities <sample_common>` - Helper functions
