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

.. _sample_crop_flip_normalize_reformat:

Crop Flip Normalize Reformat
============================

Overview
--------

The Crop Flip Normalize Reformat sample demonstrates how to combine four common image
preprocessing steps — spatial cropping, horizontal or vertical flipping, per-channel
normalization, and layout reformatting (HWC → NCHW) — into a single GPU kernel call
using CV-CUDA's ``crop_flip_normalize_reformat`` operator.  This fused pipeline is
typical in deep-learning inference pipelines where images must be cropped to a region
of interest, augmented with a flip, and then normalized and reformatted before being
fed to a model.

Usage
-----

Basic Usage
^^^^^^^^^^^

Run the pipeline on the default tabby-cat image:

.. code-block:: bash

   python3 crop_flip_normalize_reformat.py -i input.jpg

Custom Input / Output
^^^^^^^^^^^^^^^^^^^^^

Specify your own input image and save the result to a custom path:

.. code-block:: bash

   python3 crop_flip_normalize_reformat.py -i image.jpg -o cat_crop_flip_normalize_reformat.jpg

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
     - cvcuda/.cache/cat_crop_flip_normalize_reformat.jpg
     - Output image file path

Implementation
--------------

Crop, Flip, Normalize, and Reformat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/crop_flip_normalize_reformat.py
   :language: python
   :start-after: docs_tag: begin_crop_flip_normalize_reformat
   :end-before: docs_tag: end_crop_flip_normalize_reformat
   :dedent:

Key points:

1. **Fused kernel**: All four operations (crop, flip, normalize, reformat) run in a
   single GPU kernel, avoiding intermediate allocations and memory-bandwidth overhead.
2. **ImageBatchVarShape input**: The operator requires images wrapped in a
   :py:class:`cvcuda.ImageBatchVarShape`, which supports batches of varying-size images.
3. **Crop rectangle tensor**: The ``rect`` tensor has shape ``[N, 1, 1, 4]`` with
   ``[crop_x, crop_y, crop_width, crop_height]`` stored per image in the last dimension.
4. **SCALE_IS_STDDEV flag**: When :py:data:`cvcuda.NormalizeFlags.SCALE_IS_STDDEV` is
   set the ``scale`` argument is interpreted as per-channel standard deviation, matching
   the common ``(pixel/255 - mean) / std`` convention used by PyTorch models.
5. **NCHW output layout**: Passing ``out_layout="NCHW"`` reformats the data from the
   interleaved HWC format that the camera/decoder produces into the planar CHW format
   expected by most deep-learning frameworks — no separate ``reformat`` call is required.

Expected Output
^^^^^^^^^^^^^^^

The output shows the central 80% of the image, flipped horizontally, with pixel values
de-normalized back to uint8 for display:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_crop_flip_normalize_reformat.jpg
          :width: 100%

          Output: Cropped, Flipped, Normalized (visualized as uint8)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.crop_flip_normalize_reformat`
     - Crop a region of interest, optionally flip, normalize per channel, and reformat
       the layout in a single fused GPU kernel

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save the result image
* ``cuda_memcpy_h2d`` - Upload crop-rect, flip-code, and normalization parameter arrays to GPU
* ``cuda_memcpy_d2h`` - Download the float32 result to CPU for inverse-normalization

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Simple spatial resize
* :ref:`Normalize Operator <sample_normalize>` - Standalone per-channel normalization
* :ref:`Common Utilities <sample_common>` - Helper functions
