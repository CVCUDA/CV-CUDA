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

.. _sample_brightness_contrast:

Brightness Contrast
===================

Overview
--------

The Brightness Contrast sample demonstrates GPU-accelerated brightness and contrast adjustment
using CV-CUDA's ``brightness_contrast`` operator.  The operator applies a per-image affine
transform to every pixel:

.. code-block:: text

   output = brightness * (contrast * (input − contrast_center) + contrast_center) + brightness_shift

where ``brightness``, ``contrast``, ``brightness_shift``, and ``contrast_center`` are
per-image scalar tensors.  This formulation separates multiplicative brightness from the
contrast pivot, giving fine-grained control over image appearance.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply the default brightening/contrast boost to the bundled test image:

.. code-block:: bash

   python3 brightness_contrast.py

Custom Input
^^^^^^^^^^^^

Supply your own image:

.. code-block:: bash

   python3 brightness_contrast.py -i input.jpg -o output.jpg

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
     - cvcuda/.cache/cat_brightness_contrast.jpg
     - Output image file path

Implementation
--------------

Parameter Setup
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/brightness_contrast.py
   :language: python
   :start-after: docs_tag: begin_brightness_contrast_setup
   :end-before: docs_tag: end_brightness_contrast_setup
   :dedent:

Operator Call
^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/brightness_contrast.py
   :language: python
   :start-after: docs_tag: begin_brightness_contrast
   :end-before: docs_tag: end_brightness_contrast
   :dedent:

Key points:

1. **Parameter tensors** — each of ``brightness``, ``contrast``, ``brightness_shift``, and
   ``contrast_center`` is a 1-D ``"N"``-layout float32 tensor with one element per image.
2. **All parameters are optional** — you may pass any subset; omitted parameters default to
   identity values (brightness=1, contrast=1, brightness_shift=0, contrast_center=0).
3. **Dtype preserved** — the output tensor has the same dtype and layout as the input, so no
   post-processing conversion is needed for uint8 HWC images.
4. **contrast_center pivot** — setting ``contrast_center`` to 127.0 for uint8 data places the
   pivot at mid-gray, which preserves overall luminance while expanding tonal range.

Expected Output
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_brightness_contrast.jpg
          :width: 100%

          Output: Brightness × 1.5, Contrast × 1.4

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.brightness_contrast`
     - Per-image affine pixel transform for brightness and contrast adjustment

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save adjusted image
* ``cuda_memcpy_h2d`` - Upload per-image scalar parameters from host to device

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images with CV-CUDA
* :ref:`Common Utilities <sample_common>` - Helper functions used by all operator samples
