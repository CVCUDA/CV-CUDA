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

.. _sample_gamma_contrast:

Gamma Contrast
==============

Overview
--------

The Gamma Contrast sample demonstrates how to apply per-image gamma correction using CV-CUDA's
GPU-accelerated ``gamma_contrast`` operator. Gamma correction maps each normalised pixel value
``p`` to ``p^gamma``, which is widely used to match display transfer functions (e.g. the sRGB
standard uses gamma ≈ 2.2) or to adjust the perceptual brightness of an image.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply standard sRGB gamma correction (gamma = 2.2) to an image:

.. code-block:: bash

   python3 gamma_contrast.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify a custom output file:

.. code-block:: bash

   python3 gamma_contrast.py -i input.jpg -o my_gamma_output.jpg

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
     - cvcuda/.cache/cat_gamma_contrast.jpg
     - Output image file path

Implementation
--------------

Gamma Contrast Correction
^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/gamma_contrast.py
   :language: python
   :start-after: docs_tag: begin_gamma_contrast
   :end-before: docs_tag: end_gamma_contrast
   :dedent:

Key points:

1. **ImageBatchVarShape input**: This sample demonstrates the var-shape overload, where a
   single-image tensor is wrapped with ``cvcuda.as_image`` and pushed into a
   ``cvcuda.ImageBatchVarShape``. ``gamma_contrast`` also accepts plain ``cvcuda.Tensor``
   input/output, with either a per-sample gamma tensor or a host-scalar ``gamma``/``gain``.
   The host-scalar overload accepts ``round=cvcuda.Round.NEAREST`` (the default) or
   ``round=cvcuda.Round.TRUNCATE`` for integer outputs.
2. **Per-image gamma**: The gamma argument is a 1-D ``float32`` tensor with one value per image
   in the batch, enabling different corrections per image in the same call.
3. **Standard gamma 2.2**: A value of 2.2 matches the sRGB display transfer function, darkening
   mid-tones to compensate for how monitors render brightness non-linearly.
4. **In-place output extraction**: The output ``ImageBatchVarShape`` contains ``cvcuda.Image``
   objects; ``cvcuda.as_tensor`` converts the first image back to a writable HWC tensor with no
   data copy.
5. **uint8 passthrough**: Because the input is already uint8 RGB8, the operator preserves that
   dtype and the result can be written directly with ``write_image``.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image with gamma-corrected pixel intensities:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_gamma_contrast.jpg
          :width: 100%

          Output: Gamma-corrected (gamma = 2.2)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.gamma_contrast`
     - Apply per-image power-law (gamma) contrast correction to an image batch

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save gamma-corrected image
* ``cuda_memcpy_h2d`` - Upload the per-image gamma values to the GPU

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic GPU image resizing
* :ref:`Common Utilities <sample_common>` - Helper functions
