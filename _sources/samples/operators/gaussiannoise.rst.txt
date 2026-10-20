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

.. _sample_gaussiannoise:

Gaussian Noise
==============

Overview
--------

The Gaussian Noise sample demonstrates how to add per-image Gaussian noise to an
image using CV-CUDA's GPU-accelerated ``gaussiannoise`` operator. The operator
accepts per-image mean (``mu``) and standard deviation (``sigma``) tensors, making
it straightforward to apply different noise levels to images in a batch.

Usage
-----

Basic Usage
^^^^^^^^^^^

Add Gaussian noise with default settings:

.. code-block:: bash

   python3 gaussiannoise.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Save the noisy image to a specific location:

.. code-block:: bash

   python3 gaussiannoise.py -i input.jpg -o noisy_cat.jpg

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
     - cvcuda/.cache/cat_gaussiannoise.jpg
     - Output image file path

Implementation
--------------

Applying Gaussian Noise
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/gaussiannoise.py
   :language: python
   :start-after: docs_tag: begin_gaussiannoise
   :end-before: docs_tag: end_gaussiannoise
   :dedent:

Key points:

1. **Per-image parameters**: ``mu`` and ``sigma`` are rank-1 tensors with layout ``"N"``, one scalar per image in the batch.
2. **per_channel flag**: When ``False`` the same noise sample is applied to every colour channel; set to ``True`` for independent per-channel noise.
3. **Reproducibility**: The ``seed`` parameter pins the PRNG state so results are deterministic across runs.
4. **Data type preservation**: The output tensor keeps the same dtype and layout as the input; no implicit conversion occurs.
5. **Clipping**: For ``U8`` inputs the operator automatically clamps the noisy values to ``[0, 255]``.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image with visible Gaussian noise (``sigma=25``):

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_gaussiannoise.jpg
          :width: 100%

          Output: Image with Gaussian Noise (sigma=25)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.gaussiannoise`
     - Add per-image Gaussian noise with configurable mu, sigma, and per-channel control

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save noisy image
* ``cuda_memcpy_h2d`` - Upload mu/sigma parameter arrays to GPU

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic image transformation
* :ref:`Common Utilities <sample_common>` - Helper functions
