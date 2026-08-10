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

.. _sample_convertto:

Convert To
==========

Overview
--------

The Convert To sample demonstrates dtype conversion using CV-CUDA's GPU-accelerated
``convertto`` operator. It converts a ``uint8`` image to ``float32`` with a scale factor
of ``1/255`` (normalising pixel values to ``[0, 1]``), then converts the result back to
``uint8`` by applying the inverse scale of ``255``. This round-trip is a fundamental
pre/post-processing step for deep-learning inference pipelines.

Usage
-----

Basic Usage
^^^^^^^^^^^

Convert the default tabby cat image:

.. code-block:: bash

   python3 convertto.py

Custom Input/Output
^^^^^^^^^^^^^^^^^^^

Specify explicit input and output paths:

.. code-block:: bash

   python3 convertto.py -i image.jpg -o cat_convertto.jpg

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
     - cvcuda/.cache/cat_convertto.jpg
     - Output image file path

Implementation
--------------

Convert To Operator
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/convertto.py
   :language: python
   :start-after: docs_tag: begin_convertto
   :end-before: docs_tag: end_convertto
   :dedent:

Key points:

1. **dtype parameter**: Pass a ``numpy`` dtype (e.g. ``np.float32``) or a ``cvcuda.Type``
   enum value — both are accepted by ``cvcuda.convertto``.
2. **scale parameter**: Each output pixel is computed as
   ``out = src * scale + offset``.  Omitting ``scale`` defaults to ``1.0``.
3. **offset parameter**: An optional additive bias applied after scaling; defaults to
   ``0.0`` when omitted.
4. **Layout preservation**: The output tensor always has the same layout (HWC, NHWC,
   CHW, NCHW) as the input tensor.

Expected Output
^^^^^^^^^^^^^^^

The output image is visually identical to the input because the uint8→float32→uint8
round-trip is lossless for pixel values in ``[0, 255]``:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_convertto.jpg
          :width: 100%

          Output: uint8 round-trip via float32

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.convertto`
     - Convert tensor dtype with optional scale and offset

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save converted image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - GPU-accelerated image resizing
* :ref:`Common Utilities <sample_common>` - Helper functions used by all samples
