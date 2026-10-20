..
   # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _sample_gaussian:

Gaussian
========

Overview
--------

The Gaussian Blur sample demonstrates how to apply Gaussian smoothing to images using CV-CUDA.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply Gaussian blur with default parameters:

.. code-block:: bash

   python3 gaussian.py -i input.jpg

The output will be saved as ``cvcuda/.cache/cat_blurred.jpg`` with a 9×9 kernel and sigma=1.5.

Custom Output Path
^^^^^^^^^^^^^^^^^^

Specify the output file:

.. code-block:: bash

   python3 gaussian.py -i image.jpg -o blurred.jpg

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
     - cvcuda/.cache/cat_blurred.jpg
     - Output image file path

Implementation
--------------

Complete Code
^^^^^^^^^^^^^

The entire sample is remarkably concise:

.. literalinclude:: ../../../../samples/operators/gaussian.py
   :language: python
   :start-after: docs_tag: begin_main
   :end-before: docs_tag: end_main
   :dedent:

That's it! Just three key steps:

1. **Load** the image from disk
2. **Apply** Gaussian blur
3. **Save** the result

Expected Output
^^^^^^^^^^^^^^^

The output shows the image with Gaussian blur applied, smoothing the image while preserving overall structure:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_blurred.jpg
          :width: 100%

          Output: Gaussian Blurred

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.gaussian`
     - Apply Gaussian blur with specified kernel and sigma

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image into CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save CV-CUDA tensor as image

See Also
--------

* :ref:`Hello World Sample <sample_hello_world>` - Uses Gaussian blur in pipeline
* :ref:`Segmentation Sample <sample_segmentation>` - Uses blur for background effects
* :py:func:`cvcuda.bilateral_filter` - Edge-preserving alternative
* :ref:`Common Utilities <sample_common>` - Helper functions
