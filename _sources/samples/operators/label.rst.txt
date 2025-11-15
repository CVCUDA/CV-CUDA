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

.. _sample_label:

Label
=====

Overview
--------

The Connected Components Labeling sample demonstrates how to identify and label distinct regions in binary images using CV-CUDA.

Usage
-----

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   python3 label.py -i input.jpg -o labeled.jpg

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
     - cvcuda/.cache/cat_labeled.jpg
     - Output image file path with color-coded labels

Implementation
--------------

Preprocessing
^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/label.py
   :language: python
   :start-after: docs_tag: begin_preprocessing
   :end-before: docs_tag: end_preprocessing
   :dedent:

Steps:
1. Convert to grayscale
2. Histogram equalization for contrast
3. Threshold to create binary image

Connected Components
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/label.py
   :language: python
   :start-after: docs_tag: begin_labeling
   :end-before: docs_tag: end_labeling
   :dedent:

Output:
* ``cc_labels``: Integer label for each pixel (0=background, 1,2,3...=objects)
* Second output: Number of components found
* Third output: Statistics (not used in this sample)

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_labeled.jpg
          :width: 100%

          Output with Labeled Components

Visualization
^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/label.py
   :language: python
   :start-after: docs_tag: begin_visualization
   :end-before: docs_tag: end_visualization
   :dedent:

The output image shows each connected component in a unique random color.

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.cvtcolor`
     - Convert RGB to grayscale
   * - :py:func:`cvcuda.histogrameq`
     - Enhance contrast
   * - :py:func:`cvcuda.threshold`
     - Create binary image
   * - :py:func:`cvcuda.label`
     - Find connected components

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image
* :ref:`write_image() <common_write_image>` - Save result
* :ref:`cuda_memcpy_h2d() <common_cuda_memcpy_h2d>` - Upload threshold values
* :ref:`cuda_memcpy_d2h() <common_cuda_memcpy_d2h>` - Download labels for visualization

See Also
--------

* :ref:`Segmentation Sample <sample_segmentation>` - Pixel-level classification
* :ref:`Common Utilities <sample_common>` - Helper functions
