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

.. _sample_bilateral_filter:

Bilateral Filter
================

Overview
--------

The Bilateral Filter sample demonstrates edge-preserving image smoothing using
CV-CUDA's GPU-accelerated bilateral filter operator. Unlike a standard Gaussian
blur, the bilateral filter weighs contributions by both spatial proximity and
color similarity, so it reduces noise in flat regions while leaving edges sharp.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply bilateral filter to an image with default parameters:

.. code-block:: bash

   python3 bilateral_filter.py -i input.jpg

Custom Parameters
^^^^^^^^^^^^^^^^^

Specify a custom output path:

.. code-block:: bash

   python3 bilateral_filter.py -i input.jpg -o cat_bilateral_filter.jpg

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
     - cvcuda/.cache/cat_bilateral_filter.jpg
     - Output image file path

Implementation
--------------

Bilateral Filter Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/bilateral_filter.py
   :language: python
   :start-after: docs_tag: begin_bilateral_filter
   :end-before: docs_tag: end_bilateral_filter
   :dedent:

Key points:

1. **Edge Preservation**: Unlike Gaussian blur, bilateral filter preserves sharp
   edges by weighting pixel contributions by color similarity (``sigma_color``)
   as well as spatial distance (``sigma_space``).
2. **Diameter**: Controls the size of the pixel neighborhood considered for each
   output pixel. Larger values produce stronger smoothing but increase runtime.
3. **Sigma Color**: Higher values allow more dissimilar colors to be blended,
   reducing edge-preservation strength toward a plain Gaussian blur.
4. **Sigma Space**: Controls spatial falloff; behaves like the radius of a
   Gaussian blur and determines how far neighboring pixels contribute.
5. **Border Mode**: ``cvcuda.Border.REFLECT`` mirrors edge pixels outward,
   avoiding darkening or artifacts at image boundaries.

Expected Output
^^^^^^^^^^^^^^^

The output retains sharp edges (fur markings, whiskers) while noise and texture
in flat regions is smoothed:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_bilateral_filter.jpg
          :width: 100%

          Output: Edge-Preserving Bilateral Filter

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.bilateral_filter`
     - Apply edge-preserving bilateral smoothing to an image

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save filtered image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images with GPU acceleration
* :ref:`Common Utilities <sample_common>` - Helper functions
