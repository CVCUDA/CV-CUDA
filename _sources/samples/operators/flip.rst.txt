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

.. _sample_flip:

Flip
====

Overview
--------

The Flip sample demonstrates GPU-accelerated image flipping using CV-CUDA's flip operator.
The operator mirrors an image along one or both axes: horizontal (left-right), vertical
(top-bottom), or both simultaneously.

Usage
-----

Basic Usage
^^^^^^^^^^^

Flip the default input image horizontally (left-right mirror):

.. code-block:: bash

   python3 flip.py -i input.jpg

Custom Output Path
^^^^^^^^^^^^^^^^^^

Write the flipped result to a custom path:

.. code-block:: bash

   python3 flip.py -i input.jpg -o cat_flip.jpg

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
     - cvcuda/.cache/cat_flip.jpg
     - Output image file path

Implementation
--------------

Horizontal Flip
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/flip.py
   :language: python
   :start-after: docs_tag: begin_flip
   :end-before: docs_tag: end_flip
   :dedent:

Key points:

1. **flipCode=1** mirrors the image left-right (horizontal flip).
2. **flipCode=0** mirrors the image top-to-bottom (vertical flip).
3. **flipCode=-1** mirrors the image along both axes simultaneously.

Expected Output
^^^^^^^^^^^^^^^

The output is a mirror image of the input flipped left-right:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_flip.jpg
          :width: 100%

          Output: Horizontally Flipped Image

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.flip`
     - Mirror an image along one or both spatial axes

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save flipped image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images with CV-CUDA
* :ref:`Common Utilities <sample_common>` - Helper functions
