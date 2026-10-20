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

.. _sample_morphology:

Morphology
==========

Overview
--------

The Morphology sample demonstrates GPU-accelerated morphological image processing using CV-CUDA.
It applies a dilation followed by an erosion (equivalent to a morphological close operation) to
fill small dark gaps while preserving the main structures of the image. The sample illustrates how to
choose a structuring element size and how to supply a workspace tensor when required.

Usage
-----

Basic Usage
^^^^^^^^^^^

Apply morphological close (dilate then erode) with a 5×5 kernel to the default cat image:

.. code-block:: bash

   python3 morphology.py -i input.jpg

Custom Input
^^^^^^^^^^^^

Process a different image and save the result explicitly:

.. code-block:: bash

   python3 morphology.py -i image.jpg -o cat_morphology.jpg

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
     - cvcuda/.cache/cat_morphology.jpg
     - Output image file path

Implementation
--------------

Morphological Close (Dilate then Erode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/morphology.py
   :language: python
   :start-after: docs_tag: begin_morphology
   :end-before: docs_tag: end_morphology
   :dedent:

Key points:

1. **Batch reshape**: The HWC tensor returned by ``read_image`` is reshaped to NHWC before calling the operator, which accepts both layouts.
2. **Structuring element**: ``mask_size=[5, 5]`` selects a 5×5 rectangular kernel; ``anchor=[-1, -1]`` auto-centres it.
3. **DILATE then ERODE**: Applying dilation followed by erosion is a morphological close, which fills small dark holes and gaps while keeping large bright structures intact.
4. **Workspace tensor**: A workspace tensor of the same shape and dtype as the input is required when passing the result of one morphological call into a second one; it is used internally by the operator as scratch memory.
5. **Output reshape**: The NHWC result is reshaped back to HWC before ``write_image`` to produce a standard single-image output.

Expected Output
^^^^^^^^^^^^^^^

The output shows the image after morphological closing — small dark gaps are filled and bright regions are slightly expanded:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_morphology.jpg
          :width: 100%

          Output: Morphological Close (5×5 kernel)

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.morphology`
     - Apply dilation and erosion with a rectangular structuring element

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save the morphologically processed image
* :ref:`parse_image_args() <common_parse_image_args>` - Parse ``--input`` / ``--output`` CLI arguments

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Basic spatial transform operator
* :ref:`Common Utilities <sample_common>` - Helper functions used across samples
