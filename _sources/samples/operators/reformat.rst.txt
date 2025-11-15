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

.. _sample_reformat:

Reformat
========

Overview
--------

The Reformat sample demonstrates tensor layout transformations using CV-CUDA.

Usage
-----

Basic Usage
^^^^^^^^^^^

Run layout conversion demonstrations:

.. code-block:: bash

   python3 reformat.py -i input.jpg

Command-Line Arguments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 30 35

   * - Argument
     - Short Form
     - Default
     - Description
   * - ``--input``
     - ``-i``
     - tabby_tiger_cat.jpg
     - Input image file path

Implementation
--------------

Layout Conversions
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/reformat.py
   :language: python
   :start-after: docs_tag: begin_reformat
   :end-before: docs_tag: end_reformat
   :dedent:

The sample demonstrates three key conversions:

1. **HWC → CHW**: Single image, channels-last to channels-first
2. **CHW → HWC**: Reverse of above
3. **NHWC → NCHW**: Batched images with layout change

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.reformat`
     - Convert between different tensor layouts
   * - :py:func:`cvcuda.stack`
     - Add batch dimension (used alongside reformat)

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image (returns HWC)

See Also
--------

* :ref:`Classification Sample <sample_classification>` - Uses reformat for model input
* :ref:`Stack Operator <sample_stack>` - Adding batch dimensions
* :ref:`Common Utilities <sample_common>` - Helper functions
