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

.. _sample_stack:

Stack
=====

Overview
--------

The Stack sample demonstrates how to combine multiple images into a batched tensor using CV-CUDA's stack operator.

Usage
-----

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   python3 stack.py -i input.jpg

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

Stacking Images
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/stack.py
   :language: python
   :start-after: docs_tag: begin_stack
   :end-before: docs_tag: end_stack
   :dedent:

Understanding Stack
^^^^^^^^^^^^^^^^^^^

Combines multiple tensors along a new batch dimension. For example, three HWC images of shape (480, 640, 3) become a single NHWC batch of shape (3, 480, 640, 3). All input tensors must have the same shape, data type, and layout.

Common Use Cases
^^^^^^^^^^^^^^^^

**Batch Inference:**

.. code-block:: python

   images = [read_image(p) for p in image_paths]
   batch = cvcuda.stack(images)
   predictions = model(batch)

**Batch Operations:**

.. code-block:: python

   batch = cvcuda.stack(images)
   resized_batch = cvcuda.resize(batch, (batch.shape[0], 224, 224, 3))
   blurred_batch = cvcuda.gaussian(resized_batch, (5, 5), (1.0, 1.0))

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.stack`
     - Combine multiple tensors into batch

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load images
* :ref:`zero_copy_split() <common_zero_copy_split>` - Split batch back to individual tensors

See Also
--------

* :ref:`Hello World Sample <sample_hello_world>` - Uses stack for batching
* :ref:`Reformat Operator <sample_reformat>` - Layout conversions
* :ref:`Common Utilities <sample_common>` - Helper functions
