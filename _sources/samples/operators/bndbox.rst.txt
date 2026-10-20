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

.. _sample_bndbox:

Bounding Boxes
==============

Overview
--------

The Bounding Boxes sample demonstrates GPU-accelerated axis-aligned bounding-box
rendering using CV-CUDA's ``bndbox`` operator.  Three colored rectangles are drawn
over a cat image, each with an independent border color and thickness.

Usage
-----

Basic Usage
^^^^^^^^^^^

Draw boxes on the default cat image:

.. code-block:: bash

   python3 bndbox.py

Custom Input
^^^^^^^^^^^^

Specify your own image:

.. code-block:: bash

   python3 bndbox.py -i input.jpg -o cat_bndbox.jpg

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
     - cvcuda/.cache/cat_bndbox.jpg
     - Output image file path

Implementation
--------------

Bounding Box Rendering
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/bndbox.py
   :language: python
   :start-after: docs_tag: begin_bndbox
   :end-before: docs_tag: end_bndbox
   :dedent:

Key points:

1. **Tensor layout**: :py:func:`cvcuda.bndbox` supports ``NHWC``/``HWC`` and
   ``NCHW``/``CHW`` tensors. The sample uses ``NHWC``: use ``cvcuda.stack`` to
   add the batch dimension to a single ``HWC`` image.
2. **BndBoxesI structure**: One list of ``BndBoxI`` objects per batch image; each box specifies ``(x, y, width, height)`` in pixel coordinates.
3. **Fill alpha 0**: Setting the RGBA fill alpha to 0 draws only the border, leaving interior pixels unchanged.
4. **In-place semantics**: The operator returns a new tensor but operates on a copy; the source tensor is not modified.
5. **Layout restoration**: Reshape the NHWC output back to HWC before passing to ``write_image``.

Expected Output
^^^^^^^^^^^^^^^

The output shows the original cat image with three colored bounding boxes drawn on it:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_bndbox.jpg
          :width: 100%

          Output: Three colored bounding boxes drawn on the cat

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.bndbox`
     - Draw axis-aligned bounding boxes with configurable border color and thickness

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save the annotated output image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Scale images before annotation
* :ref:`Common Utilities <sample_common>` - Helper functions
