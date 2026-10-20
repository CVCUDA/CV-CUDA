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

.. _sample_osd:

On-Screen Display
=================

Overview
--------

The On-Screen Display (OSD) sample demonstrates GPU-accelerated compositing of
visual annotations—bounding boxes, text labels, lines, circles, arrows, polygons,
and clock overlays—onto an image using CV-CUDA's :py:func:`cvcuda.osd` operator.

Usage
-----

Basic Usage
^^^^^^^^^^^

Draw the default set of OSD elements onto the sample cat image:

.. code-block:: bash

   python3 osd.py -i input.jpg

Custom Input and Output
^^^^^^^^^^^^^^^^^^^^^^^

Specify a custom input image and output path:

.. code-block:: bash

   python3 osd.py -i input.jpg -o annotated.jpg

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
     - cvcuda/.cache/cat_osd.jpg
     - Output image file path with OSD annotations

Implementation
--------------

OSD Element Setup
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/osd.py
   :language: python
   :start-after: docs_tag: begin_osd_setup
   :end-before: docs_tag: end_osd_setup
   :dedent:

OSD Operator Call
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../samples/operators/osd.py
   :language: python
   :start-after: docs_tag: begin_osd
   :end-before: docs_tag: end_osd
   :dedent:

Key points:

1. **Tensor layout**: :py:func:`cvcuda.osd` supports ``NHWC``/``HWC`` and
   ``NCHW``/``CHW`` tensors. The sample uses ``NHWC``: a single ``HWC`` image is
   reshaped to ``(1, H, W, C)`` before the call and squeezed back afterward.
2. **Elements list-of-lists**: :py:class:`cvcuda.Elements` takes a list with one
   inner list per image in the batch.  Each inner list may contain any mix of the
   supported primitive types.
3. **In-place compositing**: The output tensor shares shape, dtype, and layout
   with the input; all primitives are alpha-blended onto it in a single GPU pass.
4. **Coordinate scaling**: Positions and sizes are computed relative to the image
   dimensions so the overlay adapts to any input resolution.
5. **Primitive variety**: The sample showcases all major OSD primitives—
   :py:class:`cvcuda.BndBoxI`, :py:class:`cvcuda.Label`, :py:class:`cvcuda.Line`,
   :py:class:`cvcuda.Circle`, :py:class:`cvcuda.Arrow`,
   :py:class:`cvcuda.PolyLine`, and :py:class:`cvcuda.Clock`.

Expected Output
^^^^^^^^^^^^^^^

The output is the input image with all OSD annotations composited on top:

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_osd.jpg
          :width: 100%

          Output: Image with OSD annotations

CV-CUDA Operators Used
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.osd`
     - Composite bounding boxes, text, lines, shapes, and overlays onto images

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save annotated image

See Also
--------

* :ref:`Resize Operator <sample_resize>` - Resize images before annotation
* :ref:`Common Utilities <sample_common>` - Helper functions
