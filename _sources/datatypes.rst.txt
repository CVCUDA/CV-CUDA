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

.. _datatypes:

Data Types
==========

CV-CUDA provides high-performance data type abstractions for computer vision operations.

This guide covers the core data types including Tensors, Images, and their batched variants.

Tensor vs. Image
----------------

CV-CUDA provides two primary data types:

* :py:class:`Tensor <cvcuda.Tensor>` - N-dimensional arrays for generic planar data
* :py:class:`Image <cvcuda.Image>` – 2D data type with color information, support for multi-planar formats, and data layouts

When to Use Each
^^^^^^^^^^^^^^^^

Tensor
~~~~~~

Use a Tensor when:

1. **Generic N-dimensional data** - Not specifically image/color data
2. **Non-color information** - Feature maps, masks, depth maps
3. **Uniform data layout** - All dimensions follow regular striding
4. **ML pipeline integration** - Tensors map naturally to ML frameworks like PyTorch

Image
~~~~~

Use an Image when:

1. **Color-correct processing is required** - Image carries :py:class:`Format <cvcuda.Format>` metadata including color space, encoding, and transfer functions
2. **Multi-planar formats** - Formats like NV12 have planes with different dimensions (Y plane is full resolution, UV plane is half resolution)
3. **Complex color formats** - YUV420, YUV422, raw Bayer patterns, etc.

.. note::
   **Why Images exist separately from Tensors:**

   The :py:class:`Image <cvcuda.Image>` class provides a more complete abstraction for image data with methods and properties directly related to image operations.
   Even when an image could technically be represented as a Tensor, using :py:class:`Image <cvcuda.Image>` is preferred because it better maps to the underlying domain concept.

   The :py:class:`Image <cvcuda.Image>` class carries the :py:class:`Format <cvcuda.Format>` along with its data, which allows operators to correctly interpret the image's color content.
   Additionally, some image formats cannot be directly represented as Tensors at all, such as NV12, which has two planes with different dimensions - the Y plane at full resolution and the UV plane at half resolution.
   Such multi-planar formats with different plane dimensions are not possible within a single :py:class:`Tensor <cvcuda.Tensor>`.

Key Differences
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Tensor
     - Image
   * - **Primary use**
     - Generic N-D arrays
     - Color-correct image processing
   * - **Format metadata**
     - Just DataType
     - Full ImageFormat with color info
   * - **Planes**
     - N/A (no plane concept)
     - Supports multi-planar (different dims per plane)
   * - **Layouts**
     - N-dimensional strided
     - Pitch-linear and block-linear

Tensor
^^^^^^

Overview
~~~~~~~~

A :py:class:`Tensor <cvcuda.Tensor>` is an N-dimensional array with a uniform data type and layout.
:py:class:`Tensor <cvcuda.Tensor>` is flexible and can represent a wide variety of data types including uniform planar images,
segmentation masks, feature maps, depth maps, and general numerical data.

**Properties:**

* Shape (size of each dimension)
* :py:class:`DataType <cvcuda.DataType>` (e.g., ``U8``, ``F32``)
* :py:class:`TensorLayout <cvcuda.TensorLayout>` (e.g., ``NHWC``, ``NCHW``, ``HWC``)
* Strides (byte offset between elements in each dimension)

Common :py:class:`Tensor <cvcuda.Tensor>` Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :py:class:`TensorLayout <cvcuda.TensorLayout>` describes the semantic meaning of each dimension in a :py:class:`Tensor <cvcuda.Tensor>` using dimension labels.
Each label indicates what kind of information that dimension represents (e.g., batch size, height, width, channels).

Standard layouts:

* :py:data:`cvcuda.TensorLayout.NHWC` - **N**\umber of samples, **H**\eight, **W**\idth, **C**\hannels - Batch dimension followed by spatial dimensions with interleaved channels
* :py:data:`cvcuda.TensorLayout.NCHW` - **N**\umber of samples, **C**\hannels, **H**\eight, **W**\idth - Batch dimension followed by channels, then spatial dimensions (common in ML frameworks)
* :py:data:`cvcuda.TensorLayout.HWC` - **H**\eight, **W**\idth, **C**\hannels - Single 2D image with packed channels in one plane
* :py:data:`cvcuda.TensorLayout.CHW` - **C**\hannels, **H**\eight, **W**\idth - Single 2D multi-planar image where each channel is in its own plane
* :py:data:`cvcuda.TensorLayout.NW` - **N**\umber of samples, **W**\idth - 2D data without spatial dimensions

**Standard dimension labels:**

A label represents the semantic meaning for a dimension of a tensor. Common labels such as H (height) or W (width)
are found, but there are additional common labels:

* ``N`` - Batch/samples
* ``C`` - Channels
* ``H`` - Height
* ``W`` - Width
* ``D`` - Depth (3D spatial dimension)
* ``F`` - Frames (temporal depth for video)

Creating Tensors
~~~~~~~~~~~~~~~~

.. literalinclude:: ../../samples/datatypes/tensor.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

Tensor Documentation
~~~~~~~~~~~~~~~~~~~~

Refer to the following for more information:

* :py:class:`Tensor <cvcuda.Tensor>` documentation
* :py:class:`TensorLayout <cvcuda.TensorLayout>` documentation
* :py:class:`Type <cvcuda.Type>` documentation

Image
^^^^^

Overview
~~~~~~~~

An :py:class:`Image <cvcuda.Image>` in CV-CUDA is a 2D array of pixels, where each pixel is a unit of visual data composed of one or more color channels that may be stored across one or more planes.
:py:class:`Image <cvcuda.Image>` includes metadata about how the pixel data should be interpreted.

**Properties:**

* Width and height dimensions
* :py:class:`Format <cvcuda.Format>` - encodes color model, color space, chroma subsampling, memory layout, data type, channel swizzle, and packing
* Can have multiple planes (e.g., NV12 has 2 planes: Y and UV)

Common Image Formats
~~~~~~~~~~~~~~~~~~~~

**Simple RGB/BGR formats:**

* :py:data:`cvcuda.Format.RGB8` - **8-bit RGB interleaved** - Red, Green, Blue channels packed together (RGBRGBRGB...) in a single plane, 8 bits per channel
* :py:data:`cvcuda.Format.RGBA8` - **8-bit RGBA interleaved** - Red, Green, Blue, Alpha channels packed together (RGBARGBA...) in a single plane, 8 bits per channel
* :py:data:`cvcuda.Format.BGR8` - **8-bit BGR interleaved** - Blue, Green, Red channels packed together (BGRBGRBGR...) in a single plane, 8 bits per channel (common in OpenCV)
* :py:data:`cvcuda.Format.RGB8p` - **8-bit RGB planar** - Red, Green, Blue channels in separate planes (RRR...GGG...BBB...), 8 bits per channel

**Single channel formats:**

* :py:data:`cvcuda.Format.U8` - **8-bit grayscale/single channel** - Unsigned 8-bit integer, single plane
* :py:data:`cvcuda.Format.F32` - **32-bit float single channel** - 32-bit floating point, single plane (useful for depth maps, feature maps)

**YUV formats (video/camera):**

* :py:data:`cvcuda.Format.NV12` - **YUV420 semi-planar** - 2 planes: full-resolution Y (luma) + half-resolution interleaved UV (chroma). Common in video codecs and cameras
* :py:data:`cvcuda.Format.NV21` - **YUV420 semi-planar (VU order)** - 2 planes: full-resolution Y (luma) + half-resolution interleaved VU (chroma, opposite order from NV12)
* :py:data:`cvcuda.Format.YUV8p` - **YUV444 planar** - 3 separate full-resolution planes: Y, U, V. No chroma subsampling
* :py:data:`cvcuda.Format.YUYV` - **YUV422 packed** - Single plane with horizontally subsampled chroma (YUYV YUYV...)

.. image:: content/DataLayout_1D.svg
   :width: 75%
   :align: center

Creating Images
~~~~~~~~~~~~~~~

You can allocate an Image directly using several methods:

.. literalinclude:: ../../samples/datatypes/image.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

Image Documentation
~~~~~~~~~~~~~~~~~~~

Refer to the following for more information:

* :py:class:`Image <cvcuda.Image>` documentation
* :py:class:`Format <cvcuda.Format>` documentation

Converting Between Image and Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Image to Tensor
~~~~~~~~~~~~~~~~

:py:class:`Image <cvcuda.Image>` objects can be wrapped as :py:class:`Tensor <cvcuda.Tensor>` objects when they meet certain requirements.

See :py:func:`as_tensor <cvcuda.as_tensor>` for more details.

**Requirements for Image → Tensor conversion:**

* Must be pitch-linear
* No chroma subsampling (must be 4:4:4)
* All planes must have the same data type and size

.. note::
   Image planes can be discontiguous (independent buffers in memory).
   Tensor strides can represent this by encoding the address difference between planes
   as the stride value for the outermost dimension.
   This allows wrapping discontiguous multi-planar images (e.g., 2-planar formats like planar RGBA) as tensors,
   provided all planes have identical dimensions and data types.

**Example:**

.. literalinclude:: ../../samples/datatypes/conversions.py
   :language: python
   :pyobject: image_to_tensor

Tensor to Image
~~~~~~~~~~~~~~~

:py:class:`Tensor <cvcuda.Tensor>` objects can be loaded into :py:class:`Image <cvcuda.Image>` objects, but not directly. In order to do so, you pass the foreign interface accessible via the ``.cuda()`` method to the :py:func:`as_image <cvcuda.as_image>` function.

.. note::
   While it is possible to load a :py:class:`Tensor <cvcuda.Tensor>` into an :py:class:`Image <cvcuda.Image>`, it is not recommended.

**Example:**

.. literalinclude:: ../../samples/datatypes/conversions.py
   :language: python
   :pyobject: tensor_to_image

Batched Data Types
------------------

CV-CUDA provides three container types for batching multiple :py:class:`Image <cvcuda.Image>` or :py:class:`Tensor <cvcuda.Tensor>` together for batch processing.

TensorBatch
^^^^^^^^^^^

:py:class:`TensorBatch <cvcuda.TensorBatch>` is a container that holds multiple :py:class:`Tensor <cvcuda.Tensor>` with varying shapes.

**Requirements:**

* All :py:class:`Tensor <cvcuda.Tensor>`\s must have the same data type
* All :py:class:`Tensor <cvcuda.Tensor>`\s must have the same rank (number of dimensions)
* All :py:class:`Tensor <cvcuda.Tensor>`\s must have the same layout

.. image:: content/TensorBatch.svg
   :width: 75%
   :align: center

**Creating TensorBatch**

.. literalinclude:: ../../samples/datatypes/tensorbatch.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

ImageBatch
^^^^^^^^^^

:py:class:`ImageBatch <cvcuda.ImageBatch>` is a container for :py:class:`Image <cvcuda.Image>`\s with uniform dimensions and formats.

**Requirements:**

* All :py:class:`Image <cvcuda.Image>`\s must have the same size (width and height)
* All :py:class:`Image <cvcuda.Image>`\s must have the same format

.. image:: content/ImageBatch.svg
   :width: 75%
   :align: center

**Creating ImageBatch**

.. literalinclude:: ../../samples/datatypes/imagebatch.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

.. note::
   **ImageBatch in Python:** There is no separate :py:class:`ImageBatch <cvcuda.ImageBatch>` class in the Python API.
   For uniform image batches, use either a :py:class:`Tensor <cvcuda.Tensor>` with :py:data:`cvcuda.TensorLayout.NHWC` layout or :py:class:`ImageBatchVarShape <cvcuda.ImageBatchVarShape>`.
   The :py:class:`ImageBatchVarShape <cvcuda.ImageBatchVarShape>` class handles both uniform image batches (all same size/format) and variable-shape batches (mixed sizes/formats).

ImageBatchVarShape
^^^^^^^^^^^^^^^^^^

:py:class:`ImageBatchVarShape <cvcuda.ImageBatchVarShape>` is a container for :py:class:`Image <cvcuda.Image>` with variable shapes and formats, providing maximum flexibility.

**Requirements:**

* No requirements - each :py:class:`Image <cvcuda.Image>` can have different dimensions and formats

.. image:: content/ImageBatchVarShape.svg
   :width: 75%
   :align: center

**Creating ImageBatchVarShape**

.. literalinclude:: ../../samples/datatypes/imagebatchvarshape.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

Batch Types Comparison
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - :py:class:`TensorBatch <cvcuda.TensorBatch>`
     - :py:class:`ImageBatch <cvcuda.ImageBatch>`
     - :py:class:`ImageBatchVarShape <cvcuda.ImageBatchVarShape>`
   * - **Data Type**
     - :py:class:`Tensor <cvcuda.Tensor>`
     - :py:class:`Image <cvcuda.Image>`
     - :py:class:`Image <cvcuda.Image>`
   * - **Shape Flexibility**
     - Different shapes (same rank)
     - All same size
     - Different sizes
   * - **Format Flexibility**
     - Same dtype/layout
     - Same format
     - Different formats
   * - **Restrictions**
     - Same rank, dtype, layout
     - Same size, format
     - None
   * - **Use Case**
     - Variable-size feature maps
     - Uniform image batches
     - Mixed image sizes/formats
   * - **Python API**
     - :py:class:`TensorBatch <cvcuda.TensorBatch>`
     - Use :py:class:`Tensor <cvcuda.Tensor>` (NHWC) or :py:class:`ImageBatchVarShape <cvcuda.ImageBatchVarShape>`
     - :py:class:`ImageBatchVarShape <cvcuda.ImageBatchVarShape>`

.. note::
   The :py:class:`TensorBatch <cvcuda.TensorBatch>`, :py:class:`ImageBatch <cvcuda.ImageBatch>` and :py:class:`ImageBatchVarShape <cvcuda.ImageBatchVarShape>` classes are not interchangeable.
