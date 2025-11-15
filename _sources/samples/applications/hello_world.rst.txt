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

.. _sample_hello_world:

Hello World
===========

Overview
--------

The Hello World sample demonstrates the fundamental workflow of CV-CUDA for GPU-accelerated computer vision processing. This sample showcases:

* Load images from disk into CV-CUDA tensors
* Resize images to a target resolution
* Batch multiple images into a single tensor
* Apply GPU-accelerated Gaussian blur
* Split the batch and save results back to disk

All operations are performed entirely on the GPU without copying data back to the host, demonstrating CV-CUDA's zero-copy interoperability and efficient batch processing capabilities.

Usage
-----

Run with default parameters (processes tabby_tiger_cat.jpg):

.. code-block:: bash

   python3 hello_world.py

Process a single image with custom dimensions:

.. code-block:: bash

   python3 hello_world.py -i input.jpg -o output.jpg --width 512 --height 512

Process multiple images in a single batch:

.. code-block:: bash

   python3 hello_world.py -i img1.jpg img2.jpg img3.jpg -o out1.jpg out2.jpg out3.jpg

Command-Line Arguments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Argument
     - Short Form
     - Default
     - Description
   * - ``--inputs``
     - ``-i``
     - tabby_tiger_cat.jpg
     - Input image file path(s). Multiple images can be specified.
   * - ``--outputs``
     - ``-o``
     - cvcuda/.cache/cat_hw.jpg
     - Output image file path(s). Must match number of inputs.
   * - ``--width``
     -
     - 224
     - Target width for resized images
   * - ``--height``
     -
     - 224
     - Target height for resized images
   * - ``--kernel``
     - ``-k``
     - 5
     - Kernel size for Gaussian blur (must be odd)
   * - ``--sigma``
     - ``-s``
     - 1.0
     - Sigma value for Gaussian blur

Implementation Details
----------------------

The Hello World sample follows this processing pipeline:

1. Image loading (decode from disk using nvImageCodec to CV-CUDA tensors)
2. Resizing (to target dimensions)
3. Batching (stack into single batched tensor)
4. Gaussian blur
5. Splitting (back to individual tensors)
6. Saving (encode to disk)

All operations execute entirely on the GPU.

Code Walkthrough
^^^^^^^^^^^^^^^^

Loading Images
""""""""""""""

.. literalinclude:: ../../../../samples/applications/hello_world.py
   :language: python
   :start-after: docs_tag: begin_load_images
   :end-before: docs_tag: end_load_images
   :dedent:

Images are:

* Decoded using nvImageCodec's GPU decoder
* Converted to CV-CUDA tensors with HWC (Height-Width-Channels) layout
* Kept in GPU memory throughout

Resizing Images
"""""""""""""""

.. literalinclude:: ../../../../samples/applications/hello_world.py
   :language: python
   :start-after: docs_tag: begin_resize
   :end-before: docs_tag: end_resize
   :dedent:

Each image is resized to the target dimensions using linear (bilinear) interpolation for smooth results.

Batching Images
"""""""""""""""

.. literalinclude:: ../../../../samples/applications/hello_world.py
   :language: python
   :start-after: docs_tag: begin_batch
   :end-before: docs_tag: end_batch
   :dedent:

The :py:func:`cvcuda.stack` operation combines individual HWC tensors into a single NHWC tensor, enabling efficient batched processing.

Applying Gaussian Blur
"""""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/hello_world.py
   :language: python
   :start-after: docs_tag: begin_gaussian_blur
   :end-before: docs_tag: end_gaussian_blur
   :dedent:

The Gaussian blur operation applies to entire batch simultaneously with constant border handling for edge pixels.

Splitting and Saving
""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/hello_world.py
   :language: python
   :start-after: docs_tag: begin_split_and_save
   :end-before: docs_tag: end_split_and_save
   :dedent:

The ``zero_copy_split()`` helper function splits the batched tensor back to individual images without memory copying.

Expected Output
---------------

The output will be a 224×224 image with a 5×5 Gaussian blur applied. The output will be smoothed while preserving major features and colors of the original image.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_hw.jpg
          :width: 100%

          Output: Resized and Blurred

CV-CUDA Operators Used
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.resize`
     - Resize images to target dimensions using bilinear interpolation
   * - :py:func:`cvcuda.stack`
     - Combine multiple tensors into a batched tensor along a new dimension
   * - :py:func:`cvcuda.gaussian`
     - Apply Gaussian blur filter for smoothing

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`zero_copy_split() <common_zero_copy_split>` - Split batched tensors efficiently

Recap
-----

The Hello World sample introduced you to CV-CUDA's core concepts:

**GPU-Only Processing**
  All operations happened on the GPU. The image was loaded from disk directly to GPU memory, processed entirely on the GPU, and saved from GPU memory back to disk. No CPU-GPU memory copies were needed for the actual image data.

**Batching for Efficiency**
  Images are converted to batch format (NHWC) to enable efficient parallel processing of multiple images.

**Zero-Copy Interoperability**
  CV-CUDA integrates seamlessly with nvImageCodec for image I/O, using the ``__cuda_array_interface__`` protocol for zero-copy data sharing between libraries.

**Operator Chaining**
  The sample chained multiple operations (load → resize → stack → blur → split → save) in a pipeline, showing how CV-CUDA operators work together.

See Also
--------

* :ref:`Gaussian Blur Operator Sample <sample_gaussian>` - Detailed Gaussian blur documentation
* :ref:`Resize Operator Sample <sample_resize>` - Detailed resize documentation
* :ref:`Stack Operator Sample <sample_stack>` - Detailed stack documentation
* :ref:`Common Utilities <sample_common>` - Shared helper functions

Next Steps
----------

After mastering the Hello World sample, explore:

* :ref:`Image Classification <sample_classification>` - End-to-end deep learning inference
* :ref:`Object Detection <sample_object_detection>` - Detection with bounding boxes
* :ref:`Semantic Segmentation <sample_segmentation>` - Pixel-level classification
