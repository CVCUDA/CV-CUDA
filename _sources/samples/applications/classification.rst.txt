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

.. _sample_classification:

Image Classification
====================

Overview
--------

The Image Classification sample demonstrates an end-to-end GPU-accelerated deep learning inference pipeline using CV-CUDA for preprocessing and TensorRT for model inference. This sample showcases:

* Loading and preprocessing images entirely on GPU
* ImageNet normalization with mean and standard deviation
* Integration with TensorRT for high-performance inference
* Processing images through a ResNet50 classification model
* Extracting top-K predictions

Usage
-----

Process an image with default ResNet50 model:

.. code-block:: bash

   python3 classification.py -i image.jpg

The sample will:

1. Download ResNet50 model weights (first run only)
2. Export to ONNX format (first run only)
3. Build TensorRT engine (first run only)
4. Process the image and display top 5 predictions

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
     - cvcuda/.cache/cat_classified.jpg
     - Output file path for predictions
   * - ``--width``
     -
     - 224
     - Target width for model input
   * - ``--height``
     -
     - 224
     - Target height for model input

Implementation Details
----------------------

The classification pipeline consists of:

1. Model setup (ONNX export and TensorRT engine building, cached after first run)
2. Image loading into GPU memory
3. Preprocessing (resize, normalize, reformat)
4. TensorRT inference
5. Extracting and displaying top-K predictions

Code Walkthrough
^^^^^^^^^^^^^^^^

Model Setup
"""""""""""

.. literalinclude:: ../../../../samples/applications/classification.py
   :language: python
   :start-after: docs_tag: begin_model_setup
   :end-before: docs_tag: end_model_setup
   :dedent:

The model setup process:

* **ONNX Export**: Exports PyTorch ResNet50 to ONNX format
* **TensorRT Build**: Compiles ONNX to optimized TensorRT engine
* **Caching**: Models are cached in ``cvcuda/.cache/`` directory
* **Automatic**: Only runs on first execution

Loading Input Image
"""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/classification.py
   :language: python
   :start-after: docs_tag: begin_read_image
   :end-before: docs_tag: end_read_image
   :dedent:

The image is loaded:

* Directly into GPU memory
* As a CV-CUDA tensor
* In HWC (Height-Width-Channels) layout

Preprocessing Pipeline
"""""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/classification.py
   :language: python
   :start-after: docs_tag: begin_preprocessing
   :end-before: docs_tag: end_preprocessing
   :dedent:

The preprocessing steps:

1. **Setup Normalization Parameters**: ImageNet mean and std deviation
2. **Add Batch Dimension**: Convert HWC → NHWC using :py:func:`cvcuda.stack`
3. **Resize**: Scale to target size (default 224×224)
4. **Convert to Float**: Convert uint8 [0,255] → float32 [0.0,1.0]
5. **Normalize**: Apply ImageNet normalization: ``(x - mean) / std``
6. **Reformat**: Convert NHWC → NCHW for PyTorch-style models

ImageNet Normalization
""""""""""""""""""""""

The standard ImageNet normalization uses:

* **Mean**: [0.485, 0.456, 0.406] for RGB channels
* **Std**: [0.229, 0.224, 0.225] for RGB channels

This normalization is critical for pretrained models and is applied as:

.. code-block:: python

   normalized = (image / 255.0 - mean) / std

Running Inference
"""""""""""""""""

.. literalinclude:: ../../../../samples/applications/classification.py
   :language: python
   :start-after: docs_tag: begin_inference
   :end-before: docs_tag: end_inference
   :dedent:

TensorRT inference:

* Takes CV-CUDA tensors as input via ``__cuda_array_interface__``
* Returns CV-CUDA tensors as output
* Runs entirely on GPU
* Output shape: [1, 1000] (1 batch, 1000 ImageNet classes)

Postprocessing Results
""""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/classification.py
   :language: python
   :start-after: docs_tag: begin_postprocessing
   :end-before: docs_tag: end_postprocessing
   :dedent:

The postprocessing:

* Copies results to CPU memory
* Sorts predictions by confidence
* Displays top 5 most confident classes

Expected Output
---------------

Example Output
^^^^^^^^^^^^^^

.. code-block:: text

   Processing image: tabby_tiger_cat.jpg
   Top 5 Predictions (placeholder values):
     1. Class 282: 0.615 (Tiger Cat, if using tabby_tiger_cat)
     2. Class 281: 0.254
     3. Class 283: 0.024
     4. Class 284: 0.001
     5. Class 285: 0.000

.. note::

   Class indices correspond to ImageNet-1K classes. Class 281-293 represent various cat breeds.

Interpreting Results
^^^^^^^^^^^^^^^^^^^^

The output shows:

* **Class Index**: ImageNet class ID (0-999)
* **Confidence Score**: Higher values indicate higher confidence
* **Relative Ranking**: Sorted from most to least confident

For ImageNet class names, refer to the `ImageNet class list <https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`_.

CV-CUDA Operators Used
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.stack`
     - Add batch dimension to single image
   * - :py:func:`cvcuda.resize`
     - Resize image to model input size (configurable, default 224×224)
   * - :py:func:`cvcuda.convertto`
     - Convert uint8 to float32 and scale to [0,1]
   * - :py:func:`cvcuda.normalize`
     - Apply ImageNet mean and standard deviation normalization
   * - :py:func:`cvcuda.reformat`
     - Convert NHWC layout to NCHW for model input

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`cuda_memcpy_h2d() <common_cuda_memcpy_h2d>` - Copy normalization parameters to GPU
* :ref:`cuda_memcpy_d2h() <common_cuda_memcpy_d2h>` - Copy inference results to CPU
* :ref:`TRT <common_trt>` - TensorRT engine wrapper
* :ref:`engine_from_onnx() <common_engine_from_onnx>` - Build TensorRT engine
* :ref:`export_classifier_onnx() <common_export_classifier_onnx>` - Export PyTorch model to ONNX

See Also
--------

* :ref:`Object Detection Sample <sample_object_detection>` - Detection with RetinaNet
* :ref:`Semantic Segmentation Sample <sample_segmentation>` - Pixel-level classification
* :ref:`Common Utilities <sample_common>` - Helper functions reference

References
----------

* `ImageNet Dataset <https://www.image-net.org/>`_
* `ResNet Paper <https://arxiv.org/abs/1512.03385>`_ - Deep Residual Learning
* `TensorRT Documentation <https://docs.nvidia.com/deeplearning/tensorrt/>`_
* `ONNX Documentation <https://onnx.ai/>`_
