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

.. _sample_segmentation:

Semantic Segmentation
=====================

Overview
--------

The Semantic Segmentation sample demonstrates pixel-level classification using CV-CUDA for preprocessing and TensorRT for inference. This advanced sample showcases:

* Dense pixel-wise prediction for semantic segmentation
* FCN-ResNet101 model for accurate segmentation
* Advanced post-processing with bilateral filtering
* Background blurring with foreground preservation
* Smooth edge generation using joint bilateral filter

Usage
-----

Segment an image and create a blurred background effect:

.. code-block:: bash

   python3 segmentation.py -i image.jpg

The sample will:

1. Download FCN-ResNet101 model (first run only)
2. Export to ONNX and build TensorRT engine (first run only)
3. Segment the image to find objects (e.g., cats)
4. Create smooth mask with bilateral filtering
5. Blur background and composite with foreground
6. Save result as ``cvcuda/.cache/cat_segmented.jpg``

Specify custom output path:

.. code-block:: bash

   python3 segmentation.py -i portrait.jpg -o segmented_portrait.jpg

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
     - cvcuda/.cache/cat_segmented.jpg
     - Output segmented image path
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

The segmentation pipeline consists of:

1. Model setup (FCN-ResNet101 export and TensorRT engine building)
2. Image loading
3. Preprocessing (resize and ImageNet normalization)
4. Semantic segmentation inference
5. Post-processing (extract class probabilities, refine masks with bilateral filtering)
6. Background blur and compositing
7. Saving result

Code Walkthrough
^^^^^^^^^^^^^^^^

Model Setup
"""""""""""

.. literalinclude:: ../../../../samples/applications/segmentation.py
   :language: python
   :start-after: docs_tag: begin_model_setup
   :end-before: docs_tag: end_model_setup
   :dedent:

Model details:

* **FCN-ResNet101**: Fully Convolutional Network with ResNet101 backbone
* **Training**: Pretrained on COCO+VOC datasets
* **Classes**: 21 classes (Pascal VOC) including background, person, cat, dog, etc.
* **Output**: Dense predictions for each pixel

Loading and Preprocessing
""""""""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/segmentation.py
   :language: python
   :start-after: docs_tag: begin_preprocessing
   :end-before: docs_tag: end_preprocessing
   :dedent:

Preprocessing includes:

1. **Normalization Setup**: ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
2. **Batching**: Add batch dimension (HWC → NHWC)
3. **Resizing**: Scale to target model input size (default 224×224)
4. **Float Conversion**: uint8 [0,255] → float32 [0,1]
5. **Normalization**: ``(x - mean) / std``
6. **Layout**: NHWC → NCHW

Running Inference
"""""""""""""""""

.. literalinclude:: ../../../../samples/applications/segmentation.py
   :language: python
   :start-after: docs_tag: begin_inference
   :end-before: docs_tag: end_inference
   :dedent:

Inference output:

* **Shape**: [1, 21, H, W] - Batch × Classes × Height × Width
* **Values**: Probabilities (post-softmax) for each class, range [0, 1]
* **Semantics**: Higher values indicate higher confidence for that class

Post-Processing and Effects
""""""""""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/segmentation.py
   :language: python
   :start-after: docs_tag: begin_postprocessing
   :end-before: docs_tag: end_postprocessing
   :dedent:

Advanced post-processing:

1. **Class Extraction**: Extract probability map for target class (cat = class 8)
2. **Scale to uint8**: Scale probabilities [0, 1] to [0, 255] for mask
3. **Upscaling**: Resize mask to original image size
4. **Background Blur**: Apply Gaussian blur to create blurred version
5. **Bilateral Filtering**: Smooth mask edges while preserving boundaries
6. **Compositing**: Blend original foreground with blurred background

Joint Bilateral Filter
^^^^^^^^^^^^^^^^^^^^^^

The joint bilateral filter (:py:func:`cvcuda.joint_bilateral_filter`) is key to quality:

* **Purpose**: Smooth mask while respecting image edges
* **Joint**: Uses grayscale image to guide filtering
* **Parameters**: diameter=5, sigma_color=50, sigma_space=1
* **Result**: Smooth transitions without halo artifacts

Expected Output
---------------

The output shows the segmented object (e.g., cat) in focus with a smoothly blurred background, creating a portrait-style effect similar to DSLR bokeh.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_segmented.jpg
          :width: 100%

          Output with Segmented Background

Understanding Segmentation
---------------------------

FCN Output Format
^^^^^^^^^^^^^^^^^

FCN outputs a probability map for each class:

.. code-block:: python

   output.shape = [1, 21, 224, 224]
   # output[0, 8, :, :] = probabilities for "cat" class at each pixel

Class Indices (Pascal VOC):

* 0: Background
* 8: Cat
* 12: Dog
* 15: Person

Modify ``class_index`` in the code to segment different objects.

CV-CUDA Operators Used
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operator
     - Purpose
   * - :py:func:`cvcuda.stack`
     - Add batch dimension
   * - :py:func:`cvcuda.resize`
     - Resize image and masks to different resolutions
   * - :py:func:`cvcuda.convertto`
     - Convert data types and normalize
   * - :py:func:`cvcuda.normalize`
     - Apply ImageNet normalization
   * - :py:func:`cvcuda.reformat`
     - Convert between NHWC and NCHW layouts
   * - :py:func:`cvcuda.gaussian`
     - Blur background for aesthetic effect
   * - :py:func:`cvcuda.cvtcolor`
     - Convert RGB to grayscale for bilateral filter
   * - :py:func:`cvcuda.joint_bilateral_filter`
     - Smooth mask edges while preserving boundaries
   * - :py:func:`cvcuda.composite`
     - Blend foreground and blurred background

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save result
* :ref:`cuda_memcpy_h2d() <common_cuda_memcpy_h2d>` - Upload normalization parameters
* :ref:`cuda_memcpy_d2h() <common_cuda_memcpy_d2h>` - Download segmentation results
* :ref:`zero_copy_split() <common_zero_copy_split>` - Split batch efficiently
* :ref:`TRT <common_trt>` - TensorRT wrapper
* :ref:`engine_from_onnx() <common_engine_from_onnx>` - Build engine
* :ref:`export_segmentation_onnx() <common_export_segmentation_onnx>` - Export FCN model

See Also
--------

* :ref:`Image Classification Sample <sample_classification>` - Related preprocessing
* :ref:`Object Detection Sample <sample_object_detection>` - Bounding box detection
* :ref:`Gaussian Blur Operator <sample_gaussian>` - Blur effects
* :ref:`Common Utilities <sample_common>` - Helper functions

References
----------

* `FCN Paper <https://arxiv.org/abs/1411.4038>`_ - Fully Convolutional Networks for Semantic Segmentation
* `COCO Dataset <https://cocodataset.org/>`_
