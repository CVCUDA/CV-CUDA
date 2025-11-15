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

.. _sample_object_detection:

Object Detection
================

Overview
--------

The Object Detection sample demonstrates GPU-accelerated object detection using CV-CUDA for preprocessing and TensorRT with EfficientNMS for inference. This sample showcases:

* End-to-end object detection pipeline on GPU
* RetinaNet model with ResNet50-FPN backbone
* EfficientNMS TensorRT plugin for fast post-processing
* Bounding box drawing on detected objects
* Integration between CV-CUDA, TensorRT, and visualization

Usage
-----

Detect objects in an image:

.. code-block:: bash

   python3 object_detection.py -i image.jpg

The sample will:

1. Download RetinaNet model weights (first run only)
2. Export model with EfficientNMS to ONNX (first run only)
3. Build TensorRT engine (first run only)
4. Detect objects and draw bounding boxes
5. Save output as ``cvcuda/.cache/cat_detections.jpg``

Specify custom output path:

.. code-block:: bash

   python3 object_detection.py -i street.jpg -o detections.jpg

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
     - cvcuda/.cache/cat_detections.jpg
     - Output image file path with drawn boxes
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

The object detection pipeline consists of:

1. Model setup (RetinaNet+EfficientNMS export and TensorRT engine building)
2. Image loading into GPU
3. Preprocessing (resize and normalize)
4. TensorRT detection
5. Drawing bounding boxes
6. Saving annotated image

Code Walkthrough
^^^^^^^^^^^^^^^^

Model Setup and Export
"""""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/object_detection.py
   :language: python
   :start-after: docs_tag: begin_model_setup
   :end-before: docs_tag: end_model_setup
   :dedent:

The model export process:

* **RetinaNet**: Loads pretrained RetinaNet with ResNet50-FPN backbone
* **EfficientNMS**: Adds TensorRT EfficientNMS plugin to model graph
* **ONNX Export**: Exports complete detection pipeline
* **TensorRT Build**: Compiles to optimized engine

.. note::

   EfficientNMS performs Non-Maximum Suppression (NMS) on GPU, eliminating the need for CPU post-processing.

Loading Input Image
"""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/object_detection.py
   :language: python
   :start-after: docs_tag: begin_read_image
   :end-before: docs_tag: end_read_image
   :dedent:

Image is loaded directly into GPU memory with original dimensions preserved for later bbox scaling.

Preprocessing Pipeline
"""""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/object_detection.py
   :language: python
   :start-after: docs_tag: begin_preprocessing
   :end-before: docs_tag: end_preprocessing
   :dedent:

Preprocessing steps:

1. **Add Batch Dimension**: HWC → NHWC using :py:func:`cvcuda.stack`
2. **Resize**: Scale to target model input size (default 224×224)
3. **Normalize**: Convert to float32 [0,1] range
4. **Reformat**: NHWC → NCHW for model input

Running Inference
"""""""""""""""""

.. literalinclude:: ../../../../samples/applications/object_detection.py
   :language: python
   :start-after: docs_tag: begin_inference
   :end-before: docs_tag: end_inference
   :dedent:

Inference outputs from EfficientNMS:

* **num_detections**: [1, 1] - Number of valid detections
* **boxes**: [1, max_detections, 4] - Bounding boxes [x1, y1, x2, y2]
* **scores**: [1, max_detections] - Confidence scores
* **classes**: [1, max_detections] - Class indices

All outputs are already filtered and sorted by EfficientNMS.

Postprocessing and Visualization
"""""""""""""""""""""""""""""""""

.. literalinclude:: ../../../../samples/applications/object_detection.py
   :language: python
   :start-after: docs_tag: begin_postprocessing
   :end-before: docs_tag: end_postprocessing
   :dedent:

Postprocessing:

1. **Copy to Host**: Transfer detection results to CPU
2. **Scale Boxes**: Scale from model input size to original image size
3. **Create Bounding Boxes**: Build CV-CUDA bounding box objects
4. **Draw Boxes**: Use :py:func:`cvcuda.bndbox` to draw on GPU
5. **Save Result**: Write annotated image

Expected Output
---------------

Console output shows detected bounding boxes:

.. code-block:: text

   Box 0: (45, 67, 312, 389)
   Box 1: (150, 200, 280, 350)
   ...

Each box shows (x1, y1, x2, y2) coordinates in the original image space.

The output image will have red bounding boxes drawn around detected objects (e.g., cat).

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../../content/tabby_tiger_cat.jpg
          :width: 100%

          Original Input Image

     - .. figure:: ../../content/cat_detections.jpg
          :width: 100%

          Output with Detected Objects

Understanding Detection Output
------------------------------

* **Bounding Box Format**: Corner format with (x1, y1) top-left and (x2, y2) bottom-right
* **Confidence Scores**: Range [0, 1] where 1 is highest confidence
* **Class Labels**: RetinaNet is trained on COCO dataset with 80 classes

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
     - Resize to model input size (configurable, default 224×224)
   * - :py:func:`cvcuda.convertto`
     - Convert to float32 and normalize to [0,1]
   * - :py:func:`cvcuda.reformat`
     - Convert NHWC to NCHW
   * - :py:func:`cvcuda.bndbox`
     - Draw bounding boxes on GPU

Common Utilities Used
^^^^^^^^^^^^^^^^^^^^^

* :ref:`read_image() <common_read_image>` - Load image as CV-CUDA tensor
* :ref:`write_image() <common_write_image>` - Save image from CV-CUDA tensor
* :ref:`cuda_memcpy_d2h() <common_cuda_memcpy_d2h>` - Copy detection results to CPU
* :ref:`TRT <common_trt>` - TensorRT engine wrapper
* :ref:`engine_from_onnx() <common_engine_from_onnx>` - Build TensorRT engine
* :ref:`export_retinanet_onnx() <common_export_retinanet_onnx>` - Export detection model with EfficientNMS

See Also
--------

* :ref:`Image Classification Sample <sample_classification>` - Single-class prediction
* :ref:`Semantic Segmentation Sample <sample_segmentation>` - Pixel-level segmentation
* :ref:`Common Utilities <sample_common>` - Helper functions reference
* :py:func:`cvcuda.bndbox` API - Drawing bounding boxes API reference

References
----------

* `RetinaNet Paper <https://arxiv.org/abs/1708.02002>`_ - Focal Loss for Dense Object Detection
* `COCO Dataset <https://cocodataset.org/>`_ - Common Objects in Context
* `TensorRT EfficientNMS <https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin>`_
* `Feature Pyramid Networks <https://arxiv.org/abs/1612.03144>`_
