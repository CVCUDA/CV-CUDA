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

.. _sample_applications:

Applications
============

End-to-end deep learning pipelines combining preprocessing, inference, and post-processing.

Overview
--------

The application samples demonstrate complete workflows for common computer vision tasks:

* **Hello World** - Introduction to CV-CUDA with GPU-only image processing
* **Image Classification** - ResNet50 classification with TensorRT inference
* **Object Detection** - RetinaNet detection with bounding box visualization
* **Semantic Segmentation** - FCN-ResNet101 with artistic background effects

All applications showcase:

* GPU-accelerated preprocessing with CV-CUDA
* TensorRT inference integration
* Post-processing and visualization
* Model export from PyTorch to ONNX to TensorRT

Application Samples
-------------------

.. toctree::
   :maxdepth: 1

   Hello World <applications/hello_world>
   Image Classification <applications/classification>
   Object Detection <applications/object_detection>
   Semantic Segmentation <applications/segmentation>

Common Patterns
---------------

Model Export
^^^^^^^^^^^^

All applications follow this pattern for model preparation:

1. **Export PyTorch model to ONNX** (first run only)
2. **Build TensorRT engine from ONNX** (first run only, cached)
3. **Load cached engine** (subsequent runs)

Preprocessing Pipeline
^^^^^^^^^^^^^^^^^^^^^^

Standard preprocessing steps:

1. Load image with :ref:`read_image() <common_read_image>`
2. Add batch dimension with :py:func:`cvcuda.stack`
3. Resize to model input size with :py:func:`cvcuda.resize`
4. Convert to float32 with :py:func:`cvcuda.convertto`
5. Normalize (if needed) with :py:func:`cvcuda.normalize`
6. Reformat to NCHW with :py:func:`cvcuda.reformat`

Inference
^^^^^^^^^

Using the :ref:`TRT wrapper <common_trt>`:

.. code-block:: python

   model = TRT(engine_path)
   outputs = model([input_tensor])

See Also
--------

* :ref:`Operators <sample_operators>` - Individual CV-CUDA operators
* :ref:`Common Utilities <sample_common>` - Helper functions
* :ref:`Hello World <sample_hello_world>` - Simple introduction
