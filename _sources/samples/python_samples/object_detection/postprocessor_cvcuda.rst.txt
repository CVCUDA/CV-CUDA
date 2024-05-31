..
   # SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _preprocessor_cvcuda:

Object Detection Post-processing Pipeline using CVCUDA
======================================================


CVCUDA helps accelerate the post-processing pipeline of the object detection sample tremendously. Easy interoperability with PyTorch tensors also makes it easy to integrate with PyTorch and other data loaders that supports the tensor layout.

**The exact post-processing operations are:** ::

   Bounding box and score detections from the network -> Interpolate bounding boxes to the image size -> Filter the bounding boxes using NMS -> Render the bounding boxes -> Blur the ROI's

The postprocessing parameters are initialized based on the model architecture

.. literalinclude:: ../../../../../samples/object_detection/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_init_postprocessorcvcuda
   :end-before: end_init_postprocessorcvcuda
   :dedent:

The output bounding boxes are rendered using the cuOSD based bounding box operators. We will use BndBox and BoxBlur operators to render the bounding boxes and blur the regions inside the bounding boxes.
The bounding box display settings and blur parameters are initialized in the BoundingBoxUtilsCvcuda class

.. literalinclude:: ../../../../../samples/object_detection/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_init_cuosd_bboxes
   :end-before: end_init_cuosd_bboxes
   :dedent:

The model divides an input image into a grid and predicts four normalized bounding-box parameters (xc, yc, w, h) and confidence value per output class.
These values then need to be interpolated to the original resolution. The interpolated bounding boxes are then filtered using a clustering algorithms like NMS

.. literalinclude:: ../../../../../samples/object_detection/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_call_filterbboxcvcuda
   :end-before: end_call_filterbboxcvcuda
   :dedent:

We will then invoke the BndBox and BoxBlur operators as follows

.. literalinclude:: ../../../../../samples/object_detection/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_call_cuosd_bboxes
   :end-before: end_call_cuosd_bboxes
   :dedent:

The output buffer is converted to the required layout for the encoder and returned

.. literalinclude:: ../../../../../samples/object_detection/python/pipelines.py
   :language: python
   :linenos:
   :start-after: start_outbuffer
   :end-before: end_outbuffer
   :dedent:
