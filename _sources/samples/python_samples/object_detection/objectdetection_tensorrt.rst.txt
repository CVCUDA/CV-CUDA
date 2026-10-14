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

.. _objectdetection_tensorrt:

Object Detection Inference Using TensorRT
==========================================

The object detection sample in CVCUDA uses the `Peoplenet Model <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet>`_ from NGC. The etlt model must be serialized to tensorrt using the tao-converter. The tensorrt model is provided as input to the sample application. The model supports implicit batch size. We will need to specify the max batch size during serialization.
We will allocate the output Tensors in advance for TensorRT based on the output layer dimensions inferred from the tenorrt model loaded.

.. literalinclude:: ../../../../../samples/object_detection/python/model_inference.py
   :language: python
   :linenos:
   :start-after: begin_init_objectdetectiontensorrt
   :end-before: end_init_objectdetectiontensorrt
   :dedent:

To run the inference the ``__call__`` method is used. It uses the correct I/O bindings and makes sure to use the CUDA stream to perform the forward inference pass. In passing the inputs, we are directly going to pass the data from the CVCUDA tensor without further conversions. The API to do so does involve accessing an internal member named ``__cuda_array_interface__`` as shown in the code below.

.. literalinclude:: ../../../../../samples/object_detection/python/model_inference.py
   :language: python
   :linenos:
   :start-after: begin_call_objectdetectiontensorrt
   :end-before: end_call_objectdetectiontensorrt
   :dedent:
