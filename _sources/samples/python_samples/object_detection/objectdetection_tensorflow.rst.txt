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

.. _objectdetection_tensorflow:

Object Detection Inference Using TensorFlow
==========================================

The object detection sample in CVCUDA uses the `Peoplenet Model <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet>`_ from NGC. The HDF5 model file is downloaded from NGC. We use appropriate GPU device with Keras to load the model.

.. literalinclude:: ../../../../../samples/object_detection/python/model_inference.py
   :language: python
   :linenos:
   :start-after: begin_init_objectdetectiontensorflow
   :end-before: end_init_objectdetectiontensorflow
   :dedent:

To run the inference the ``__call__`` method is used. It converts incoming tensor from formats such as ``torch.Tensor``, ``nvcv.Tensor`` or ``numpy.ndarray`` to a ``tensorflow.Tensor`` object. Since both PyTorch and CVCUDA tensors support the dlpack interface, we use that to convert them to the tensorflow tensor. At the time of this writing, a bug prevents conversion of tensors which are not flattened out beforehand. Hence we temporarily note down the shape of input tensor, flatten it out, use dlpack to convert to tensorflow.Tensor and then reshape it back to its original shape.


.. literalinclude:: ../../../../../samples/object_detection/python/model_inference.py
   :language: python
   :linenos:
   :start-after: begin_call_objectdetectiontensorflow
   :end-before: end_call_objectdetectiontensorflow
   :dedent:
