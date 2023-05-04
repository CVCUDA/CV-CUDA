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

.. _segmentation_tensorrt:

Semantic Segmentation Inference Using TensorRT
====================

The semantic segmentation sample in CVCUDA uses the ``fcn_resnet101`` deep learning model from the ``torchvision`` library. Since the model does not come with the softmax layer at the end, we are going to add one. The following code snippet shows how the model is setup for inference use case with TensorRT.

TensorRT requires a serialized TensorRT engine to run the inference. One can generate such an engine by first converting an existing PyTorch model to ONNX and then converting the ONNX to a TensorRT engine. The serialized TensorRT engine is good to work on the specific GPU with the maximum batch size it was given at the creation time. Since ONNX and TensorRT model generation is a time consuming operation, we avoid doing this every-time by first checking if one of those already exists (most likely due to a previous run of this sample.) If so, we simply use those models rather than generating a new one.

Finally we take care of setting up the I/O bindings. We allocate the output Tensors in advance for TensorRT. Helper methods such as ``convert_onnx_to_tensorrt`` and ``setup_tensort_bindings`` are defined in the helper script file ``samples/common/python/trt_utils.py``

.. literalinclude:: ../../../../../samples/segmentation/python/model_inference.py
   :language: python
   :linenos:
   :start-after: begin_init_segmentationtensorrt
   :end-before: end_init_segmentationtensorrt
   :dedent:

To run the inference the ``__call__`` method is used. It uses the correct I/O bindings and makes sure to use the CUDA stream to perform the forward inference pass. In passing the inputs, we are directly going to pass the data from the CVCUDA tensor without further conversions. The API to do so does involve accessing an internal member named ``__cuda_array_interface__`` as shown in the code below.

.. literalinclude:: ../../../../../samples/segmentation/python/model_inference.py
   :language: python
   :linenos:
   :start-after: begin_call_segmentationtensorrt
   :end-before: end_call_segmentationtensorrt
   :dedent:
