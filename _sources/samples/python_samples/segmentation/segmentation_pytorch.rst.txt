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

.. _segmentation_pytorch:

Semantic Segmentation Inference Using PyTorch
====================

The semantic segmentation sample in CVCUDA uses the ``fcn_resnet101`` deep learning model from the ``torchvision`` library. Since the model does not come with the softmax layer at the end, we are going to add one. The following code snippet shows how the model is setup for inference use case with PyTorch.

.. literalinclude:: ../../../../../samples/segmentation/python/model_inference.py
   :language: python
   :linenos:
   :start-after: begin_init_segmentationpytorch
   :end-before: end_init_segmentationpytorch
   :dedent:

To run the inference the ``__call__`` method is used. It makes sure to use the CUDA stream and perform the forward inference pass without computing gradients.

.. literalinclude:: ../../../../../samples/segmentation/python/model_inference.py
   :language: python
   :linenos:
   :start-after: begin_call_segmentationpytorch
   :end-before: end_call_segmentationpytorch
   :dedent:
