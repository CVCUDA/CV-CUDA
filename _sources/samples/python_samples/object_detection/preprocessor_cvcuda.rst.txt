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

Object Detection Pre-processing Pipeline using CVCUDA
=====================================================

CVCUDA helps accelerate the pre-processing pipeline of the object detection sample tremendously. Easy interoperability with PyTorch tensors also makes it easy to integrate with PyTorch and other data loaders that supports the tensor layout.

**The exact pre-processing operations are:** ::

   Tensor Conversion -> Resize -> Convert Datatype(Float) -> Normalize (to 0-1 range) -> Convert to NCHW

The Tensor conversion operation helps in converting non CVCUDA tensors/data to CVCUDA tensors.

.. literalinclude:: ../../../../../samples/object_detection/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_tensor_conversion
   :end-before: end_tensor_conversion
   :dedent:

The remaining the pipeline code is easy to follow along with only basic operations such as resize and normalized being used.

.. literalinclude:: ../../../../../samples/object_detection/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_preproc_pipeline
   :end-before: end_preproc_pipeline
   :dedent:
