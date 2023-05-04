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

Semantic Segmentation Post-processing Pipeline using CVCUDA
====================


CVCUDA helps accelerate the post-processing pipeline of the semantic segmentation sample tremendously. Easy interoperability with PyTorch tensors also makes it easy to integrate with PyTorch and other data loaders that supports the tensor layout.

**The exact post-processing operations are:** ::

   Create Binary mask -> Upscale the mask -> Blur the input frames -> Joint Bilateral filter to smooth the mask -> Overlay the masks onto the original frame

Since the network outputs the class probabilities (0-1) for all the classes supported by the network, we must first take out the class of interest from it and upscale its values to bring it in the uint8 (0-255) range. These operations will be done using PyTorch math and the resulting tensor will be converted to CVCUDA.


.. literalinclude:: ../../../../../samples/segmentation/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_proces_probs
   :end-before: end_proces_probs
   :dedent:

The remaining the pipeline code is easy to follow along.

.. literalinclude:: ../../../../../samples/segmentation/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_postproc_pipeline
   :end-before: end_postproc_pipeline
   :dedent:
