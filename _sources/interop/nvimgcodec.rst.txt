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

NvImgCodec
----------

NvImgCodec is NVIDIA's hardware-accelerated image codec library, providing high-performance decoding
and encoding for various image formats (JPEG, PNG, TIFF, etc.). It can decode images directly to GPU
memory.

**Key Points:**

* Decodes images directly to GPU memory
* Supports hardware-accelerated encoding
* Integrates seamlessly with CV-CUDA via CUDA Array Interface

**Required Imports:**

.. literalinclude:: ../../../samples/interoperability/nvimgcodec_interop.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**Setup NvImgCodec:**

.. literalinclude:: ../../../samples/interoperability/nvimgcodec_interop.py
   :language: python
   :start-after: docs_tag: begin_init_nvimgcodec
   :end-before: docs_tag: end_init_nvimgcodec
   :dedent: 4

**NvImgCodec to CV-CUDA:**

.. literalinclude:: ../../../samples/interoperability/nvimgcodec_interop.py
   :language: python
   :start-after: docs_tag: begin_nvimgcodec_to_cvcuda
   :end-before: docs_tag: end_nvimgcodec_to_cvcuda
   :dedent: 4

The second parameter ``"HWC"`` specifies the layout (Height × Width × Channels). NvImgCodec images
are decoded directly to GPU memory and can be immediately converted to CV-CUDA tensors.

**Process with CV-CUDA:**

.. literalinclude:: ../../../samples/interoperability/nvimgcodec_interop.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_resize
   :end-before: docs_tag: end_cvcuda_resize
   :dedent: 4

You can apply any CV-CUDA operation to the tensor. Here we resize the image to 224×224 using cubic
interpolation.

**CV-CUDA to NvImgCodec:**

.. literalinclude:: ../../../samples/interoperability/nvimgcodec_interop.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_to_nvimgcodec
   :end-before: docs_tag: end_cvcuda_to_nvimgcodec
   :dedent: 4

The processed CV-CUDA tensor can be converted back to an NvImgCodec image and encoded to disk, all
without leaving GPU memory.

**Typical Use Cases:**

* Batch image preprocessing for inference
* Image transformation pipelines (resize, color conversion, etc.)
* High-throughput image processing
* Building end-to-end GPU pipelines from disk to inference

**Complete Example:** See ``samples/interoperability/nvimgcodec_interop.py``
