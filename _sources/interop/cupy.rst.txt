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

CuPy
----

CuPy is a NumPy-compatible GPU array library that provides GPU acceleration for numerical operations.
It's an excellent choice when you want to use NumPy-like operations on the GPU.

**Key Points:**

* CuPy arrays are already on GPU, no explicit device transfer needed
* Use :py:func:`cvcuda.as_tensor` to convert CuPy arrays to CV-CUDA
* Use ``cupy.asarray()`` to convert CV-CUDA tensors back to CuPy
* CuPy provides the most NumPy-like interface for GPU arrays

**Required Imports:**

.. literalinclude:: ../../../samples/interoperability/cupy_interop.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**CuPy to CV-CUDA:**

.. literalinclude:: ../../../samples/interoperability/cupy_interop.py
   :language: python
   :start-after: docs_tag: begin_cupy_to_cvcuda
   :end-before: docs_tag: end_cupy_to_cvcuda
   :dedent: 4

CuPy arrays are created directly on the GPU and can be immediately converted to CV-CUDA tensors.

**CV-CUDA to CuPy:**

.. literalinclude:: ../../../samples/interoperability/cupy_interop.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_to_cupy
   :end-before: docs_tag: end_cvcuda_to_cupy
   :dedent: 4

The ``cupy.asarray()`` function recognizes the CUDA Array Interface and creates a CuPy array that
views the same GPU memory as the CV-CUDA tensor.

**Complete Example:** See ``samples/interoperability/cupy_interop.py``
