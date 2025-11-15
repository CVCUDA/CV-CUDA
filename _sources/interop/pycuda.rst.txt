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

PyCUDA
------

PyCUDA provides Python access to CUDA with more low-level control than other frameworks. It's useful
when you need fine-grained control over GPU memory and kernel execution.

**Key Points:**

* PyCUDA requires ``import pycuda.autoinit`` to initialize the CUDA context
* Use ``gpuarray.to_gpu()`` to transfer NumPy arrays to GPU
* PyCUDA ``GPUArray`` can be converted to CV-CUDA using :py:func:`cvcuda.as_tensor`
* Converting back requires manually constructing a ``GPUArray`` with the shared GPU pointer

**Required Imports:**

.. literalinclude:: ../../../samples/interoperability/pycuda_interop.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**PyCUDA to CV-CUDA:**

.. literalinclude:: ../../../samples/interoperability/pycuda_interop.py
   :language: python
   :start-after: docs_tag: begin_pycuda_to_cvcuda
   :end-before: docs_tag: end_pycuda_to_cvcuda
   :dedent: 4

**CV-CUDA to PyCUDA:**

.. literalinclude:: ../../../samples/interoperability/pycuda_interop.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_to_pycuda
   :end-before: docs_tag: end_cvcuda_to_pycuda
   :dedent: 4

Note that converting from CV-CUDA to PyCUDA requires extracting the GPU pointer from the CUDA Array
Interface and manually constructing a ``GPUArray`` object. The ``gpudata`` parameter takes the pointer
directly.

**Complete Example:** See ``samples/interoperability/pycuda_interop.py``
