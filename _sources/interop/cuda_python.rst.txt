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

CUDA Python
-----------

CUDA Python provides direct Python bindings to the CUDA Runtime API (``cuda.bindings.runtime``),
offering the most low-level control over CUDA operations. Since CUDA Python is equivalent to using
CUDA directly, you can use it to allocate memory on GPU, copy data between CPU and GPU, and
even write your own CUDA kernels while using CV-CUDA.

**Common Utilities**

To simplify working with CUDA Python, we provide utility functions in ``cuda_python_common.py``:

**CudaBuffer Class:**

The ``CudaBuffer`` class is a lightweight wrapper that implements the ``__cuda_array_interface__``
protocol, making raw CUDA memory accessible to CV-CUDA and other frameworks:

.. literalinclude:: ../../../samples/interoperability/cuda_python_common.py
   :language: python
   :start-after: docs_tag: begin_cuda_buffer
   :end-before: docs_tag: end_cuda_buffer

Key features:

* Allocates CUDA memory using ``cudart.cudaMalloc()``
* Implements ``__cuda_array_interface__`` for zero-copy interop
* Manages memory lifetime (frees memory in ``__del__`` if it owns it)
* Can wrap existing CUDA pointers without taking ownership via ``from_cuda()`` class method

**Memory Copy Utilities:**

.. literalinclude:: ../../../samples/interoperability/cuda_python_common.py
   :language: python
   :start-after: docs_tag: being_cuda_memcpy_h2d
   :end-before: docs_tag: end_cuda_memcpy_h2d

.. literalinclude:: ../../../samples/interoperability/cuda_python_common.py
   :language: python
   :start-after: docs_tag: begin_cuda_memcpy_d2h
   :end-before: docs_tag: end_cuda_memcpy_d2h

These functions wrap ``cudart.cudaMemcpy()`` to transfer data between host (CPU) and device (GPU).
Each function supports the ``__cuda_array_interface__`` or object with interface as the device array.
You can utilize :py:class:`cvcuda.Tensor` via the ``.cuda()`` method.

----

**Approach 1: Custom CudaBuffer**

This approach uses the custom ``CudaBuffer`` class to manually allocate CUDA memory and transfer data.

**Required Imports:**

.. literalinclude:: ../../../samples/interoperability/cuda_python_interop_1.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**CUDA Python to CV-CUDA:**

.. literalinclude:: ../../../samples/interoperability/cuda_python_interop_1.py
   :language: python
   :start-after: docs_tag: begin_cuda_python_to_cvcuda
   :end-before: docs_tag: end_cuda_python_to_cvcuda
   :dedent: 4

**CV-CUDA to CUDA Python:**

.. literalinclude:: ../../../samples/interoperability/cuda_python_interop_1.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_to_cuda_python
   :end-before: docs_tag: end_cvcuda_to_cuda_python
   :dedent: 4

**When to use this approach:**

* You need full control over memory allocation
* You want explicit memory management

**Complete Example:** See ``samples/interoperability/cuda_python_interop_1.py``

----

**Approach 2: Using CV-CUDA Tensor as Buffer**

This approach leverages CV-CUDA's own memory allocation by using :py:class:`cvcuda.Tensor` as a
raw buffer, then copying data into it. This is useful when you want CV-CUDA to manage the memory.

**Required Imports:**

.. literalinclude:: ../../../samples/interoperability/cuda_python_interop_2.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**CUDA Python to CV-CUDA:**

.. literalinclude:: ../../../samples/interoperability/cuda_python_interop_2.py
   :language: python
   :start-after: docs_tag: begin_cuda_python_to_cvcuda
   :end-before: docs_tag: end_cuda_python_to_cvcuda
   :dedent: 4

**Key differences from Approach 1:**

* Uses :py:class:`cvcuda.Tensor` to allocate GPU memory instead of ``CudaBuffer``
* CV-CUDA manages the memory lifecycle

**Conversion to Tensor:**

.. literalinclude:: ../../../samples/interoperability/cuda_python_interop_2.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_to_cuda_python
   :end-before: docs_tag: end_cvcuda_to_cuda_python
   :dedent: 4

**When to use this approach:**

* You want CV-CUDA to manage memory allocation
* You're working with contiguous data that will be reshaped
* You want simpler memory management (no manual ``CudaBuffer``)
* You're already using CV-CUDA Tensors in your pipeline

**Complete Example:** See ``samples/interoperability/cuda_python_interop_2.py``

----

**Summary of CUDA Python Approaches**

.. list-table:: Comparison of CUDA Python Approaches
   :header-rows: 1
   :widths: 20 25 25 30

   * - Approach
     - Memory Allocation
     - Data Transfer
     - Best Use Case
   * - Approach 1
     - Custom ``CudaBuffer``
     - Manual ``cuda_memcpy_h2d/d2h``
     - Full control, interfacing with existing CUDA code
   * - Approach 2
     - :py:class:`cvcuda.Tensor`
     - Manual ``cuda_memcpy_h2d/d2h``
     - Let CV-CUDA manage memory, simpler code

Both approaches achieve zero-copy interoperability between CUDA Python and CV-CUDA, but differ in
memory management and allocation strategy.
