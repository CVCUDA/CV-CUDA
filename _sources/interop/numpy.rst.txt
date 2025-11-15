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

NumPy
-----

NumPy is the fundamental package for numerical computing in Python. While NumPy arrays reside on the
CPU, you can transfer them to GPU using any of the GPU-accelerated frameworks mentioned above. The
``numpy_interop.py`` example demonstrates four different methods.

**Required Imports:**

.. literalinclude:: ../../../samples/interoperability/numpy_interop.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**Method 1: Via CUDA Python**

.. literalinclude:: ../../../samples/interoperability/numpy_interop.py
   :language: python
   :start-after: docs_tag: begin_numpy_cuda_python
   :end-before: docs_tag: end_numpy_cuda_python
   :dedent: 4

This method gives you the most control over memory allocation and transfer.

**Method 2: Via PyTorch**

.. literalinclude:: ../../../samples/interoperability/numpy_interop.py
   :language: python
   :start-after: docs_tag: begin_numpy_torch
   :end-before: docs_tag: end_numpy_torch
   :dedent: 4

PyTorch provides a convenient ``torch.from_numpy()`` method that creates a tensor sharing memory with
the NumPy array (on CPU), then ``.cuda()`` transfers it to GPU.

**Method 3: Via CuPy**

.. literalinclude:: ../../../samples/interoperability/numpy_interop.py
   :language: python
   :start-after: docs_tag: begin_numpy_cupy
   :end-before: docs_tag: end_numpy_cupy
   :dedent: 4

CuPy's ``cp.asarray()`` directly transfers NumPy arrays to GPU with NumPy-compatible semantics.

**Method 4: Via PyCUDA**

.. literalinclude:: ../../../samples/interoperability/numpy_interop.py
   :language: python
   :start-after: docs_tag: begin_numpy_pycuda
   :end-before: docs_tag: end_numpy_pycuda
   :dedent: 4

PyCUDA provides ``gpuarray.to_gpu()`` for straightforward CPU-to-GPU transfer.

**Choosing a Method:**

* **PyTorch** - Easy to integrate with existing PyTorch workflows, but has a large download size
* **CuPy** - NumPy-like syntax for GPU operations, but requires building during installation
* **PyCUDA** - Good if you are already using PyCUDA in your pipelines, but requires building during installation
* **CUDA Python** - Best for maximum control and custom CUDA integration, but requires low-level management and CUDA knowledge

**Complete Example:** See ``samples/interoperability/numpy_interop.py``
