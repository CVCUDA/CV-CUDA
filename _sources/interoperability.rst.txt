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

.. _interoperability:

Interoperability
================

CV-CUDA provides interoperability with various GPU-accelerated Python frameworks through the
`CUDA Array Interface <https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html>`_ protocol.
This standard protocol enables efficient zero-copy data exchange between CV-CUDA and other libraries,
allowing you to:

* Convert tensors between frameworks without copying data
* Build end-to-end GPU pipelines that combine multiple libraries
* Leverage CV-CUDA's optimized computer vision operations within your existing workflows
* Move data between CPU and GPU seamlessly

The key to this interoperability is the ``__cuda_array_interface__`` property, which CV-CUDA tensors
expose via the ``.cuda()`` method. This property provides metadata about the GPU buffer (pointer, shape,
dtype, strides) that other frameworks can use to create their own tensor views of the same memory.

.. _interoperability_venv_installation:

Setting Up the Environment
--------------------------

Use the provided installation script to automatically detect your CUDA version and install all required dependencies:

.. code-block:: bash

   cd samples
   ./install_interop_dependencies.sh

This script will:

- Detect your CUDA version (12 or 13)
- Create a virtual environment at ``venv_samples``
- Install all required dependencies for interoperability samples (PyTorch, CuPy, PyCUDA, PyNvVideoCodec, CV-CUDA, etc.)

After installation, activate the virtual environment:

.. code-block:: bash

   source venv_samples/bin/activate

.. _frameworks:

Frameworks
----------

CV-CUDA interoperates with the following frameworks through the CUDA Array Interface protocol:

.. toctree::
   :maxdepth: 1
   :caption: Frameworks

   interop/pytorch
   interop/cuda_python
   interop/numpy
   interop/nvimgcodec
   interop/pynvvideocodec
   interop/cupy
   interop/pycuda


Best Practices
--------------

1. **Memory Management:**

   * Be aware of whether tensors share memory (zero-copy) or are copied
   * When converting from CV-CUDA to PyTorch, use ``.clone()`` if avoiding shared buffers
   * Ensure CUDA buffers are not freed while other frameworks still reference them

2. **Data Layout:**

   * CV-CUDA uses HWC (Height × Width × Channels) layout by default for images
   * PyTorch typically uses CHW (Channels × Height × Width) layout - use ``.permute()`` to convert
   * Be explicit about layout when converting with :py:func:`cvcuda.as_tensor` (e.g., ``cvcuda.as_tensor(obj, "HWC")``)

3. **Device Management:**

   * Ensure all operations occur on the same GPU device
   * Use appropriate CUDA streams for concurrent operations
   * PyTorch, CuPy, and PyCUDA have their own stream management

4. **Performance:**

   * Use batch operations when possible (e.g., PyNvVideoCodec batch decoding)
   * Minimize CPU-GPU transfers
   * Decode/encode directly to/from GPU memory with NvImgCodec/PyNvVideoCodec
   * Consider memory alignment for optimal performance

5. **Error Handling:**

   * Check return codes when using CUDA Python directly
   * Validate tensor shapes and dtypes after conversion
   * Handle codec errors appropriately in pipelines
