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

PyTorch
-------

PyTorch is one of the most popular deep learning frameworks.
CV-CUDA allows seamless interoperability with PyTorch, allowing you to use CV-CUDA's
optimized computer vision operations within your existing PyTorch workflows.

**Key Points:**

* PyTorch tensors must be on GPU (``.cuda()``) to convert to CV-CUDA
* Use :py:func:`cvcuda.as_tensor` to convert PyTorch tensors to CV-CUDA
* Use ``torch.as_tensor()`` to convert CV-CUDA tensors back to PyTorch
* You can use ``.clone()`` when converting from CV-CUDA to avoid shared memory issues, but this will incur a memcpy operation and potential performance degradation. Both torch and cvcuda support zero-copy intepability through their ``as_tensor`` functions.

**Required Imports:**

.. literalinclude:: ../../../samples/interoperability/pytorch_interop.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**PyTorch to CV-CUDA:**

.. literalinclude:: ../../../samples/interoperability/pytorch_interop.py
   :language: python
   :start-after: docs_tag: begin_torch_to_cvcuda
   :end-before: docs_tag: end_torch_to_cvcuda
   :dedent: 4

The PyTorch tensor must be moved to GPU using ``.cuda()`` before conversion. The :py:func:`cvcuda.as_tensor`
function creates a CV-CUDA tensor that shares the same GPU memory as the PyTorch tensor using the ``__cuda_array_interface__`` protocol.

**CV-CUDA to PyTorch:**

.. literalinclude:: ../../../samples/interoperability/pytorch_interop.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_to_torch
   :end-before: docs_tag: end_cvcuda_to_torch
   :dedent: 4

The ``.clone()`` call is important to avoid multiple tensors sharing the same GPU buffer, which can
lead to unexpected behavior if one tensor is modified or deallocated.

**Complete Example:** See ``samples/interoperability/pytorch_interop.py``
