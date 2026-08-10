..
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. literalinclude:: ../../../../samples/interoperability/pytorch_interop.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**PyTorch to CV-CUDA:**

.. literalinclude:: ../../../../samples/interoperability/pytorch_interop.py
   :language: python
   :start-after: docs_tag: begin_torch_to_cvcuda
   :end-before: docs_tag: end_torch_to_cvcuda
   :dedent: 4

The PyTorch tensor must be moved to GPU using ``.cuda()`` before conversion. The :py:func:`cvcuda.as_tensor`
function creates a CV-CUDA tensor that shares the same GPU memory as the PyTorch tensor using the ``__cuda_array_interface__`` protocol.

**CV-CUDA to PyTorch:**

.. literalinclude:: ../../../../samples/interoperability/pytorch_interop.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_to_torch
   :end-before: docs_tag: end_cvcuda_to_torch
   :dedent: 4

The ``.clone()`` call is important to avoid multiple tensors sharing the same GPU buffer, which can
lead to unexpected behavior if one tensor is modified or deallocated.

**Stream synchronization:**

CV-CUDA streams are non-blocking, so they do not implicitly synchronize with PyTorch's
current stream.  CV-CUDA inserts the necessary cross-stream barrier automatically when
wrapping a PyTorch tensor via :py:func:`cvcuda.as_tensor`; no explicit
``torch.cuda.synchronize()`` is required:

.. code-block:: python

    src = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8).cuda()
    src_nv = cvcuda.as_tensor(src, "NHWC")
    target = cvcuda.Stream()
    with target:
        out_nv = cvcuda.flip(src_nv, -1, stream=target)
    target.sync()

To opt out (e.g., when you've manually synchronized), export your buffer with
``stream: -1`` in its CAI dict.

.. note::

   **Defensive synchronization for PyTorch buffers.**  PyTorch's
   ``__cuda_array_interface__`` is **CAI v2** — it carries no ``stream`` field at
   all.  Per the CAI v3 spec, the ``stream`` field uses sentinel integers (``0``
   = "no stream associated, consumer must synchronize", ``1`` = legacy default
   stream, ``2`` = per-thread default stream, other positive integers = real
   stream handles).  Because PyTorch advertises v2, CV-CUDA has no producer-stream
   information and would race against any work PyTorch had queued on a
   non-default (non-blocking) stream.

   To handle this defensively, CV-CUDA falls back to ``cudaDeviceSynchronize``
   on the first use of any externally-wrapped buffer whose producer stream is
   unknown or is a default-stream sentinel.  PyTorch tensors fall under "unknown"
   (CAI v2, no stream field) and so always pay one host-side device sync at
   ``cvcuda.as_tensor`` time; subsequent CV-CUDA ops on the same wrapper take the
   event-based fast path.  CV-CUDA → CV-CUDA chains advertise the actual writer
   stream in CAI v3 and stay on the fast path end-to-end.

   This was verified on **PyTorch 2.9.0+cu128**.  If a future PyTorch release
   adopts CAI v3 and reports a real writer stream, CV-CUDA will pick that up
   automatically and skip the defensive sync.

**Complete Example:** See ``samples/interoperability/pytorch_interop.py``
