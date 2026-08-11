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

.. literalinclude:: ../../../../samples/interoperability/cupy_interop.py
   :language: python
   :start-after: docs_tag: begin_imports
   :end-before: docs_tag: end_imports

**CuPy to CV-CUDA:**

.. literalinclude:: ../../../../samples/interoperability/cupy_interop.py
   :language: python
   :start-after: docs_tag: begin_cupy_to_cvcuda
   :end-before: docs_tag: end_cupy_to_cvcuda
   :dedent: 4

CuPy arrays are created directly on the GPU and can be immediately converted to CV-CUDA tensors.

**CV-CUDA to CuPy:**

.. literalinclude:: ../../../../samples/interoperability/cupy_interop.py
   :language: python
   :start-after: docs_tag: begin_cvcuda_to_cupy
   :end-before: docs_tag: end_cvcuda_to_cupy
   :dedent: 4

The ``cupy.asarray()`` function recognizes the CUDA Array Interface and creates a CuPy array that
views the same GPU memory as the CV-CUDA tensor.

**Stream synchronization:**

CV-CUDA streams are non-blocking (``cudaStreamNonBlocking``), so they do not implicitly
synchronize with CuPy's default stream (CUDA stream 0).  CV-CUDA inserts the
necessary cross-stream barrier automatically when wrapping an external buffer
via :py:func:`cvcuda.as_tensor`; no manual ``cupy.cuda.Stream.null.synchronize()``
or equivalent is required:

.. code-block:: python

    src_cp = cupy.asarray(host_array)            # H2D on CuPy's default stream
    src_nv = cvcuda.as_tensor(src_cp, "NHWC")    # CAI stream captured here
    target = cvcuda.Stream()                     # dedicated non-blocking stream
    with target:
        out_nv = cvcuda.flip(src_nv, -1, stream=target)  # waits for src_cp
    target.sync()                                # done; result is valid

To opt out (e.g., when you've manually synchronized), export your buffer with
``stream: -1`` in its CAI dict.

.. note::

   **Defensive synchronization for CuPy buffers.**  CV-CUDA's automatic CAI v3
   honoring uses the producer's ``stream`` field to insert an event-based barrier.
   That works only if the producer reports the stream the buffer was actually
   written on.  CuPy's CAI implementation instead reports the cupy-current stream
   at the moment ``__cuda_array_interface__`` is evaluated; if a CuPy buffer was
   filled inside a ``with cupy.cuda.Stream(...):`` block but read by CV-CUDA from
   outside that block, CuPy reports the consumer's current stream (typically the
   legacy default) rather than the producer's stream.  An event-based barrier on
   the wrong stream wouldn't capture the producer's work, especially when the
   producer used a non-blocking stream that doesn't synchronize implicitly with
   the legacy default.

   To handle this defensively, CV-CUDA falls back to ``cudaDeviceSynchronize``
   on the first use of any externally-wrapped buffer whose CAI ``stream`` is a
   default-stream sentinel (``stream`` ∈ {``0``, ``1``, ``2``}).  This is
   correct regardless of which stream the producer actually used.  The cost is
   one host-side device sync per ``cvcuda.as_tensor`` of a non-cvcuda buffer;
   subsequent CV-CUDA ops on the same wrapper take the event-based fast path.
   CV-CUDA → CV-CUDA chains advertise the actual writer stream as a real
   pointer in CAI (not a sentinel), so they keep the event-based fast path
   end-to-end.

   This was reproduced on CuPy 13.6.0 and 14.0.1 (cupy-cuda12x) on CUDA 12.5;
   ``tests/cvcuda/python/test_cai_input_stream_race.py`` is the regression
   guard.

**Complete Example:** See ``samples/interoperability/cupy_interop.py``
