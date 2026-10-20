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

.. _object_cache:

Object Cache
============

CV-CUDA includes an internal resource management system that caches allocated objects for efficient reuse.
Objects used within CV-CUDA, such as :py:class:`cvcuda.Image`, :py:class:`cvcuda.Tensor`, :py:class:`cvcuda.ImageBatchVarShape`, and :py:class:`cvcuda.TensorBatch`, are automatically managed by the CV-CUDA cache.

.. note::
   Only Python objects are cached, there is no C/C++ object caching.

.. note::
   Cache limits and byte accounting are per CUDA device and shared across threads.

Wrapped vs Non-Wrapped Objects
------------------------------

The cache distinguishes between two types of objects based on memory ownership:

**Non-wrapped objects** are allocated by CV-CUDA and increase the cache size:

.. literalinclude:: ../../../samples/object_cache/basic.py
   :language: python
   :start-after: # docs-start: non-wrapped
   :end-before: # docs-end: non-wrapped

**Wrapped objects** wrap externally-managed memory and do not increase the cache size:

.. literalinclude:: ../../../samples/object_cache/basic_wrapped.py
   :language: python
   :start-after: # docs-start: wrapped
   :end-before: # docs-end: wrapped

Del and Garbage Collection
--------------------------

Both :py:class:`cvcuda.Image` and :py:class:`cvcuda.Tensor` are managed by CV-CUDA if they are allocated via CV-CUDA.
When a :py:class:`cvcuda.Tensor` or :py:class:`cvcuda.Image` has been allocated by CV-CUDA and it goes out of scope, the underlying memory is not released.
Instead, it is stored and will be reused for future allocations via the CV-CUDA object cache.

For example, when using the ``del`` keyword in Python, only the reference to the object is removed.
Reuse requires that no application-visible object, view, or exported buffer can still
observe the allocation, and that pending GPU work remains correctly ordered.
As such, it is best practice to not attempt to manually free memory.

.. note::
   You can manually free all cached memory allocations by calling :py:func:`cvcuda.clear_cache`.

Cache Reuse
-----------

When a CV-CUDA object goes out of scope, its memory is retained in the cache for efficient reuse.
Owned tensors can reuse allocations on the same device with the same data type and layout;
owned images require the same device and image format. An exact shape and stride match is
preferred. Otherwise, the smallest available idle allocation that satisfies the requested
byte size and alignment is reused, with a new descriptor for the requested shape and strides.
Pending tensor reuse still requires an exact shape and stride match.

The dimensions may differ, including a wider image with fewer rows, as long as the calculated
allocation requirements fit. The cache continues to count the allocation's full capacity,
even when the new object uses fewer bytes:

.. literalinclude:: ../../../samples/object_cache/reuse.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

In this example, ``tensor2`` has a different shape but fits in ``tensor1``'s allocation.
No new GPU memory allocation occurs.

Cache reuse also applies to wrapped objects, improving efficiency even though they don't consume cache memory.

Controlling Cache Growth
------------------------

The cache can still grow when requests exceed the capacity of available allocations,
use incompatible types or layouts, or keep earlier objects in use. For example,
successively larger tensors require new allocations:

.. literalinclude:: ../../../samples/object_cache/unbounded_growth.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

CV-CUDA uses the configured cache limit to evict entries that no longer have pending
GPU work. A matching tensor can also be reused before its previous GPU work
completes, provided no Python wrapper, exported view, reshape view, or retained
native handle can still observe it. Below the limit, operators that pass their
consumer stream to allocation require the tensor's synchronization history to
permit the new write without any cross-stream wait. Under pressure, reuse may
also add CUDA dependencies to order the new access after earlier work. Cache
lookup does not wait for the GPU on the host.

The Python bindings supply the consumer stream for tensor Flip,
PillowResize, ConvertTo, Normalize, and Reformat. Other tensor allocations retain
pressure-only pending reuse. Free-threaded release records may retain
Python wrappers themselves; such tensors conservatively remain ineligible while
the wrappers are registered, even when no user reference remains.

Variable-shape Flip, PillowResize, and Normalize can also reuse an entire image
batch with the same ordered image sizes and formats. The batch and its images
must have no observable aliases, and pending accesses must already be ordered on
the consumer's CUDA queue. Reuse leaves the batch descriptors unchanged, even
while earlier GPU work is pending. A mismatch falls back to ordinary allocation;
generic batch construction never reuses an in-flight descriptor. This early
whole-batch reuse is disabled in free-threaded Python builds. In ordinary Python
builds, pending child image allocations are retained and indexed individually
for cache accounting.

The limit controls retained cache storage, rather than total application GPU memory.
Pending work must retain its resources even when a new shape does not fit the cache
or the application lowers the limit. Those indexed allocations remain included in
the reported cache bytes and can temporarily put the cache above its limit.
Retirement during subsequent submissions or stream synchronization removes
completed pending entries and evicts excess cache storage. There is no background
retirement thread. ``clear_cache()`` explicitly waits for pending work before
clearing the cache.

An unbounded sequence of distinct shapes or outputs kept alive by the application
can still exhaust device memory. The cache limit neither rejects those allocations
nor makes live output buffers eligible for reuse. Wrapped external allocations are
not included in cache byte accounting.

Configuring the cache limit
---------------------------

.. literalinclude:: ../../../samples/object_cache/control.py
   :language: python
   :start-after: # docs-start: limit
   :end-before: # docs-end: limit

By default, the cache limit is set to half the total GPU memory of the current device when importing cvcuda:

.. literalinclude:: ../../../samples/object_cache/control_torch.py
   :language: python
   :start-after: # docs-start: default
   :end-before: # docs-end: default

The getter and setter apply to the current CUDA device. A limit larger than that
device's total memory does not provide additional memory and produces a warning.

Setting the limit to zero disables retention of completed allocations. Eligible
allocations that must remain alive for pending GPU work can still be reused safely,
including allocations larger than the limit. They leave the pending index after
retirement.

Querying cache size
-------------------

.. literalinclude:: ../../../samples/object_cache/control_torch.py
   :language: python
   :start-after: # docs-start: query
   :end-before: # docs-end: query

Multithreading Considerations
------------------------------

Ordinary reuse searches the calling thread's cache. When a thread exits, its
pending entries transfer to a shared index without losing their byte charge.
Eligible tensors in that index can be reused by another thread on an already
ordered consumer stream or under pressure;
completed entries are removed. Entries in another active thread's local cache
are not searched.

.. warning::
   Cache size and limits are shared between threads. Exercise caution in multithreaded applications.

You can clear the cache for the current thread using :py:func:`cvcuda.clear_cache` with :py:class:`cvcuda.ThreadScope`.LOCAL
and query the thread-local cache size with :py:func:`cvcuda.cache_size` with :py:class:`cvcuda.ThreadScope`.LOCAL:

.. literalinclude:: ../../../samples/object_cache/threads.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main
