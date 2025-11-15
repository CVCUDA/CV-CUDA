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

.. _object_cache:

Object Cache
============

CV-CUDA includes an internal resource management system that caches allocated objects for efficient reuse.
Objects used within CV-CUDA, such as :py:class:`cvcuda.Image`, :py:class:`cvcuda.Tensor`, :py:class:`cvcuda.ImageBatchVarShape`, and :py:class:`cvcuda.TensorBatch`, are automatically managed by the CV-CUDA cache.

.. note::
   Only Python objects are cached, there is no C/C++ object caching.

.. note::
   CV-CUDA is device agnostic and does not track which device the data resides on.

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
When the only remaining reference to the object is from the CV-CUDA object cache, the underlying memory can be reused.
As such, it is best practice to not attempt to manually free memory.

.. note::
   You can manually free all cached memory allocations by calling :py:func:`cvcuda.clear_cache`.

Cache Reuse
-----------

When an CV-CUDA object goes out of scope, its memory is retained in the cache for efficient reuse.
Creating a new object with identical specifications (shape, data type, etc.) will reuse the cached memory:

.. literalinclude:: ../../../samples/object_cache/reuse.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

In this example, ``tensor2`` reuses the memory from ``tensor1`` since they have identical shapes and data types. No new memory allocation occurs.

Cache reuse also applies to wrapped objects, improving efficiency even though they don't consume cache memory.

Controlling Cache Growth
------------------------

Certain workflows can cause unbounded cache growth, particularly when creating many non-wrapped objects with different shapes:

.. literalinclude:: ../../../samples/object_cache/unbounded_growth.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main

To manage cache growth, CV-CUDA provides a configurable cache limit with automatic clearing.
When the cache reaches this limit, it is automatically cleared. Objects larger than the cache limit are not cached.

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

You can set the cache limit larger than a single GPU's memory since CV-CUDA is device agnostic.
For example, with two 24GB GPUs, you could set a cache limit exceeding 40GB if you distribute data across both devices.

Setting the cache limit to 0 effectively disables caching, though this may impact performance since memory cannot be reused.

Querying cache size
-------------------

.. literalinclude:: ../../../samples/object_cache/control_torch.py
   :language: python
   :start-after: # docs-start: query
   :end-before: # docs-end: query

Multithreading Considerations
------------------------------

The cache uses thread-local storage internally. Objects created in one thread cannot be reused by another thread when they go out of scope.

.. warning::
   Cache size and limits are shared between threads. Exercise caution in multithreaded applications.

You can clear the cache for the current thread using :py:func:`cvcuda.clear_cache` with :py:class:`cvcuda.ThreadScope`.LOCAL
and query the thread-local cache size with :py:func:`cvcuda.cache_size` with :py:class:`cvcuda.ThreadScope`.LOCAL:

.. literalinclude:: ../../../samples/object_cache/threading.py
   :language: python
   :start-after: # docs-start: main
   :end-before: # docs-end: main
