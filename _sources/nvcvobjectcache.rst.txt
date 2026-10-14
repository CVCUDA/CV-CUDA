..
   # SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _nvcvobjectcache:


NVCV Object Cache
=================

CV-CUDA has internal Resource management.
Python objects that are used within CV-CUDA will be added to CV-CUDA's NVCV cache.

Note: CV-CUDA is device agnostic, ie CV-CUDA does not know on which device the data resides!

Basics
------

The most prominent cached objects are of the following classes: ``Image``, ``ImageBatch``, ``Stream``, ``Tensor``, ``TensorBatch``, ``ExternalCacheItem`` (eg. an operator's payload).

With respect to the cache, we differentiate objects between their used memory of the cache.
While wrapped objects do not increase the cache's size, non-wrapped objects do increase the cache.

An example of a non-wrapped object that increases the cache's memory::

   import cvcuda
   import numpy as np

   tensor = nvcv.Tensor((16, 32, 4), np.float32, nvcv.TensorLayout.HWC)

Wrapped objects are objects which do not have the memory hosted by CV-CUDA, hence they do not increase the cache's memory.
In the following python snippet, the ``cvcuda_tensor`` is a wrapped tensor, which does not increase the cache's memory.::

   import cvcuda
   import torch

   torch_tensor = torch.tensor([1], device="cuda", dtype=torch.uint8)
   cvcuda_tensor = torch.as_tensor(torch_tensor)


Cache Re-use
--------------

If a CV-CUDA object is created and runs out of scope, we can leverage the cache to efficiently create a new CV-CUDA object with the same specifics, eg of the same shape and data type::

   import cvcuda
   import numpy as np

   def create_tensor1():
      tensor1 = nvcv.Tensor((16, 32, 4), np.float32, nvcv.TensorLayout.HWC)
      return

   def create_tensor2():
      # re-use the cache
      tensor2 = nvcv.Tensor((16, 32, 4), np.float32, nvcv.TensorLayout.HWC)
      return

   create_tensor1()
   # tensor1 runs out of scope, after leaving ``create_tensor1()``
   create_tensor2()


In this case, for ``tensor2`` no new memory is being allocated, as we re-use the memory from ``tensor1``, because ``tensor1`` and ``tensor2`` have the same shape and data type.

Cache re-use is also possible for wrapped objects (even if they do not increase the cache's memory, it's more efficient to use the re-use the cache).

Controlling the cache limit
---------------------------

Some workflows can cause the cache to grow significantly, eg if one keeps creating non-wrapped tensors of different shape. Hence, rarely re-using the cache::

   import cvcuda
   import numpy as np
   import random

   def create_tensor(h, w):
      tensor1 = nvcv.Tensor((h, w, 3), np.float32, nvcv.TensorLayout.HWC)
      return

   while True:
      h = random.randint(1000, 2000)
      w = random.randint(1000, 2000)
      create_tensor(h, w)

To control that cache growth, CV-CUDA implements a user-configurable' cache limit and automatic clearance mechanism.
When the cache hits that limit, it is automatically cleared.
Similarly, if a single object is larger than the cache limit, we do not add it to the cache.
The cache limit can be controlled in the following manner::

   import cvcuda

   # Get the cache limit (in bytes)
   current_cache_limit = nvcv.get_cache_limit_inbytes()

   # Set the cache limit (in bytes)
   my_new_cache_limit = 12345 # in bytes
   nvcv.set_cache_limit_inbytes(my_new_cache_limit)

By default the cache limit is set to half the total GPU memory of the current device when importing cvcuda, eg::

   import cvcuda
   import torch

   # Set the cache limit (in bytes)
   total_mem = torch.cuda.mem_get_info()[1]
   nvcv.set_cache_limit_inbytes(total_mem // 2)

It is also feasible to set the cache limit to a value larger than the total GPU memory.
Due to CV-CUDA being device agnostic, it can happen that a larger cache than one GPU's total memory is possible.
Consider a scenario where two GPUs, each with 24GB are available.
Data of 20GB could reside on each GPU.
Setting the cache to >40GB, allows to keep all data in cache, despite the cache limit being larger than one GPU's total memory.
It is, however, the user's responsibility to distribute the data accordingly.

A cache limit of 0 effectively disables the cache.
However, a low cache limit or a disabled cache can cause a hit in performance, as already allocated memory is not being re-used, but new memory has to be allocated and deallocated.

CV-CUDA also provides querying the current cache size (in bytes). This can be helpful for debugging::

   import cvcuda

   print(nvcv.current_cache_size_inbytes())
   img = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
   print(nvcv.current_cache_size_inbytes())

Using the cache with multiple threads
-------------------------------------

Internally, the cache uses thread-local storage. As a result, CV-CUDA objects
created in a thread cannot be reused from another thread when they run out of
scope.

.. warning::
    Since the cache size and limit are shared between threads, care must be
    taken in multithreaded applications.

It is possible to clear the cache of the current thread using
``nvcv.clear_cache(nvcv.ThreadScope.LOCAL)``. Similarly,
``nvcv.cache_size(nvcv.ThreadScope.LOCAL)`` allows querying the number of
elements in the cache for the current thread:

.. code-block:: python

    import threading
    import nvcv
    import numpy as np


    def create_tensor_and_clear():
        tensor = nvcv.Tensor((16, 32, 4), np.float32, nvcv.TensorLayout.HWC)
        print(nvcv.cache_size(), nvcv.cache_size(nvcv.ThreadScope.LOCAL))  # 2 1
        nvcv.clear_cache(nvcv.ThreadScope.LOCAL)
        print(nvcv.cache_size(), nvcv.cache_size(nvcv.ThreadScope.LOCAL))  # 1 0


    tensor = nvcv.Tensor((16, 32, 4), np.float32, nvcv.TensorLayout.HWC)
    threading.Thread(target=create_tensor_and_clear).start()
