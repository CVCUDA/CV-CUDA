..
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _operator_variants:

Allocating vs. Pre-allocated Operator Variants
===============================================

Every CV-CUDA Python operator is available in two forms:

* **Allocating variant** — ``cvcuda.<op>(src, ...)`` allocates and returns a new output tensor.
* **Pre-allocated variant** — ``cvcuda.<op>_into(dst, src, ...)`` writes into a caller-supplied output tensor.

Understanding when to use each form can have a meaningful impact on pipeline throughput.

How the Allocating Variant Works
---------------------------------

When you call the allocating form, CV-CUDA creates the output tensor for you:

.. code-block:: python

    out = cvcuda.resize(src, out_shape, cvcuda.Interp.LINEAR, stream=stream)

Internally this calls ``Tensor.Create``, which consults the :ref:`object_cache` for a
previously freed tensor with matching shape, layout, and dtype.  If one is found it is
reused; otherwise a new GPU allocation is made.  Either way, a cache lookup or
allocation occurs on every call.

How the Pre-allocated Variant Works
-------------------------------------

When you call the ``_into`` form, no cache interaction takes place:

.. code-block:: python

    cvcuda.resize_into(out, src, cvcuda.Interp.LINEAR, stream=stream)

The operator writes directly into ``out``.  The object cache is not consulted, and no
GPU memory is allocated.  The return value is the same ``out`` tensor that was passed in.

When to Use Each Variant
--------------------------

**Use the allocating variant** when:

* You are writing a one-shot script or prototype where throughput is not the priority.
* The output shape changes from call to call (e.g. variable-resolution inputs with
  different target sizes), because caching handles the varying shapes for you.

**Use the pre-allocated (``_into``) variant** when:

* You have a fixed-shape inference pipeline (the common case for batch pre-processing)
  and can allocate output tensors once at startup.
* You are calling an operator in a tight loop and want to eliminate the per-iteration
  cache overhead.
* You manage your own buffer pool and need deterministic memory behaviour.

Example: pre-allocating for a fixed-shape pipeline
---------------------------------------------------

.. code-block:: python

    import cvcuda

    BATCH, H, W, C = 8, 224, 224, 3
    OUT_SHAPE = (BATCH, H, W, C)

    # Allocate output tensor once, before the loop.
    resize_out = cvcuda.Tensor(OUT_SHAPE, cvcuda.Type.U8, "NHWC")

    stream = cvcuda.Stream()
    with stream:
        for frame_batch in data_loader:
            # No allocation on each iteration — writes directly into resize_out.
            cvcuda.resize_into(resize_out, frame_batch, cvcuda.Interp.LINEAR)
            # ... further processing ...

Constraints
-----------

When using the ``_into`` variant the output tensor you supply must already have the
correct shape, layout, and dtype that the operator would have produced.  Passing an
incompatible tensor raises an exception.

.. note::

    ``_into`` variants are available for all standard operators.  The ``ImageBatchVarShape``
    overloads follow the same pattern: ``cvcuda.<op>_into(dst_batch, src_batch, ...)``.

.. seealso::

    :ref:`object_cache` — detailed description of the CV-CUDA object cache, including
    cache reuse, growth control, and multi-threading considerations.
