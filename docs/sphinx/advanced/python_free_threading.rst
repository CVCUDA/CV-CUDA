.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _python_free_threading:

Free-threaded Python
====================

CV-CUDA supports free-threaded CPython 3.14 (``3.14t``) without enabling the
global interpreter lock (GIL). Subinterpreters are not supported.

Build and check
---------------

Follow the :ref:`installation` guide for native build prerequisites. Install
Python dependencies with ``python3.14t`` and select it for the build::

    python3.14t -m pip install -r docker/requirements.build.all_pythons.txt
    python3.14t -m pip install -r tests/requirements.tests.common.txt \
        -r tests/requirements.tests.numpy2.txt -r tests/requirements.tests.cu12.txt
    bash build.sh release build-ft -DBUILD_TESTS=1 -DPYTHON_VERSIONS=3.14t

Use ``requirements.tests.cu13.txt`` for CUDA 13. Test builds also require CUPTI
development headers and libraries. To build both Python variants, set
``-DPYTHON_VERSIONS='3.14;3.14t'``.

After installing the built wheel, verify that the GIL stays disabled::

    python3.14t -c 'import sys, cvcuda; assert not sys._is_gil_enabled()'

Other extensions can enable the GIL, so repeat this check after importing your
application's dependencies.

Concurrent use
--------------

Threads can submit operators concurrently. Reads of the same allocation can
overlap; writes are ordered after earlier reads and writes, including aliases
from reshaping, wrapping CUDA buffers, and exporting image planes. This orders
CUDA work without waiting for GPU completion on the submitting Python thread.

Batch changes are serialized with submissions; submitted work retains the batch
contents it used. Structurally changing a batch during iteration raises
``RuntimeError``.

The current-stream context is local to each Python thread. Explicit streams can
be shared across threads. A wrapper for CUDA's per-thread default stream must
be used on the thread that created it.

CUDA Array Interface and DLPack exports carry producer dependencies and retain
their owners. Read-only imports reject operator writes. For external code that
writes through an exported view, establish the producer stream when wrapping
the result again, or synchronize before returning it to CV-CUDA. CV-CUDA cannot
observe arbitrary kernels launched by another library after an export.

``Image.cpu()`` waits for its producers and copies to finish before returning,
while allowing other threads to submit work.

Extension authors
-----------------

Build external bindings for the matching Python ABI. Enqueue CUDA work inside
``nvcvpy::ResourceGuard::run()``, registering resource read/write modes before
enqueuing any work. Guards are single-use; synchronization on destruction alone
does not protect concurrent submissions.

Measuring binding overhead
--------------------------

See the `binding benchmark guide
<https://github.com/CVCUDA/CV-CUDA/blob/main/bench/README.md#measure-python-binding-overhead>`_.
Compare builds using the same Python version and GIL mode; measure ordinary and
free-threaded Python separately.
