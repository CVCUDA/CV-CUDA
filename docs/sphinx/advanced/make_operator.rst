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

.. _make_operator:

Adding a New Operator
=====================

This page walks through adding a new operator to CV-CUDA end-to-end: generating
the scaffold, implementing the kernel, writing tests, adding benchmarks, and
updating docs.

The ``/make-op`` skill
----------------------

This page is the narrative how-to.  For an assisted, gated workflow, use the **make-op skill**
family (defined once under ``.agents/skills/make-op*/SKILL.md``; ``.claude/skills`` symlinks to it
for Claude Code), whose checklist and definition-of-done live in
``.agents/guidance/MAKE_OP_GUIDELINES.md`` and are enforced by the deterministic checker
``tools/make_op.py``.  Two modes:

* **Full end-to-end** — ``/make-op <Name>``: propose the operator's spec (semantics + a cited
  reference oracle such as a TorchVision/OpenCV function, plus the support matrix) and **get user
  approval**; scaffold; implement; then pass the **done-gate**
  (``tools/make_op.py <Name> --phase done --run``), a deterministic regression checklist that
  reuses ``/review-op`` (all domains) and ``/optimize-op`` preflight and additionally requires an
  independent CPU gold reference, **bit-exact** coverage across the declared support matrix,
  required equivalent-layout parity, complement negatives, the operator in the release notes, and
  tests that actually run and pass.  It then hands off to ``/optimize-op`` for performance.
* **Scaffold-only** — ``/make-op-scaffold <Name> [--bare]``: produce a wired, building skeleton
  and stop, delegating the implementation to a human or another AI.

The steps below are what those skills automate; follow them directly for a manual workflow.

.. _scaffold:

Step 1: Generate the Scaffold
------------------------------

The ``mkop.sh`` script generates no-op stubs for every file a new operator
needs and wires them into the build system.  Run it from the repository root:

.. code-block:: shell

    tools/mkop/mkop.sh <OperatorName>

The first letter of *OperatorName* is capitalized automatically, so ``clahe``
and ``Clahe`` both produce ``Clahe`` as the canonical name.

**Generated files**

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - File
     - Purpose
   * - ``src/cvcuda/include/cvcuda/Op<Name>.h``
     - Public C API header
   * - ``src/cvcuda/include/cvcuda/Op<Name>.hpp``
     - Public C++ header
   * - ``src/cvcuda/Op<Name>.cpp``
     - C API implementation (dispatches to private impl)
   * - ``src/cvcuda/priv/Op<Name>.cpp``
     - Private implementation stub
   * - ``src/cvcuda/priv/Op<Name>.hpp``
     - Private implementation header
   * - ``tests/cvcuda/system/TestOp<Name>.cpp``
     - C++ system test stub
   * - ``python/mod_cvcuda/operators/Op<Name>.cpp``
     - Python binding stub (under ``operators/``)
   * - ``tests/cvcuda/python/test_op<name>.py``
     - Python test stub
   * - ``bench/cpp/ops/Bench<Name>.cpp``
     - C++ benchmark stub
   * - ``bench/python/ops/bench_<name>.py``
     - Python benchmark stub
   * - ``bench/config/operators/<name>.json``
     - Benchmark config stub (tiers, layout axis, declared dtypes)

The script also updates these existing files:

* ``src/cvcuda/priv/CMakeLists.txt``
* ``src/cvcuda/CMakeLists.txt``
* ``tests/cvcuda/system/CMakeLists.txt``
* ``python/mod_cvcuda/CMakeLists.txt``
* ``python/mod_cvcuda/Main.cpp``
* ``python/mod_cvcuda/operators/Operators.hpp``
* ``bench/cpp/CMakeLists.txt`` and ``bench/python/CMakeLists.txt``
* ``bench/config/bench_params.json`` (the benchmark manifest)
* ``docs/sphinx/operator_list.rst`` (the operator table row)
* ``docs/sphinx/modules/python/operators.rst`` (the ``autofunction`` directives)
* the latest release notes (``docs/sphinx/relnotes/vX.Y.Z-*.rst``) — a "New Features" bullet

.. note::

   ``mkop.sh`` writes the Python binding directly under ``python/mod_cvcuda/operators/`` and
   wires the benchmark, documentation, and release-note stubs listed above.  The generated stubs
   are intentionally minimal (interleaved ``NHWC`` Tensor only, with ``TODO(make-op)`` markers);
   extend them while implementing.  The private implementation is generated as ``.cpp`` — rename
   it to ``.cu`` (and update ``src/cvcuda/priv/CMakeLists.txt``) only if you write CUDA device
   code directly in it; operators that call into legacy kernels keep ``.cpp``.

.. _implement:

Step 2: Implement the Operator
-------------------------------

1. **Rename the private implementation to** ``.cu`` for CUDA kernels:

   .. code-block:: shell

       mv src/cvcuda/priv/Op<Name>.cpp src/cvcuda/priv/Op<Name>.cu

   Update the filename in ``src/cvcuda/priv/CMakeLists.txt`` to match.

2. **Write the CUDA kernel** in ``src/cvcuda/priv/Op<Name>.cu``.

3. **Make the operator multi-GPU safe** — see :ref:`multi_gpu` below.

4. **Document supported layouts and data types** in the Doxygen comment for
   each function in ``src/cvcuda/include/cvcuda/Op<Name>.h``, using the
   standard "Limitations" table format:

   .. code-block:: c

       /*
        *  Limitations:
        *
        *  Input:
        *       Data Layout:    [kNHWC, kHWC]
        *       Channels:       [1, 3, 4]
        *
        *       Data Type      | Allowed
        *       -------------- | -------------
        *       8bit  Unsigned | Yes
        *       8bit  Signed   | No
        *       16bit Unsigned | Yes
        *       32bit Float    | Yes
        *       ...
        */

   Image operators declare both interleaved (``NHWC``/``HWC``) and planar
   (``NCHW``/``CHW``) layouts by default. If image layouts do not apply, add this
   operator-local declaration beside the Limitations table instead of maintaining
   a central exception list:

   .. code-block:: text

      Planar image layouts: Not applicable
          Reason: <why this operator's tensors do not represent images>

   The implementation must enforce these constraints at runtime and return
   ``NVCV_ERROR_INVALID_ARGUMENT`` for unsupported combinations.

5. **Add negative tests** to ``tests/cvcuda/system/TestOp<Name>.cpp`` that
   verify unsupported formats and dtypes are rejected.  Use a parameterized
   ``NVCV_TEST_SUITE_P`` value list for operators with many rejected
   combinations (see ``TestOpResize.cpp``), or individual ``TEST(..._Negative,
   ...)`` blocks for simpler cases (see ``TestOpCLAHE.cpp``).  Each case
   should assert ``NVCV_ERROR_INVALID_ARGUMENT``.

6. **Expose the parameters** in the public C and C++ headers
   (``Op<Name>.h`` and ``Op<Name>.hpp``).

7. **Build and run the tests** to confirm the implementation is correct:

   .. code-block:: shell

       cmake --build build-rel --target cvcuda_test_system
       ctest --test-dir build-rel -R TestOp<Name>

.. _multi_gpu:

Multi-GPU Safety
----------------

Any operator that allocates device memory must do so with multi-GPU in mind.
The naive approach — calling ``cudaMalloc`` in the constructor and storing the
pointer as a member — allocates on whatever GPU happens to be current at
construction time.  If the operator is later invoked on a different GPU the
kernel will access memory that lives on the wrong device, causing silent
corruption or a CUDA error.

**The pattern: ``PerDeviceResource<T>``**

``src/cvcuda/priv/PerDeviceResource.hpp`` provides a template that lazily
creates one instance of ``T`` per CUDA device.  The factory runs the first
time ``get()`` is called from a new device, with that device already set as
current.  Destruction also sets the correct device before calling the
destructor, so ``cudaFree`` always targets the right GPU.

Put device allocations in a small helper struct and wrap it:

.. code-block:: cpp

    // Op<Name>.hpp
    #include "PerDeviceResource.hpp"

    struct <Name>DeviceBuffers
    {
        void *buf = nullptr;

        <Name>DeviceBuffers(/* constructor params */)
        {
            NVCV_CHECK_THROW(cudaMalloc(&buf, size));
        }

        ~<Name>DeviceBuffers()
        {
            if (buf) NVCV_CHECK_LOG(cudaFree(buf));
        }
    };

    class <Name> final : public IOperator
    {
        // ...
        mutable PerDeviceResource<<Name>DeviceBuffers> m_deviceBuffers;
    };

Initialise it in the constructor with a factory lambda that captures the
parameters needed to size the allocation:

.. code-block:: cpp

    // Op<Name>.cu  (constructor)
    <Name>::<Name>(/* params */)
        : /* other members */
        , m_deviceBuffers([/* capture params */](int /*deviceId*/)
                          { return std::make_unique<<Name>DeviceBuffers>(/* params */); })
    {
    }

And call ``m_deviceBuffers.get()`` in ``operator()``:

.. code-block:: cpp

    void <Name>::operator()(cudaStream_t stream, ...) const
    {
        RunKernel(..., m_deviceBuffers.get().buf, stream);
    }

See ``src/cvcuda/priv/OpCLAHE.hpp`` and ``OpCLAHE.cu`` for a complete
reference implementation.

.. _testing:

Step 3: Write Tests
--------------------

The scaffold generates stubs for both C++ and Python tests.  Fill them in as
described below.  See ``TestOpCLAHE.cpp`` and ``test_opclahe.py`` as reference
implementations.

**C++ correctness tests** (``tests/cvcuda/system/TestOp<Name>.cpp``)

The standard approach is:

1. Write a plain-C++ reference implementation of the operator (or,
   alternatively, test against known reference outputs).
2. Use ``NVCV_TEST_SUITE_P`` to define a parameterized table of inputs
   (sizes, batch sizes, dtypes, operator parameters):

   .. code-block:: cpp

       NVCV_TEST_SUITE_P(Op<Name>, test::ValueList<int, int, int /*, ...*/>
       {
           // width, height, batches, ...
           {  64,  64, 1 },
           { 320, 240, 4 },
       });

3. In the test body, fill input tensors with random data, run the operator,
   copy results back to host, and compare against the reference with a tight
   pixel tolerance:

   .. code-block:: cpp

       TEST_P(Op<Name>, tensor_correct_output)
       {
           // ... create tensors, fill with random data ...
           cvcuda::<Name> op(/* params */);
           EXPECT_NO_THROW(op(stream, in, out, /* params */));
           ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
           // compare out vs. reference result
       }

4. Add a separate ``TEST(Op<Name>, varshape_correct_output)`` that builds an
   ``nvcv::ImageBatchVarShape`` with randomly sized images and runs the same
   comparison.

5. Add ``TEST(Op<Name>_Negative, ...)`` cases (covered in step 2 above).

**Python smoke tests** (``tests/cvcuda/python/test_op<name>.py``)

Python tests focus on the API surface rather than numerical correctness
(the C++ tests own that).  Cover:

* ``Tensor`` inputs with ``NHWC`` and ``HWC`` layouts using
  ``@pytest.mark.parametrize`` over a small set of shapes.
* ``ImageBatchVarShape`` inputs.
* Both the allocating variant (``cvcuda.<name>(src, ...)``) and the
  in-place variant (``cvcuda.<name>_into(dst, src, ...)``), asserting that
  the output has the expected ``shape``, ``layout``, and ``dtype``.
* Negative cases with ``pytest.raises(Exception)`` for unsupported formats
  and invalid parameter values.

.. _benchmarks:

Step 4: Add Benchmarks
-----------------------

Every operator requires matching C++ and Python benchmarks.  See
``bench/README.md`` in the repository root for full benchmark documentation.

**C++ benchmark** — create ``bench/cpp/ops/Bench<Name>.cpp`` and register it
in ``bench/cpp/CMakeLists.txt``.

**Python benchmark** — create ``bench/python/ops/bench_<name>.py`` and
register it in ``bench/python/CMakeLists.txt``.

**Shared configuration** — add an entry to ``bench/config/bench_params.json``.
This file is the single source of truth for parameter axes in both languages:

.. code-block:: json

    "<name>": {
        "dtypes": ["uint8"],
        "string_axes": {
            "shape": ["16x1080x1920"]
        },
        "int64_axes": {
            "varShape": [-1, 0]
        }
    }

.. note::

   Choose batch sizes (the leading dimension of ``shape``) so that each
   benchmark configuration runs for roughly **1–2 ms** on an H100 (or
   equivalent GPU).  Kernels that finish in tens of microseconds have high
   relative timing noise; kernels that run for tens of milliseconds make the
   full benchmark suite slow.  The 1–2 ms range gives nvbench enough signal
   to produce stable, low-noise measurements.

.. _documentation:

Step 5: Fill in the Documentation Stubs
---------------------------------------

``mkop.sh`` has **already inserted** the documentation entries listed above —
the ``operator_list.rst`` table row, the ``operators.rst`` ``autofunction``
directives (``cvcuda.<name>`` and ``cvcuda.<name>_into``), and the release-note
bullet — each with a ``TODO(make-op)`` placeholder. You do **not** add these
rows by hand; you only replace the placeholder text with real content:

1. **Operator list** — in ``docs/sphinx/operator_list.rst``, replace the
   ``TODO(make-op)`` description on the generated row:

   .. code-block:: rst

       * - <Human-readable Name> (:py:func:`cvcuda.<name>`)
         - Brief description of what the operator does.

2. **Python API reference** — the ``autofunction`` directives in
   ``docs/sphinx/modules/python/operators.rst`` are already in place; confirm
   they render once the pybind docstrings (written with the binding in Step 2) are in place.

3. **Release notes** — replace the ``TODO(make-op)`` text in the generated
   "New Features and Enhancements" bullet of the latest
   ``docs/sphinx/relnotes/vX.Y.Z-*.rst``.
