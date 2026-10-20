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

"""Regressions for reentrant release-owner destruction."""

from test_resource_lifetime import _run_repro


def test_replaced_stream_owner_finalizer_can_reenter_release_ledger():
    """Same-queue wrapper replacement must destroy the old owner unlocked."""
    _run_repro(
        r"""
        import gc

        underlying = cupy.cuda.Stream(non_blocking=True)
        source = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
        destination = cvcuda.Tensor(source.shape, source.dtype, source.layout)
        reentries = []

        # Compile the delay and initialize the operator/release path before the
        # long-running gate. Cold CUDA setup is allowed to synchronize.
        with underlying:
            _DELAY_KERNEL((1,), (1,), (1,))
        underlying.synchronize()
        warmup_wrapper = cvcuda.as_stream(int(underlying.ptr))
        cvcuda.flip_into(destination, source, 0, stream=warmup_wrapper)
        underlying.synchronize()
        cvcuda.internal.drain_pending_releases()
        del warmup_wrapper

        class ReentrantHandle(int):
            def __del__(self):
                cvcuda.internal._pending_release_batch_count()
                reentries.append(True)

        with underlying:
            _DELAY_KERNEL((1,), (1,), (2_000_000_000,))

        def submit_with_fresh_wrapper():
            owner = ReentrantHandle(underlying.ptr)
            wrapper = cvcuda.as_stream(owner)
            cvcuda.flip_into(destination, source, 0, stream=wrapper)
            del wrapper, owner
            gc.collect()

        # Two generations are retained for this queue. The third submission
        # advances the newest generation and displaces its previous wrapper.
        submit_with_fresh_wrapper()
        submit_with_fresh_wrapper()
        assert cvcuda.internal._pending_release_batch_count() == 2
        submit_with_fresh_wrapper()

        assert reentries, "displaced stream owner was not finalized"
        underlying.synchronize()
        cvcuda.internal.drain_pending_releases()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_replaced_resource_owner_is_retained_through_consumer_enqueue():
    """A displaced stream owner's finalizer must observe the submitted write."""
    _run_repro(
        r"""
        import gc

        import cupy
        import numpy

        import cvcuda


        shape = (1, 32, 32, 1)
        underlying = cupy.cuda.Stream(non_blocking=True)
        owning_stream = cvcuda.as_stream(underlying)
        observer_stream = cvcuda.Stream()
        source = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        destination = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        snapshot = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")

        expected = numpy.arange(numpy.prod(shape), dtype=numpy.uint8).reshape(shape)
        cupy.asarray(source.cuda()).set(expected)
        cupy.asarray(destination.cuda()).fill(211)
        cupy.asarray(snapshot.cuda()).fill(0)
        cupy.cuda.runtime.deviceSynchronize()
        source.submitStreamSync(owning_stream)

        armed = False
        reentries = []
        failures = []

        class ReentrantHandle(int):
            def __del__(self):
                if not armed:
                    return
                try:
                    # Flip observes `destination` on another stream. If this
                    # finalizer runs before the outer Flip is enqueued, it
                    # snapshots the old fill instead of the outer result.
                    cvcuda.flip_into(snapshot, destination, 0, stream=observer_stream)
                    observer_stream.sync()
                    reentries.append(True)
                except BaseException as error:
                    failures.append(repr(error))

        raw_owner = ReentrantHandle(underlying.ptr)
        raw_stream = cvcuda.as_stream(raw_owner)
        destination.submitStreamSync(raw_stream)
        del raw_stream, raw_owner
        gc.collect()
        assert reentries == []

        armed = True
        cvcuda.flip_into(destination, source, 0, stream=owning_stream)
        assert reentries == [True], "displaced stream owner was not finalized"
        assert failures == [], failures

        # The outer Flip and the finalizer's Flip cancel. Correct deferral and
        # history tracking therefore reproduce the original source exactly.
        numpy.testing.assert_array_equal(cupy.asarray(snapshot.cuda()).get(), expected)
        owning_stream.sync()
        cvcuda.internal.drain_pending_releases()
        print("PASS", flush=True)
        """,
        timeout=30,
    )
