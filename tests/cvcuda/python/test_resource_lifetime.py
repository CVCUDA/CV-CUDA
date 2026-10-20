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

"""Black-box regressions for asynchronous resource and wrapper lifetimes.

An operator is queued asynchronously, so the buffers it reads and writes must
stay alive until the GPU retires the kernel -- not until Python drops its
reference.  A stream therefore holds every resource an op touched and releases
it only once the work that used it has completed.

Testing that from Python is less obvious than it looks, and two plausible
approaches do *not* work:

* Dropping references and churning allocations proves nothing.  Left to itself
  the GPU retires these kernels before the host can recycle anything, so the
  race window never opens; such a test passes even with the hold deleted.
* Timing ``clear_cache`` or ``sync`` proves nothing either.  Both block anyway
  --- ``cudaFree`` and ``cudaMalloc`` synchronize the device --- so they take
  the same time with and without a hold.

What the hold actually governs is **cvcuda cache reuse**: a resource still held
by a stream counts as in use, so the cache must not hand it to the next op.
That is observable as a device pointer, deterministically and without relying
on a race.  Tests that park the GPU from Python use an SM-side busy-wait rather
than a Python host callback, which would deadlock against pytest's GIL.  Tests
that need an indefinite gate release the GIL while its host callback waits.
"""

import gc
import os
import re
import subprocess
import sys
import sysconfig
import textwrap
import weakref

import cupy
import pytest

import cvcuda
from cvcuda_test_utils import as_cupy_stream as _external_stream


_DELAY_KERNEL_SOURCE = r"""
extern "C" __global__ void cvcuda_test_busy_wait(unsigned long long cycles) {
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {}
}
"""
_DELAY_KERNEL_NAME = "cvcuda_test_busy_wait"
_DELAY_KERNEL = cupy.RawKernel(
    _DELAY_KERNEL_SOURCE,
    _DELAY_KERNEL_NAME,
)


def _park_gpu(stream, cycles=300_000_000):
    with _external_stream(int(stream.handle)):
        _DELAY_KERNEL((1,), (1,), (cycles,))


_SUBPROCESS_CUDA_PRELUDE = textwrap.dedent(
    f"""\
    import contextlib
    import threading
    import cupy
    import cvcuda
    from cvcuda_test_utils import as_cupy_stream as _external_stream

    _DELAY_KERNEL = cupy.RawKernel(
        {_DELAY_KERNEL_SOURCE!r},
        {_DELAY_KERNEL_NAME!r},
    )

    def _park_gpu(stream, cycles):
        with _external_stream(int(stream.handle)):
            _DELAY_KERNEL((1,), (1,), (cycles,))

    def _park_default(cycles):
        with cupy.cuda.Stream.null:
            _DELAY_KERNEL((1,), (1,), (cycles,))

    @contextlib.contextmanager
    def _hold_gpu(stream):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def wait_for_release(_):
            entered.set()
            release.wait()
            finished.set()

        _external_stream(int(stream.handle)).launch_host_func(wait_for_release, None)
        try:
            assert entered.wait(10), "GPU gate callback did not start"
            yield
        finally:
            release.set()
            assert finished.wait(10), "GPU gate callback did not finish"
    """
)


def _device_ptr(tensor):
    return tensor.cuda().__cuda_array_interface__["data"][0]


def _run_repro(source, timeout=60, *, cuda_prelude=False):
    script = textwrap.dedent(source)
    if cuda_prelude:
        script = _SUBPROCESS_CUDA_PRELUDE + script
    script = (
        f"import sys; sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        + script
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"subprocess timed out after {timeout}s\n"
            f"stdout: {(exc.stdout or '')[-1000:]}\n"
            f"stderr: {(exc.stderr or '')[-2000:]}"
        ) from exc

    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"subprocess failed (returncode={result.returncode})\n"
        f"stdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )


def test_pending_buffer_is_not_reused_across_independent_streams():
    """An independent stream uses separate ready storage from a pending writer."""
    original_limit = cvcuda.get_cache_limit_inbytes()
    producer = cvcuda.Stream()
    consumer = cvcuda.Stream()
    try:
        cvcuda.clear_cache()
        shape = (1, 64, 64, 3)
        first_values = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(shape)
        second_values = 255 - first_values
        cupy.cuda.get_current_stream().synchronize()
        first_source = cvcuda.as_tensor(first_values, "NHWC")
        second_source = cvcuda.as_tensor(second_values, "NHWC")

        # Keep each warm output live until all four allocations exist. The
        # consumer can then remain independent without allocating after the
        # producer is parked, which could synchronize the device.
        warm = [cvcuda.flip(first_source, 0, stream=producer) for _ in range(4)]
        producer.sync()
        cvcuda.internal.drain_pending_releases()
        item_bytes = cvcuda.internal.nbytes_in_cache(warm[0])
        del warm
        gc.collect()

        baseline_bytes = cvcuda.current_cache_size_inbytes()
        generous_limit = baseline_bytes + 4 * item_bytes
        assert baseline_bytes + 2 * item_bytes < generous_limit
        cvcuda.set_cache_limit_inbytes(generous_limit)

        _park_gpu(producer)
        first = cvcuda.flip(first_source, 0, stream=producer)
        first_id = first.id
        first_ptr = _device_ptr(first)
        del first
        gc.collect()

        second = cvcuda.flip(second_source, 0, stream=consumer)
        assert second.id != first_id
        assert _device_ptr(second) != first_ptr

        consumer.sync()
        assert cupy.array_equal(
            cupy.asarray(second.cuda()), cupy.flip(second_values, axis=1)
        )
        producer.sync()
        del second
        gc.collect()
    finally:
        try:
            producer.sync()
            consumer.sync()
            cvcuda.internal.drain_pending_releases()
        finally:
            cvcuda.set_cache_limit_inbytes(original_limit)


def test_resource_holds_across_threads_and_streams():
    """Independent streams may submit and retire holds concurrently."""
    _run_repro(
        """
        import threading
        import cvcuda

        errors = []

        def worker():
            try:
                stream = cvcuda.Stream()
                with stream:
                    for _ in range(120):
                        src = cvcuda.Tensor((1, 256, 256, 3), cvcuda.Type.U8, "NHWC")
                        out = cvcuda.flip(src, 0)
                        del src, out
                stream.sync()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        print("PASS", flush=True)
        """
    )


def test_completed_one_shot_streams_are_reclaimed():
    """Finished work must not keep discarded stream wrappers in use."""
    _run_repro(
        """
        import gc
        import cupy
        import cvcuda

        factories = {
            "native": cvcuda.Stream,
            "wrapped": lambda: cvcuda.as_stream(
                cupy.cuda.Stream(non_blocking=True)
            ),
        }
        for label, factory in factories.items():
            cvcuda.clear_cache()
            sizes = []
            ids = []
            handles = []
            for _ in range(200):
                stream = factory()
                ids.append(stream.id)
                handles.append(stream.handle)
                src = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
                out = cvcuda.flip(src, 0, stream=stream)
                cupy.cuda.runtime.deviceSynchronize()
                del out, src, stream
                gc.collect()
                sizes.append(cvcuda.cache_size())

            assert sizes[39] <= 12, (label, sizes[:40])
            assert sizes[-1] <= 12, (label, sizes)
            if label == "native":
                assert len(set(ids)) <= 2, ids
                assert len(set(handles)) <= 2, handles
        print("PASS", flush=True)
        """
    )


def test_exported_cai_keeps_its_stream_handle_alive():
    """A retained CAI view must retain the exact stream it advertises."""
    _run_repro(
        """
        import gc
        import cupy
        import cvcuda

        for label in ("native", "wrapped"):
            cvcuda.clear_cache()
            producer = None
            if label == "native":
                stream = cvcuda.Stream()
            else:
                producer = cupy.cuda.Stream(non_blocking=True)
                stream = cvcuda.as_stream(producer)

            src = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
            out = cvcuda.flip(src, 0, stream=stream)
            exported = out.cuda()
            advertised = exported.__cuda_array_interface__["stream"]
            assert advertised == stream.handle and advertised not in (0, 1, 2)

            replacement = cvcuda.Stream()
            cvcuda.flip_into(out, src, 0, stream=replacement)
            replacement.sync()
            del out, src, stream, producer, replacement
            gc.collect()
            cvcuda.clear_cache()
            gc.collect()

            consumed = cupy.asarray(exported)
            consumed.sum().get()
            del consumed, exported
        print("PASS", flush=True)
        """
    )


@pytest.mark.parametrize("max_version", [None, (1, 0)])
def test_dlpack_capsule_keeps_exported_allocation_alive(max_version):
    """A capsule must retain the allocation after its Python view is gone."""
    cvcuda.clear_cache()
    shape = (1, 64, 64, 1)
    tensor = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
    exported = tensor.cuda()
    pointer = _device_ptr(tensor)
    if max_version is None:
        capsule = exported.__dlpack__()
    else:
        capsule = exported.__dlpack__(max_version=max_version)

    del exported, tensor
    gc.collect()

    replacement = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
    assert (
        _device_ptr(replacement) != pointer
    ), "the cache reused an allocation still exported through a DLPack capsule"

    del capsule, replacement
    cvcuda.clear_cache()


@pytest.mark.parametrize("max_version", [None, (1, 0)])
def test_dlpack_manager_releases_python_owner_with_gil(max_version):
    """A native consumer may release its DLPack manager off the Python thread."""
    dlpack_call = (
        "__dlpack__()" if max_version is None else "__dlpack__(max_version=(1, 0))"
    )
    _run_repro(
        f"""
        import gc
        import weakref

        import cvcuda

        released = []
        tensor = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
        reference = weakref.ref(tensor, lambda _ref: released.append("owner"))
        exported = tensor.cuda()
        capsule = exported.{dlpack_call}

        del exported, tensor
        gc.collect()
        assert reference() is not None

        cvcuda._test.release_dlpack_capsule_off_gil(capsule)
        assert reference() is None
        assert released == ["owner"]
        print("PASS", flush=True)
        """
    )


def test_empty_resourceguard_holds_first_resource_added_by_callable():
    """An initially empty run must hold resources discovered by its callable."""
    cvcuda.internal.drain_pending_releases()
    stream = cvcuda.Stream()
    resource = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
    reference = weakref.ref(resource)

    cvcuda._test.resourceguard_run_adds_first_resource(stream, resource)
    del resource
    gc.collect()
    assert reference() is not None

    cvcuda.internal.drain_pending_releases()
    gc.collect()
    assert reference() is None


def test_direct_submit_sync_keeps_the_old_stream_handle_alive():
    """Public submitStreamSync must work without an accompanying hold."""
    _run_repro(
        """
        cvcuda.clear_cache()
        tensor = cvcuda.Tensor((1, 64, 64, 1), cvcuda.Type.U8, "NHWC")
        values = cupy.asarray(tensor.cuda())
        values.fill(0)
        observed = cupy.empty_like(values)
        cupy.cuda.runtime.deviceSynchronize()
        old_stream = cvcuda.Stream()
        new_stream = cvcuda.Stream()

        tensor.submitStreamSync(old_stream)
        old_external = _external_stream(old_stream.handle)
        with old_external:
            _DELAY_KERNEL((1,), (1,), (2_000_000_000,))
            values.fill(73)
        del old_external, old_stream
        cvcuda.clear_cache()

        tensor.submitStreamSync(new_stream)
        with _external_stream(new_stream.handle):
            cupy.copyto(observed, values)
        new_stream.sync()
        assert cupy.all(observed == 73)
        cupy.cuda.runtime.deviceSynchronize()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_same_handle_alias_preserves_the_existing_owner():
    """A non-owning alias must not replace the owner of the same handle."""
    _run_repro(
        """
        import gc
        import cupy
        import cvcuda

        producer = cupy.cuda.Stream(non_blocking=True)
        handle = producer.ptr
        owning = cvcuda.as_stream(producer)
        alias = cvcuda.as_stream(handle)
        tensor = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
        values = cupy.asarray(tensor.cuda())
        tensor.submitStreamSync(owning)

        del owning, producer
        gc.collect()
        tensor.submitStreamSync(alias)
        from cvcuda_test_utils import as_cupy_stream

        with as_cupy_stream(handle):
            values.fill(29)
        alias.sync()
        assert cupy.all(values == 29)
        print("PASS", flush=True)
        """
    )


def test_same_handle_alias_upgrades_to_a_lifetime_owner():
    """A later owning wrapper must replace a non-owning same-handle alias."""
    _run_repro(
        """
        import gc
        import weakref

        producer = cupy.cuda.Stream(non_blocking=True)
        reference = weakref.ref(producer)
        alias = cvcuda.as_stream(producer.ptr)
        owning = cvcuda.as_stream(producer)
        tensor = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")

        tensor.submitStreamSync(alias)
        tensor.submitStreamSync(owning)
        del owning, producer
        gc.collect()

        assert reference() is not None, "resource retained only the raw same-handle alias"
        assert tensor.cuda().__cuda_array_interface__["stream"] == int(alias.handle)
        alias.sync()
        print("PASS", flush=True)
        """,
        cuda_prelude=True,
    )


def test_default_stream_binding_orders_nonblocking_consumers():
    """A bound null default stream must not be confused with no binding."""
    _run_repro(
        """
        default = cvcuda.Stream.default
        consumer = cvcuda.Stream()
        src = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
        mid = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
        out = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")

        cupy.asarray(src.cuda()).fill(0)
        cupy.cuda.runtime.deviceSynchronize()
        cvcuda.flip_into(mid, src, 0, stream=default)
        default.sync()
        cvcuda.flip_into(out, mid, 0, stream=consumer)
        consumer.sync()

        cupy.asarray(src.cuda()).fill(37)
        cupy.cuda.runtime.deviceSynchronize()
        _park_default(500_000_000)
        cvcuda.flip_into(mid, src, 0, stream=default)
        cvcuda.flip_into(out, mid, 0, stream=consumer)
        consumer.sync()
        assert cupy.all(cupy.asarray(out.cuda()) == 37)

        dl_src = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
        dl_out = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
        cupy.asarray(dl_src.cuda()).fill(0)
        cupy.cuda.runtime.deviceSynchronize()
        cvcuda.flip_into(dl_out, dl_src, 0, stream=default)
        default.sync()
        cupy.asarray(dl_src.cuda()).fill(91)
        cupy.cuda.runtime.deviceSynchronize()

        dl_consumer = cupy.cuda.Stream(non_blocking=True)
        _park_default(500_000_000)
        cvcuda.flip_into(dl_out, dl_src, 0, stream=default)
        exported = dl_out.cuda()
        assert exported.__cuda_array_interface__["stream"] == 1
        with dl_consumer:
            consumed = cupy.from_dlpack(exported).copy()
        dl_consumer.synchronize()
        assert cupy.all(consumed == 91)
        default.sync()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() < 2,
    reason="requires two GPUs to reuse the default wrapper across devices",
)
def test_default_stream_binding_tracks_current_device():
    """The singleton default-stream wrapper names a distinct stream per GPU."""
    _run_repro(
        """
        with cupy.cuda.Device(0):
            token = cvcuda.Tensor((1, 1, 1, 1), cvcuda.Type.U8, "NHWC")
            default = cvcuda.Stream.default
            token.submitStreamSync(default)

        with cupy.cuda.Device(1):
            value = cupy.zeros((1024,), dtype=cupy.uint8)
            cupy.cuda.runtime.deviceSynchronize()
            consumer = cvcuda.Stream()
            _park_default(500_000_000)
            with cupy.cuda.Stream.null:
                value.fill(73)

            token.submitStreamSync(default)
            token.submitStreamSync(consumer)

            with _external_stream(consumer.handle):
                observed = value.copy()
            consumer.sync()
            assert cupy.all(observed == 73)
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() < 2,
    reason="requires two GPUs to reuse the default wrapper across devices",
)
def test_default_stream_batch_children_track_current_device():
    """A batch must propagate the default stream's current device to its children."""
    _run_repro(
        """
        with cupy.cuda.Device(0):
            image = cvcuda.Image((1, 1), cvcuda.Format.RGB8)
            batch = cvcuda.ImageBatchVarShape(1)
            batch.pushback(image)
            default = cvcuda.Stream.default
            batch.submitStreamSync(default)

        with cupy.cuda.Device(1):
            value = cupy.zeros((1024,), dtype=cupy.uint8)
            cupy.cuda.runtime.deviceSynchronize()
            consumer = cvcuda.Stream()
            _park_default(500_000_000)
            with cupy.cuda.Stream.null:
                value.fill(73)

            batch.submitStreamSync(default)
            image.submitStreamSync(consumer)

            with _external_stream(consumer.handle):
                observed = value.copy()
            consumer.sync()
            assert cupy.all(observed == 73)
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() < 2,
    reason="requires two GPUs to rebind an image wrapper across devices",
)
def test_rebound_image_layout_export_tracks_current_device():
    """A layout-converting export must describe a rebound wrapper's live device."""
    _run_repro(
        """
        import gc
        import cupy
        import cvcuda

        cvcuda.clear_cache()
        with cupy.cuda.Device(0):
            first = cupy.empty((2, 3, 3), dtype=cupy.uint8)
            image = cvcuda.as_image(first)
            wrapper_id = image.id
            del image, first
            gc.collect()

        with cupy.cuda.Device(1):
            second = cupy.empty((2, 3, 3), dtype=cupy.uint8)

        with cupy.cuda.Device(0):
            image = cvcuda.as_image(second)
            assert image.id == wrapper_id
            exported = image.cuda("HWC")
            assert exported.__dlpack_device__() == (2, 1)
        print("PASS", flush=True)
        """
    )


def test_global_drain_allows_reentrant_python_cleanup():
    """A Python callback under the drain lock may clear the cache again."""
    _run_repro(
        """
        import cvcuda

        completed = []

        def cleanup():
            cvcuda.clear_cache()
            completed.append(True)

        cvcuda.internal._drain_pending_releases_with_lock_hook(cleanup)
        assert completed == [True]
        print("PASS", flush=True)
        """,
        timeout=15,
    )


def test_concurrent_drains_wait_for_the_same_work():
    """A second global drain must not miss work detached by the first."""
    _run_repro(
        """
        import threading

        stream = cvcuda.Stream()
        src = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
        dst = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
        cvcuda.flip_into(dst, src, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()

        _park_gpu(stream, 2_000_000_000)
        cvcuda.flip_into(dst, src, 0, stream=stream)

        drain_locked = threading.Event()
        allow_drain = threading.Event()
        first_done = threading.Event()

        def after_lock():
            drain_locked.set()
            assert allow_drain.wait(10)

        def drain():
            cvcuda.internal._drain_pending_releases_with_lock_hook(after_lock)
            first_done.set()

        waiter = threading.Thread(target=drain)
        waiter.start()
        assert drain_locked.wait(10)
        allow_drain.set()
        cvcuda.internal.drain_pending_releases()
        assert first_done.wait(10)
        waiter.join()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_resource_release_can_reenter_the_global_drain():
    """Python finalizers may drain again while a completed bag is released."""
    _run_repro(
        """
        import weakref
        import cvcuda

        stream = cvcuda.Stream()
        src = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
        out = cvcuda.flip(src, 0, stream=stream)
        called = []
        reference = weakref.ref(
            out,
            lambda _: (
                cvcuda.internal.drain_pending_releases(),
                called.append(True),
            ),
        )
        del out
        cvcuda.internal.drain_pending_releases()
        assert called and reference() is None
        print("PASS", flush=True)
        """,
        timeout=10,
    )


def test_many_queued_ops_without_sync():
    """A deep pending queue must make progress without forced synchronization."""
    _run_repro(
        """
        stream = cvcuda.Stream()
        with stream:
            src = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
            dst = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
            cvcuda.flip_into(dst, src, 0)
            stream.sync()

            _park_gpu(stream, 300_000_000)

            for _ in range(3000):
                cvcuda.flip_into(dst, src, 0)
            assert 0 < cvcuda.internal._pending_release_batch_count() <= 2
        stream.sync()
        print("PASS", flush=True)
        """,
        cuda_prelude=True,
    )


def test_repeated_external_stream_wrappers_share_one_release_batch():
    """Wrapper churn around one CUDA stream must not deepen the ledger."""
    _run_repro(
        """
        cvcuda.clear_cache()
        underlying = cupy.cuda.Stream(non_blocking=True)
        source = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
        destination = cvcuda.Tensor(source.shape, source.dtype, source.layout)

        wrapper = cvcuda.as_stream(underlying)
        cvcuda.flip_into(destination, source, 0, stream=wrapper)
        underlying.synchronize()
        cvcuda.internal.drain_pending_releases()

        _park_gpu(wrapper, 2_000_000_000)
        for _ in range(1000):
            wrapper = cvcuda.as_stream(underlying)
            cvcuda.flip_into(destination, source, 0, stream=wrapper)

        assert 0 < cvcuda.internal._pending_release_batch_count() <= 2
        underlying.synchronize()
        cvcuda.internal.drain_pending_releases()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_repeated_per_thread_default_wrappers_have_bounded_release_batches():
    """Fresh wrappers for one PTDS queue must coalesce by creator thread."""
    _run_repro(
        """
        cvcuda.clear_cache()
        wrapper = cvcuda.as_stream(2)
        source = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
        destination = cvcuda.Tensor(source.shape, source.dtype, source.layout)

        cvcuda.flip_into(destination, source, 0, stream=wrapper)
        wrapper.sync()
        cvcuda.internal.drain_pending_releases()

        with _hold_gpu(wrapper):
            for _ in range(8):
                wrapper = cvcuda.as_stream(2)
                cvcuda.flip_into(destination, source, 0, stream=wrapper)

            assert cupy.cuda.runtime.streamQuery(2) != 0
            assert 0 < cvcuda.internal._pending_release_batch_count() <= 2
        wrapper.sync()
        cvcuda.internal.drain_pending_releases()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_deep_queue_does_not_wait_for_unrelated_oldest_stream():
    """Submission must not host-wait on an unrelated stream's oldest event."""
    _run_repro(
        """
        slow = cvcuda.Stream()
        fast = cvcuda.Stream()
        slow_src = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
        slow_dst = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
        fast_src = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
        fast_dst = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")

        cvcuda.flip_into(slow_dst, slow_src, 0, stream=slow)
        cvcuda.flip_into(fast_dst, fast_src, 0, stream=fast)
        slow.sync()
        fast.sync()
        cvcuda.internal.drain_pending_releases()

        with _hold_gpu(slow):
            cvcuda.flip_into(slow_dst, slow_src, 0, stream=slow)
            for _ in range(400):
                cvcuda.flip_into(fast_dst, fast_src, 0, stream=fast)

            assert cupy.cuda.runtime.streamQuery(slow.handle) != 0
            assert cvcuda.internal._pending_release_batch_count() <= 3
        slow.sync()
        fast.sync()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_cache_pressure_reuses_pending_tensor_without_waiting():
    """A full cache should stream-order a queue-only match without blocking."""
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        src = cvcuda.Tensor((1, 256, 256, 3), cvcuda.Type.U8, "NHWC")

        warmed = cvcuda.flip(src, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        del warmed
        gc.collect()

        # Leave room for exactly the warmed cache. A new duplicate output
        # would force eviction, while the queued output can be reused now and
        # ordered after its prior write on the stream.
        cvcuda.set_cache_limit_inbytes(cvcuda.current_cache_size_inbytes())
        _park_gpu(stream, 300_000_000)

        first = cvcuda.flip(src, 0, stream=stream)
        assert first.id == warmed_id
        first_id = first.id
        del first
        gc.collect()

        second = cvcuda.flip(src, 0, stream=stream)
        assert second.id == first_id
        assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0
        stream.sync()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


@pytest.mark.parametrize("host_delay", [0, 1.2])
def test_cache_pressure_does_not_displace_pending_reuse_index_for_new_key(host_delay):
    """An incompatible new key stays uncached instead of hiding pending memory."""
    _run_repro(
        """
        import gc
        import time

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        source_a = cvcuda.as_tensor(cupy.zeros((1, 256, 256, 3), dtype=cupy.uint8), "NHWC")
        source_b = cvcuda.as_tensor(cupy.zeros((1, 128, 512, 3), dtype=cupy.uint8), "NHWC")

        # CUDA allocation and first-use import may synchronize the device.
        # Prepare B before parking, then remove its index while retaining its
        # caller-owned allocation so both keys really remain in flight below.
        prepared_b = cvcuda.flip(source_b, 0, stream=stream)
        prepared_b_id = prepared_b.id
        stream.sync()
        cvcuda.clear_cache()

        warmed = cvcuda.flip(source_a, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        del warmed
        gc.collect()
        limit = cvcuda.current_cache_size_inbytes()
        cvcuda.set_cache_limit_inbytes(limit)

        with _hold_gpu(stream):
            time.sleep(_HOST_DELAY_)
            old_key = cvcuda.flip(source_a, 0, stream=stream)
            assert old_key.id == warmed_id
            del old_key
            gc.collect()

            # Publish native ownership without the free-threaded guard retaining
            # an argument wrapper, which would intentionally exclude B from reuse.
            prepared_b.submitStreamSync(stream)
            cvcuda._test.hold_resources_on_device(
                stream, [prepared_b], cupy.cuda.runtime.getDevice()
            )
            del prepared_b
            gc.collect()

            identifiers = []
            for _ in range(32):
                output = cvcuda.flip(source_b, 0, stream=stream)
                identifiers.append(output.id)
                del output
                gc.collect()

            assert set(identifiers) == {prepared_b_id}, "pending index did not stabilize the incompatible key"
            assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0
            # Both unavoidable in-flight allocations stay charged, including the
            # new key that could not enter the retained local cache.
            assert cvcuda.current_cache_size_inbytes() == 2 * limit

            reused = cvcuda.flip(source_a, 0, stream=stream)
            assert reused.id == warmed_id, "new-key allocation displaced the pending reuse index"
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() <= limit
        print("PASS", flush=True)
        """.replace(
            "_HOST_DELAY_", repr(host_delay)
        ),
        timeout=30,
        cuda_prelude=True,
    )


def test_lowered_cache_limit_converges_after_pending_work_retires():
    """A pending reuse index may delay eviction, never prevent it."""
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        source = cvcuda.as_tensor(cupy.zeros((1, 256, 256, 3), dtype=cupy.uint8), "NHWC")
        warmed = cvcuda.flip(source, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        del warmed
        gc.collect()

        _park_gpu(stream, 1_000_000_000)
        pending = cvcuda.flip(source, 0, stream=stream)
        del pending
        gc.collect()
        assert cvcuda.current_cache_size_inbytes() > 0

        cvcuda.set_cache_limit_inbytes(0)
        assert cvcuda.current_cache_size_inbytes() > 0
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() == 0
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_cache_pressure_does_not_reuse_user_owned_tensor():
    """Pressure must not make a still-visible Tensor alias a new output."""
    _run_repro(
        """
        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        src = cvcuda.Tensor((1, 256, 256, 3), cvcuda.Type.U8, "NHWC")

        warmed = cvcuda.flip(src, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        del warmed

        cvcuda.set_cache_limit_inbytes(cvcuda.current_cache_size_inbytes())
        _park_gpu(stream, 300_000_000)

        first = cvcuda.flip(src, 0, stream=stream)
        second = cvcuda.flip(src, 0, stream=stream)
        assert second.id != first.id
        assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0
        stream.sync()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_cache_pressure_pending_reuse_orders_cross_stream_write():
    """Pending reuse must carry the old producer dependency to a new stream."""
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        producer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        first_values = cupy.arange(256 * 256 * 3, dtype=cupy.uint8).reshape(1, 256, 256, 3)
        second_values = 255 - first_values
        first_src = cvcuda.as_tensor(first_values, "NHWC")
        second_src = cvcuda.as_tensor(second_values, "NHWC")
        second_src.submitStreamSync(consumer)
        consumer.sync()

        warmed = cvcuda.flip(first_src, 0, stream=producer)
        producer.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        del warmed
        gc.collect()

        cvcuda.set_cache_limit_inbytes(cvcuda.current_cache_size_inbytes())
        _park_gpu(producer, 2_000_000_000)

        first = cvcuda.flip(first_src, 0, stream=producer)
        assert first.id == warmed_id
        first_id = first.id
        del first

        second = cvcuda.flip(second_src, 0, stream=consumer)
        assert second.id == first_id, (second.id, first_id)
        assert (
            cupy.cuda.runtime.streamQuery(int(consumer.handle)) != 0
        ), "consumer stream unexpectedly completed"

        consumer.sync()
        producer.sync()
        expected = cupy.flip(second_values, axis=1)
        assert cupy.array_equal(cupy.asarray(second.cuda()), expected), "cross-stream write was not ordered"
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Free-threaded argument wrappers still retain consumed intermediates",
)
def test_cache_pressure_pending_reuse_waits_for_cross_stream_reader():
    """A new writer must wait for every pending read of the reused Tensor."""
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        producer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        first_values = cupy.arange(256 * 256 * 3, dtype=cupy.uint8).reshape(1, 256, 256, 3)
        second_values = 255 - first_values
        first_src = cvcuda.as_tensor(first_values, "NHWC")
        second_src = cvcuda.as_tensor(second_values, "NHWC")
        second_src.submitStreamSync(consumer)
        consumer.sync()

        warmed = cvcuda.flip(first_src, 0, stream=producer)
        warmed_converted = cvcuda.convertto(warmed, cvcuda.Type.F32, stream=producer)
        producer.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        del warmed, warmed_converted
        gc.collect()

        cvcuda.set_cache_limit_inbytes(cvcuda.current_cache_size_inbytes())
        _park_gpu(producer, 2_000_000_000)

        first = cvcuda.flip(first_src, 0, stream=producer)
        assert first.id == warmed_id
        first_id = first.id
        converted = cvcuda.convertto(first, cvcuda.Type.F32, stream=producer)
        del first

        second = cvcuda.flip(second_src, 0, stream=consumer)
        assert second.id == first_id, (second.id, first_id)
        assert (
            cupy.cuda.runtime.streamQuery(int(consumer.handle)) != 0
        ), "consumer stream unexpectedly completed"

        consumer.sync()
        producer.sync()
        assert cupy.array_equal(
            cupy.asarray(converted.cuda()), cupy.flip(first_values, axis=1).astype(cupy.float32)
        ), "pending reader observed the reused output"
        assert cupy.array_equal(
            cupy.asarray(second.cuda()), cupy.flip(second_values, axis=1)
        ), "cross-stream write produced the wrong output"
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_cache_pressure_does_not_reuse_tensor_with_live_reshape_alias():
    """A reshape view must keep its shared allocation out of pressure reuse."""
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        producer = cvcuda.Stream()
        consumer = cvcuda.Stream()
        first_values = cupy.arange(256 * 256 * 3, dtype=cupy.uint8).reshape(1, 256, 256, 3)
        second_values = 255 - first_values
        first_src = cvcuda.as_tensor(first_values, "NHWC")
        second_src = cvcuda.as_tensor(second_values, "NHWC")

        warmed = cvcuda.flip(first_src, 0, stream=producer)
        producer.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        del warmed
        gc.collect()

        cvcuda.set_cache_limit_inbytes(cvcuda.current_cache_size_inbytes())
        _park_gpu(producer, 2_000_000_000)

        first = cvcuda.flip(first_src, 0, stream=producer)
        assert first.id == warmed_id
        first_id = first.id
        alias = first.reshape((1, 256 * 256, 3), "NWC")
        del first
        gc.collect()

        second = cvcuda.flip(second_src, 0, stream=consumer)
        assert second.id != first_id, "pressure reuse aliased a live reshape view"

        producer.sync()
        consumer.sync()
        expected = cupy.flip(first_values, axis=1).reshape(1, 256 * 256, 3)
        assert cupy.array_equal(cupy.asarray(alias.cuda()), expected), "reshape view was overwritten"
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_cache_does_not_reuse_completed_tensor_with_live_reshape_alias():
    """Ordinary cache hits must also respect allocation-level aliases."""
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        first_values = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(1, 64, 64, 3)
        second_values = 255 - first_values
        first_src = cvcuda.as_tensor(first_values, "NHWC")
        second_src = cvcuda.as_tensor(second_values, "NHWC")

        first = cvcuda.flip(first_src, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        first_id = first.id
        alias = first.reshape((1, 64 * 64, 3), "NWC")
        del first
        gc.collect()

        second = cvcuda.flip(second_src, 0, stream=stream)
        assert second.id != first_id, "ordinary cache reuse aliased a live reshape view"
        stream.sync()
        expected = cupy.flip(first_values, axis=1).reshape(1, 64 * 64, 3)
        assert cupy.array_equal(cupy.asarray(alias.cuda()), expected), "reshape view was overwritten"

        # Alias safety must not permanently strand the original allocation.
        del alias
        gc.collect()
        third = cvcuda.flip(first_src, 0, stream=stream)
        assert third.id == first_id, "allocation remained non-reusable after its last alias died"
        stream.sync()
        print("PASS", flush=True)
        """,
        cuda_prelude=True,
    )


def test_cache_does_not_reuse_tensor_with_live_nested_reshape_alias():
    """The terminal view owns storage after every intermediate view is gone."""
    _run_repro(
        """
        import gc

        cvcuda.clear_cache()
        stream = cvcuda.Stream()
        first_values = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(1, 64, 64, 3)
        second_values = 255 - first_values
        first_src = cvcuda.as_tensor(first_values, "NHWC")
        second_src = cvcuda.as_tensor(second_values, "NHWC")

        first = cvcuda.flip(first_src, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        first_id = first.id

        outer_view = first.reshape((1, 16, 256, 3), "NHWC")
        inner_view = outer_view.reshape((1, 4096, 3), "NWC")
        terminal_view = inner_view.reshape((4096, 3), "WC")
        del first, outer_view, inner_view
        gc.collect()

        second = cvcuda.flip(second_src, 0, stream=stream)
        assert second.id != first_id, "cache reuse aliased a live nested reshape view"
        stream.sync()
        expected = cupy.flip(first_values, axis=1).reshape(4096, 3)
        assert cupy.array_equal(
            cupy.asarray(terminal_view.cuda()), expected
        ), "terminal reshape view was overwritten"

        del terminal_view
        gc.collect()
        third = cvcuda.flip(first_src, 0, stream=stream)
        assert third.id == first_id, "nested reshape allocation remained non-reusable"
        stream.sync()
        print("PASS", flush=True)
        """,
        timeout=30,
        cuda_prelude=True,
    )


def test_deep_concurrent_release_queue_preserves_pending_owners():
    """A deep multi-stream ledger must retain every resource until completion."""
    _run_repro(
        """
        import threading

        # Compile the kernel once here: three threads racing cupy's first-launch
        # JIT is a flake, and is not what this test is about.
        warmup = cvcuda.Stream()
        _park_gpu(warmup, 1)
        warmup.sync()

        errors = []

        def submit():
            try:
                stream = cvcuda.Stream()
                with stream:
                    src = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
                    dst = cvcuda.Tensor((1, 64, 64, 3), cvcuda.Type.U8, "NHWC")
                    # Allocate before parking: cudaMalloc would synchronize.
                    cvcuda.flip_into(dst, src, 0)
                    stream.sync()
                    # Re-park every round so submissions and event recycling
                    # overlap repeatedly across all three threads.
                    for _ in range(8):
                        _park_gpu(stream, 150_000_000)
                        for _ in range(400):
                            cvcuda.flip_into(dst, src, 0)
                stream.sync()
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                errors.append(exc)

        threads = [threading.Thread(target=submit) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            # Bounded: a stalled release ledger must fail the test, not hang CI.
            thread.join(timeout=120)

        assert not any(t.is_alive() for t in threads), "a submitting thread hung"
        assert not errors, errors

        # The event pool has now been churned heavily. The hold must still be intact.
        stream = cvcuda.Stream()
        other = cvcuda.Stream()
        device = cupy.cuda.runtime.getDevice()
        with stream:
            src = cvcuda.Tensor((1, 256, 256, 3), cvcuda.Type.U8, "NHWC")
            warm = [cvcuda.flip(src, 0) for _ in range(4)]
            item_bytes = cvcuda.internal.nbytes_in_cache(warm[0])
            assert cvcuda.current_cache_size_inbytes() <= (
                cvcuda.get_cache_limit_inbytes() - item_bytes
            ), "the cache must have headroom to reject cross-stream pending reuse"
            del warm
            stream.sync()

            # Pending reuse is allowed on the same stream or under cache pressure.
            cvcuda.set_cache_limit_inbytes(
                cvcuda.current_cache_size_inbytes()
                + 8 * cvcuda.internal.nbytes_in_cache(src)
            )
            _park_gpu(stream, 300_000_000)
            first = cvcuda.flip(src, 0)
            first_ptr = first.cuda().__cuda_array_interface__["data"][0]
            del first

            # Poll the writer's release record before the independent allocation.
            cvcuda._test.hold_resources_on_device(other, [src], device)
            assert cupy.cuda.runtime.streamQuery(stream.handle) != 0
            second = cvcuda.flip(src, 0, stream=other)
            second_ptr = second.cuda().__cuda_array_interface__["data"][0]
        other.sync()
        stream.sync()

        assert first_ptr != second_ptr, (
            "the cache reused a buffer whose release event was recycled early"
        )
        print("PASS", flush=True)
        """,
        timeout=300,
        cuda_prelude=True,
    )


def test_clear_cache_with_work_in_flight():
    """Cache clearing must safely drain outstanding resource holds."""
    _run_repro(
        """
        stream = cvcuda.Stream()
        with stream:
            src = cvcuda.Tensor((1, 512, 512, 3), cvcuda.Type.U8, "NHWC")
            warm = cvcuda.flip(src, 0)
            del warm
            stream.sync()

            _park_gpu(stream, 300_000_000)

            out = cvcuda.flip(src, 0)
            del src, out
            cvcuda.clear_cache()
        stream.sync()
        print("PASS", flush=True)
        """,
        cuda_prelude=True,
    )


def test_wrapper_churn_preserves_metadata_and_releases_resources():
    """Lookups describe live wrappers, which are released after their work retires."""
    _run_repro(
        """
        import gc
        import weakref

        import cvcuda

        shapes = [
            ((1, 4, 4, 3), cvcuda.Type.U8),
            ((1, 7, 5, 3), cvcuda.Type.U8),
            ((2, 4, 4, 3), cvcuda.Type.F32),
        ]
        stream = cvcuda.Stream()
        refs = []
        with stream:
            for i in range(30):
                shape, dtype = shapes[i % len(shapes)]
                src = cvcuda.Tensor(shape, dtype, "NHWC")

                out = cvcuda.flip(src, 0)
                assert tuple(out.shape) == shape
                assert out.dtype == dtype
                refs.extend((weakref.ref(src), weakref.ref(out)))
                del src, out
        stream.sync()
        cvcuda.clear_cache()
        gc.collect()
        assert all(ref() is None for ref in refs), "completed wrappers were retained"
        print("PASS", flush=True)
        """
    )


def test_rebound_image_wrapper_reports_current_buffer():
    """A reused image wrapper must resolve to its newly bound native image."""
    _run_repro(
        """
        import cupy
        import cvcuda

        seen = set()
        reused = False
        stream = cvcuda.Stream()
        with stream:
            for i in range(500):
                fill = i % 251
                gpu = cupy.full((6, 8, 3), fill, dtype=cupy.uint8)
                img = cvcuda.as_image(gpu)
                address = id(img)
                reused = address in seen
                seen.add(address)

                out = cvcuda.flip(cvcuda.as_tensor(img), 0)
                back = cupy.asarray(out.cuda())
                assert int(back.min()) == fill
                assert int(back.max()) == fill
                del gpu, img, out, back
                if reused:
                    break
        stream.sync()
        assert reused, "no image wrapper address was reused"
        print("PASS", flush=True)
        """
    )


_LEGACY_CAPI_SHAPE = (1, 256, 256, 3)


def _legacy_capi_pairs(*resources):
    return [("w", resource) for resource in resources]


@pytest.mark.parametrize("indices", [(0, 1, 2), (0, 1)])
def test_completed_release_reuse_does_not_keep_extra_owners(indices):
    cvcuda.clear_cache()
    old_limit = cvcuda.get_cache_limit_inbytes()
    stream = cvcuda.Stream()
    resources = [
        cvcuda.Tensor((1, 16, 16, 1), cvcuda.Type.U8, "NHWC") for _ in range(3)
    ]
    sizes = [cvcuda.internal.nbytes_in_cache(resource) for resource in resources]
    try:
        cvcuda.set_cache_limit_inbytes(0)
        cvcuda._test.capi_legacy_hold_resources(stream, _legacy_capi_pairs(*resources))
        # The record is completed, but remains available to the next submission.
        cupy.cuda.runtime.streamSynchronize(int(stream.handle))
        assert cvcuda.current_cache_size_inbytes() == sum(sizes)

        selected = [resources[index] for index in indices]
        cvcuda._test.capi_legacy_hold_resources(stream, _legacy_capi_pairs(*selected))
        assert cvcuda.internal._pending_release_batch_count() == 1
        assert cvcuda.current_cache_size_inbytes() == sum(
            sizes[index] for index in indices
        )
        stream.sync()
        assert cvcuda.current_cache_size_inbytes() == 0
    finally:
        cvcuda.internal.drain_pending_releases()
        cvcuda.set_cache_limit_inbytes(old_limit)


@pytest.mark.parametrize("reuse_completed", [False, True])
def test_new_raw_alias_submission_inherits_completed_stream_owner(reuse_completed):
    _run_repro(
        f"""
        import gc
        import weakref

        producer = cupy.cuda.Stream(non_blocking=True)
        reference = weakref.ref(producer)
        owning = cvcuda.as_stream(producer)
        alias = cvcuda.as_stream(producer.ptr)
        resources = [cvcuda.Tensor((1, 16, 16, 1), cvcuda.Type.U8, "NHWC") for _ in range(3)]
        device = cupy.cuda.runtime.getDevice()
        cvcuda.internal.drain_pending_releases()

        _park_gpu(alias, 500_000_000)
        cvcuda._test.hold_resources_on_device(alias, [resources[0]], device)
        cvcuda._test.hold_resources_on_device(owning, [resources[1]], device)
        assert cvcuda.internal._pending_release_batch_count() == 2
        producer.synchronize()
        del owning, producer
        gc.collect()
        assert reference() is not None

        _park_gpu(alias, 1_000_000_000)
        selected = resources[0 if {reuse_completed!r} else 2]
        cvcuda._test.hold_resources_on_device(alias, [selected], device)
        gc.collect()
        assert reference() is not None, "completed records dropped the new submission's stream owner"
        assert cupy.cuda.runtime.streamQuery(int(alias.handle)) != 0

        cvcuda.internal.drain_pending_releases()
        gc.collect()
        assert reference() is None
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


def test_post_publication_prune_failure_preserves_submission():
    _run_repro(
        """
        import numpy as np

        stream = cvcuda.Stream()
        source = cvcuda.Tensor((1, 16, 16, 1), cvcuda.Type.U8, "NHWC")
        outputs = [cvcuda.Tensor(source.shape, source.dtype, source.layout) for _ in range(2)]
        cupy.asarray(source.cuda()).fill(47)
        cupy.cuda.get_current_stream().synchronize()
        for output in outputs:
            cvcuda.flip_into(output, source, 0, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        cvcuda.set_cache_limit_inbytes(0)

        cvcuda.flip_into(outputs[0], source, 0, stream=stream)
        cupy.cuda.runtime.streamSynchronize(int(stream.handle))
        cvcuda._test.fail_next_cache_prune(True)
        try:
            # Different owners force the completed record to retire after
            # this new submission has committed its ownership transfer.
            cvcuda.flip_into(outputs[1], source, 0, stream=stream)
            assert not cvcuda._test.fail_next_cache_prune(False), "prune failure was not exercised"
        finally:
            cvcuda._test.fail_next_cache_prune(False)

        stream.sync()
        np.testing.assert_array_equal(cupy.asnumpy(cupy.asarray(outputs[1].cuda())), 47)
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() == 0
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


def test_unattached_cache_admission_preserves_python_owned_entries():
    _run_repro(
        """
        import gc
        import weakref

        cvcuda.clear_cache()
        producer = cupy.cuda.Stream(non_blocking=True)
        reference = weakref.ref(producer)
        stream = cvcuda.as_stream(producer)
        victim = cvcuda.Tensor((1, 16, 16, 1), cvcuda.Type.U8, "NHWC")
        victim_bytes = cvcuda.internal.nbytes_in_cache(victim)
        victim.submitStreamSync(stream)
        stream.sync()
        cvcuda.set_cache_limit_inbytes(victim_bytes)
        del victim, stream, producer
        gc.collect()
        assert reference() is not None

        # It cannot enter the small cache yet. Raising the limit makes it
        # admissible only if the worker evicts the Python-owning victim.
        incoming = cvcuda.Tensor((1, 64, 64, 1), cvcuda.Type.U8, "NHWC")
        incoming_bytes = cvcuda.internal.nbytes_in_cache(incoming)
        assert incoming_bytes > victim_bytes
        cvcuda.set_cache_limit_inbytes(incoming_bytes)
        assert cvcuda.current_cache_size_inbytes() == victim_bytes
        cvcuda._test.add_to_unanchored_worker_cache(incoming)
        assert reference() is not None, "unattached admission evicted a Python owner"
        assert cvcuda.current_cache_size_inbytes() == victim_bytes

        cvcuda.clear_cache()
        gc.collect()
        assert reference() is None
        print("PASS", flush=True)
        """,
        timeout=20,
        cuda_prelude=True,
    )


@pytest.mark.parametrize(
    "entry_point,expected_same_pointer",
    [
        ("capi_legacy_hold_resources", False),
        ("capi_legacy_sync_and_hold", False),
        ("capi_legacy_submit_sync_only", True),
    ],
)
def test_legacy_capi_resource_holds(entry_point, expected_same_pointer):
    """The retained legacy ABI must preserve its sync and hold semantics."""
    stream = cvcuda.Stream()
    warm = cvcuda.Tensor(_LEGACY_CAPI_SHAPE, cvcuda.Type.U8, "NHWC")
    del warm
    stream.sync()
    _park_gpu(stream)

    first = cvcuda.Tensor(_LEGACY_CAPI_SHAPE, cvcuda.Type.U8, "NHWC")
    first_pointer = _device_ptr(first)
    getattr(cvcuda._test, entry_point)(stream, _legacy_capi_pairs(first))
    del first

    second = cvcuda.Tensor(_LEGACY_CAPI_SHAPE, cvcuda.Type.U8, "NHWC")
    second_pointer = _device_ptr(second)
    del second
    stream.sync()
    assert (first_pointer == second_pointer) is expected_same_pointer


@pytest.mark.parametrize(
    "entry_point,label,checks_mode",
    [
        ("capi_legacy_hold_resources", "Hold resources failed", True),
        ("capi_legacy_submit_sync_only", "Submit sync only failed", False),
        ("capi_legacy_sync_and_hold", "Sync and hold resources failed", True),
    ],
)
def test_legacy_capi_rejects_malformed_lists(entry_point, label, checks_mode):
    """Legacy tuple shape and lock-mode validation remain ABI contracts."""
    stream = cvcuda.Stream()
    resource = cvcuda.Tensor(_LEGACY_CAPI_SHAPE, cvcuda.Type.U8, "NHWC")
    call = getattr(cvcuda._test, entry_point)
    invalid = [(object(), None), ([(resource,)], "two elements")]
    if checks_mode:
        invalid.append(([("not-a-lock-mode", resource)], "Lock mode not understood"))

    for payload, pattern in invalid:
        with pytest.raises(ValueError) as error:
            call(stream, payload)
        assert str(error.value).startswith(label)
        if pattern is not None:
            assert re.search(pattern, str(error.value))

    if not checks_mode:
        call(stream, [("not-a-lock-mode", resource)])
    stream.sync()
