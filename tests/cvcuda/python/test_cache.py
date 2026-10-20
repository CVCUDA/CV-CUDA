# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

import cuda.bindings.runtime as cudart

import cvcuda
import cvcuda_util as util
import cupy

RNG = np.random.default_rng(12345)


@pytest.mark.parametrize("kind", ["tensor", "image"])
@pytest.mark.parametrize("scan", [False, True])
def test_cache_does_not_filter_live_descriptors(kind, scan):
    cvcuda.clear_cache()
    if kind == "tensor":
        item = cvcuda.Tensor((8, 16), np.uint8, "HW")
    else:
        item = cvcuda.Image((16, 8), cvcuda.Format.U8)
    cvcuda._test.cache_filter_on_live_item(item, scan)


@pytest.mark.parametrize("kind", ["tensor", "image"])
def test_cache_reuses_capacity_across_shapes(kind):
    cvcuda.clear_cache()
    pointer = capacity = None
    for width, height in [(64, 32), (128, 8), (64, 24), (64, 32)]:
        if kind == "tensor":
            shape = (height, width, 3)
            item = cvcuda.Tensor(shape, np.uint8, "HWC", rowalign=1)
        else:
            shape = (3, height, width)
            item = cvcuda.Image((width, height), cvcuda.Format.RGB8p, rowalign=1)

        exported = item.cuda()
        data = cupy.asarray(exported)
        gold = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        assert exported.shape == shape
        assert exported.strides == gold.strides
        if pointer is None:
            pointer = data.data.ptr
            capacity = cvcuda.internal.nbytes_in_cache(item)
        assert data.data.ptr == pointer
        assert cvcuda.internal.nbytes_in_cache(item) == capacity
        assert cvcuda.current_cache_size_inbytes() == capacity

        data.set(gold)
        cupy.cuda.get_current_stream().synchronize()
        np.testing.assert_array_equal(
            item.cpu() if kind == "image" else data.get(), gold
        )
        del data, exported, item


@pytest.mark.parametrize("kind", ["tensor", "image"])
def test_cache_rebuilds_strides_for_row_alignment(kind):
    cvcuda.clear_cache()
    pointer = None
    for alignment, pitch in [(64, 64), (1, 37), (64, 64)]:
        if kind == "tensor":
            item = cvcuda.Tensor((11, 37), np.uint8, "HW", rowalign=alignment)
        else:
            item = cvcuda.Image((37, 11), cvcuda.Format.U8, rowalign=alignment)
        assert item.cuda().strides == (pitch, 1)
        current_pointer = item.cuda().__cuda_array_interface__["data"][0]
        if pointer is None:
            pointer = current_pointer
        assert current_pointer == pointer
        del item


def test_cache_selects_smallest_fitting_allocation():
    cvcuda.clear_cache()
    small = cvcuda.Tensor((4096,), np.uint8, "W")
    large = cvcuda.Tensor((8192,), np.uint8, "W")
    small_pointer = small.cuda().__cuda_array_interface__["data"][0]
    large_pointer = large.cuda().__cuda_array_interface__["data"][0]
    del small, large

    fitting = cvcuda.Tensor((3072,), np.uint8, "W")
    assert fitting.cuda().__cuda_array_interface__["data"][0] == small_pointer
    del fitting

    oversized = cvcuda.Tensor((12288,), np.uint8, "W")
    assert oversized.cuda().__cuda_array_interface__["data"][0] not in (
        small_pointer,
        large_pointer,
    )
    exact = cvcuda.Tensor((8192,), np.uint8, "W")
    assert exact.cuda().__cuda_array_interface__["data"][0] == large_pointer


def test_worker_cache_is_anchored_to_its_python_thread_state():
    before = cvcuda._test.cache_anchors_installed()

    def worker():
        cvcuda.Image.zeros((256, 256), cvcuda.Format.RGB8)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    # The anchor is what lets the worker release its own items, while it still
    # holds the GIL and a thread state. If it silently stopped installing, the
    # destructor would leak instead and every other test would still pass, so
    # this count is the only thing that would notice.
    assert cvcuda._test.cache_anchors_installed() == before + 1


def test_worker_cache_destruction_does_not_acquire_gil():
    program = textwrap.dedent(
        """
        import atexit
        import threading

        import cvcuda

        cvcuda._test.arm_cache_tls_destructor()

        def create_thread_local_cache():
            cvcuda.cache_size(cvcuda.ThreadScope.LOCAL)

        thread = threading.Thread(target=create_thread_local_cache, daemon=True)
        thread.start()
        cvcuda._test.wait_cache_tls_destructor()

        # Keep the main thread's GIL while the worker finishes its C++ TLS
        # destructor. Acquiring the GIL from that destructor deadlocks here.
        atexit.register(cvcuda._test.release_and_wait_cache_tls_destructor)
        """
    )

    # The hook suspends teardown, so it stays inert unless a process asks for
    # it. Backstop only: the hook's own waits time out first and say why.
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "CVCUDA_CACHE_TLS_TEST_HOOK": "1"},
        )
    except subprocess.TimeoutExpired:
        pytest.fail("worker Cache destructor deadlocked acquiring the GIL")

    assert result.returncode == 0, result.stderr
    # An atexit callback that raises is only printed, so returncode alone would
    # miss a release_and_wait that timed out instead of completing.
    assert "test hook" not in result.stderr, result.stderr


def test_cache_destruction_hook_is_immortal():
    program = textwrap.dedent(
        """
        import threading

        import cvcuda

        cvcuda._test.arm_cache_tls_destructor()

        def create_thread_local_cache():
            cvcuda.cache_size(cvcuda.ThreadScope.LOCAL)

        thread = threading.Thread(target=create_thread_local_cache, daemon=True)
        thread.start()
        cvcuda._test.wait_cache_tls_destructor()
        cvcuda._test.release_and_wait_cache_tls_destructor()
        thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError("worker cache destructor did not finish")
        """
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "CVCUDA_CACHE_TLS_TEST_HOOK": "fail-if-destroyed"},
        )
    except subprocess.TimeoutExpired:
        pytest.fail("worker cache destructor did not finish")

    assert result.returncode == 0, result.stderr


def test_unanchored_worker_cache_leak_remains_charged():
    # This fallback intentionally leaks, so isolate its permanent cache charge.
    program = textwrap.dedent(
        """
        import cvcuda

        shape = (1, 16, 16, 3)
        item = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        nbytes = cvcuda.internal.nbytes_in_cache(item)
        assert nbytes > 0
        cvcuda.clear_cache()
        cvcuda.set_cache_limit_inbytes(nbytes)

        # The native worker has no Python state. Its joined TLS destructor
        # must retain the cache items without touching Python-owned objects.
        cvcuda._test.add_to_unanchored_worker_cache(item)
        del item
        assert cvcuda.current_cache_size_inbytes() == nbytes

        another = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        assert cvcuda.cache_size(cvcuda.ThreadScope.LOCAL) == 0
        assert cvcuda.current_cache_size_inbytes() == nbytes
        del another

        cvcuda.clear_cache()
        assert cvcuda.current_cache_size_inbytes() == nbytes
        cvcuda.set_cache_limit_inbytes(0)
        assert cvcuda.current_cache_size_inbytes() == nbytes
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
        # This subprocess must report its one deliberate leak, even when the
        # surrounding test suite aborts on unexpected leaks.
        env={**os.environ, "NVCV_LEAK_DETECTION": "warn"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.splitlines() == [
        "WARNING: Tensor leak detection: 1 handle still in use"
    ]


def test_clear_cache_inside_op():
    tensor = cvcuda.Tensor(
        (100, 1500, 1500, 3), cvcuda.Type.U8, cvcuda.TensorLayout.NHWC
    )
    map = cvcuda.Tensor((100, 1500, 1500, 2), cvcuda.Type.F32, cvcuda.TensorLayout.NHWC)
    with cvcuda.Stream():
        out = cvcuda.remap(tensor, map)
        cvcuda.clear_cache()
    del tensor
    del map
    del out
    gc.collect()


def test_clear_cache_empties_gcbag():
    # Make sure there's no work scheduled on the stream, it's all ours.
    workstream = cvcuda.Stream()

    # In order to test if the GCBag was really emptied,

    # we create a CUDA buffer,
    tensor = cupy.asarray(np.ndarray([100, 1500, 1500, 3], np.uint8))
    # keep track of its initial refcount.
    orig_tensor_refcount = sys.getrefcount(tensor)
    # and wrap it in a cvcuda tensor 'cvwrapper'
    cvwrapper = cvcuda.as_tensor(tensor, cvcuda.TensorLayout.NHWC)

    # We can then indirectly tell if 'cvwrapper' was destroyed by
    # monitoring 'tensor's refcount.
    # This works because we know 'cvwrapper' holds a reference to
    # 'tensor', as proved by the following assert:
    wrapped_tensor_refcount = sys.getrefcount(tensor)
    assert wrapped_tensor_refcount > orig_tensor_refcount

    # We need now to make sure cvwrapper is in the GCBag.
    # For that, we need to use it in operator
    with workstream:
        cvcuda.median_blur(cvwrapper, [3, 3], stream=workstream)
        # And make sure it finishes.
        workstream.sync()
    # cvwrapper being referenced by others shouldn't change tensor's refcount.
    assert sys.getrefcount(tensor) == wrapped_tensor_refcount

    # Clearing the cache must also drain completed resource holds. Otherwise
    # callers need to submit an unrelated operator before memory is released.
    cvcuda.clear_cache()

    # We can now release it from python side. We can't track its lifetime
    # directly anymore.
    del cvwrapper

    # The wrapped tensor has the same refcount it had when we've created it.
    assert sys.getrefcount(tensor) == orig_tensor_refcount


def test_cache_limit_get_set():
    cvcuda.clear_cache()

    # Verify initial cache limit (half of total gpu mem)
    total = cupy.cuda.Device().mem_info[1]
    assert cvcuda.get_cache_limit_inbytes() == total // 2

    # Verify we can also set the cache limit
    cvcuda.set_cache_limit_inbytes(total)
    assert cvcuda.get_cache_limit_inbytes() == total


def test_cache_current_byte_size():
    cvcuda.clear_cache()

    cvcuda_cache_size = 0
    assert cvcuda.current_cache_size_inbytes() == cvcuda_cache_size

    img_create = cvcuda.Image.zeros((1, 1), cvcuda.Format.F32)
    cvcuda_cache_size += cvcuda.internal.nbytes_in_cache(img_create)
    assert cvcuda.current_cache_size_inbytes() == cvcuda_cache_size

    image_batch_create = cvcuda.ImageBatchVarShape(5)
    cvcuda_cache_size += cvcuda.internal.nbytes_in_cache(image_batch_create)
    assert cvcuda.current_cache_size_inbytes() == cvcuda_cache_size

    stream = cvcuda.Stream()
    cvcuda_cache_size += cvcuda.internal.nbytes_in_cache(stream)
    assert cvcuda.current_cache_size_inbytes() == cvcuda_cache_size

    tensor_create = cvcuda.Tensor(2, (37, 7), cvcuda.Format.RGB8, rowalign=1)
    cvcuda_cache_size += cvcuda.internal.nbytes_in_cache(tensor_create)
    assert cvcuda.current_cache_size_inbytes() == cvcuda_cache_size

    tensor_batch_create = cvcuda.TensorBatch(10)
    cvcuda_cache_size += cvcuda.internal.nbytes_in_cache(tensor_batch_create)
    assert cvcuda.current_cache_size_inbytes() == cvcuda_cache_size


def test_clear_cache_local_credits_back_the_byte_count():
    cvcuda.clear_cache()

    item = cvcuda.Image.zeros((1024, 1024), cvcuda.Format.RGB8)
    assert cvcuda.current_cache_size_inbytes() == cvcuda.internal.nbytes_in_cache(item)

    cvcuda.clear_cache(cvcuda.ThreadScope.LOCAL)
    assert cvcuda.cache_size() == 0
    # A LOCAL clear used to drop the items without crediting their bytes back,
    # so this kept reporting a cache that no longer existed.
    assert cvcuda.current_cache_size_inbytes() == 0


def test_clear_cache_local_does_not_evict_another_thread():
    def image(side):
        return cvcuda.Image.zeros((side, side), cvcuda.Format.RGB8)

    cvcuda.clear_cache()
    # Increasing sizes require separate allocations even with capacity reuse.
    worker_first_nbytes = cvcuda.internal.nbytes_in_cache(image(512))
    worker_second_nbytes = cvcuda.internal.nbytes_in_cache(image(1024))
    cvcuda.clear_cache()

    original_limit = cvcuda.get_cache_limit_inbytes()
    # Room for exactly the worker's two images, so what decides whether its
    # second add evicts is the byte total the main thread leaves behind.
    cvcuda.set_cache_limit_inbytes(worker_first_nbytes + worker_second_nbytes)

    cached_first = threading.Event()
    main_cleared = threading.Event()
    worker_size = {}

    def worker():
        image(512)
        worker_size["before"] = cvcuda.cache_size()
        cached_first.set()
        assert main_cleared.wait(timeout=30)
        image(1024)
        worker_size["after"] = cvcuda.cache_size()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert cached_first.wait(timeout=30)
        image(256)
        cvcuda.clear_cache(cvcuda.ThreadScope.LOCAL)
    finally:
        main_cleared.set()
        thread.join(timeout=30)
        cvcuda.set_cache_limit_inbytes(original_limit)

    # The main thread's clear credited its bytes back, so the worker's second
    # add stayed under the limit. Without that it saw a cache that looked full
    # and evicted the image it had already cached.
    assert worker_size["after"] == worker_size["before"] + 1


def test_global_cache_limit_is_atomic_across_thread_local_caches():
    cvcuda.clear_cache()
    original_limit = cvcuda.get_cache_limit_inbytes()
    first = cvcuda.Tensor((1, 257, 257, 3), cvcuda.Type.U8, "NHWC")
    item_nbytes = cvcuda.internal.nbytes_in_cache(first)
    cvcuda.clear_cache()
    cvcuda.set_cache_limit_inbytes(item_nbytes)

    first = cvcuda.Tensor((1, 257, 257, 3), cvcuda.Type.U8, "NHWC")
    worker_done = threading.Event()
    release_worker = threading.Event()

    def worker():
        second = cvcuda.Tensor((1, 256, 256, 3), cvcuda.Type.U8, "NHWC")
        worker_done.set()
        assert release_worker.wait(timeout=30)
        assert second.shape == (1, 256, 256, 3)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert worker_done.wait(timeout=30)
        assert cvcuda.current_cache_size_inbytes() <= item_nbytes
    finally:
        release_worker.set()
        thread.join(timeout=30)
        del first
        cvcuda.clear_cache()
        cvcuda.set_cache_limit_inbytes(original_limit)


def test_cache_external_cacheitem():
    cvcuda.clear_cache()

    input_tensor = np.random.rand(2, 30, 16, 1).astype(np.uint8)
    input_tensor = input_tensor * 255
    input_tensor = cupy.asarray(input_tensor)
    frames_cvcuda = cvcuda.as_tensor(input_tensor, "NHWC")
    assert cvcuda.current_cache_size_inbytes() == 0

    frames_cvcuda_out = cvcuda.advcvtcolor(
        frames_cvcuda, cvcuda.ColorConversion.YUV2RGB_NV12, cvcuda.ColorSpec.BT2020
    )
    assert (
        cvcuda.current_cache_size_inbytes()
        == cvcuda.internal.nbytes_in_cache(frames_cvcuda_out)
    ) and (cvcuda.internal.nbytes_in_cache(frames_cvcuda_out) > 0)


def test_cache_limit_clearing():
    cvcuda.clear_cache()

    img_create = cvcuda.Image.zeros((1, 1), cvcuda.Format.F32)
    img_cache_size = cvcuda.internal.nbytes_in_cache(img_create)

    # Cache should be emptied if new set limit is smaller than current cache size
    cvcuda.set_cache_limit_inbytes(img_cache_size - 1)
    assert cvcuda.current_cache_size_inbytes() == 0
    del img_create

    # Element should not be added to Cache, if its size exceeds cache limit
    cvcuda.set_cache_limit_inbytes(img_cache_size - 1)
    img_create = cvcuda.Image.zeros((1, 1), cvcuda.Format.F32)
    assert cvcuda.current_cache_size_inbytes() == 0
    del img_create

    # If cache grows too large, cache should be emptied and new element should be added
    cvcuda.set_cache_limit_inbytes(img_cache_size)
    img_create = cvcuda.Image.zeros((1, 1), cvcuda.Format.F32)
    assert cvcuda.current_cache_size_inbytes() == img_cache_size
    img_create2 = cvcuda.Image.zeros((1, 1), cvcuda.Format.F32)
    assert cvcuda.current_cache_size_inbytes() == img_cache_size
    del img_create
    del img_create2


def test_cache_zero_cache_limit():
    cvcuda.set_cache_limit_inbytes(0)

    assert cvcuda.get_cache_limit_inbytes() == 0

    img_create = cvcuda.Image.zeros((1, 1), cvcuda.Format.F32)
    assert cvcuda.internal.nbytes_in_cache(img_create) > 0
    assert cvcuda.current_cache_size_inbytes() == 0


def test_cache_negative_cache_limit():
    with pytest.raises(ValueError):
        cvcuda.set_cache_limit_inbytes(-1)


def test_parallel_cache_size():
    """Check that the cache size is properly synced accross threads."""

    def create_tensors(thread_no: int, h: int, w: int):
        N = items_per_thread[thread_no]

        for _ in range(N):
            tensor = cvcuda.Tensor((h, w), np.uint8)
            tensors.append(tensor)

        assert cvcuda.cache_size(cvcuda.ThreadScope.LOCAL) == N
        assert (
            N <= cvcuda.cache_size(cvcuda.ThreadScope.GLOBAL) <= nb_items * nb_threads
        )

        # Keep all threads alive until the assertions
        barrier.wait()

    # Ensure that the cache limit was not altered by another test
    cvcuda.set_cache_limit_inbytes(cupy.cuda.Device().mem_info[1] // 2)
    cvcuda.clear_cache()

    nb_threads = len(os.sched_getaffinity(0))
    items_per_thread = RNG.integers(50, 200, size=nb_threads)
    nb_items = items_per_thread.sum()
    tensors = []
    barrier = threading.Barrier(nb_threads)
    util.run_parallel(create_tensors, 16, 32)

    assert cvcuda.cache_size(cvcuda.ThreadScope.LOCAL) == 0
    # Wait a bit for worker thread C++ Cache destructors to run and update the global state
    time.sleep(1)  # 1 second is enough for now in our case.

    # Other threads have been destroyed - the cache is empty again
    assert (
        cvcuda.cache_size(cvcuda.ThreadScope.GLOBAL)
        == cvcuda.current_cache_size_inbytes()
        == 0
    )


def test_parallel_clear_cache():
    """Make sure that cvcuda.clear_cache clears the cache for all threads."""

    def clear_cache():
        done_event.wait()  # wait for the main thread to be ready
        cvcuda.clear_cache()
        clear_event.set()  # notify that the cache has been cleared

    # Ensure that the cache limit was not altered by another test
    cvcuda.set_cache_limit_inbytes(cupy.cuda.Device().mem_info[1] // 2)
    cvcuda.clear_cache()

    done_event = threading.Event()
    clear_event = threading.Event()
    clear_thread = threading.Thread(target=clear_cache, daemon=True)
    clear_thread.start()

    h, w = 16, 32
    cvcuda.Tensor((h, w), np.uint8)
    size_inbytes = cvcuda.current_cache_size_inbytes()
    assert cvcuda.cache_size() == 1
    assert size_inbytes > 0

    done_event.set()
    clear_event.wait()

    assert cvcuda.cache_size() == cvcuda.current_cache_size_inbytes() == 0
    cvcuda.Tensor((h, w), np.uint8)
    assert cvcuda.cache_size() == 1
    assert cvcuda.current_cache_size_inbytes() == size_inbytes


# ---------------------------------------------------------------------------
# Multi-GPU cache tests (skipped when fewer than 2 GPUs are available)
# ---------------------------------------------------------------------------

_err, NUM_GPUS = cudart.cudaGetDeviceCount()
if _err != cudart.cudaError_t.cudaSuccess:
    NUM_GPUS = 0

requires_multi_gpu = pytest.mark.skipif(
    NUM_GPUS < 2,
    reason="Multi-GPU cache tests require at least 2 GPUs",
)


@pytest.fixture()
def _restore_device_and_limits():
    """Restore CUDA device 0 and per-device cache limits after each multi-GPU cache test."""
    yield
    for gpu_id in range(NUM_GPUS):
        cudart.cudaSetDevice(gpu_id)
        total = cupy.cuda.Device().mem_info[1]
        cvcuda.set_cache_limit_inbytes(total // 2)
    cudart.cudaSetDevice(0)
    cvcuda.clear_cache()


def _image(side):
    return cvcuda.Image.zeros((side, side), cvcuda.Format.RGB8)


def test_add_eviction_enforces_limit_across_thread_local_caches():
    cvcuda.clear_cache()
    main_nbytes = cvcuda.internal.nbytes_in_cache(_image(512))
    cvcuda.clear_cache()

    cached = threading.Event()
    release = threading.Event()

    def worker():
        _image(1024)
        cached.set()
        assert release.wait(timeout=30)

    thread = threading.Thread(target=worker)
    thread.start()
    original_limit = cvcuda.get_cache_limit_inbytes()
    try:
        assert cached.wait(timeout=30)
        # Lowering the process-wide limit evicts eligible entries regardless of
        # which thread-local lookup cache owns them.
        cvcuda.set_cache_limit_inbytes(main_nbytes)
        _image(512)

        assert cvcuda.current_cache_size_inbytes() == main_nbytes
    finally:
        release.set()
        thread.join(timeout=30)
        cvcuda.set_cache_limit_inbytes(original_limit)
        cvcuda.clear_cache()


def test_lowering_the_limit_evicts_across_thread_local_caches():
    cvcuda.clear_cache()
    worker_nbytes = cvcuda.internal.nbytes_in_cache(_image(1024))
    cvcuda.clear_cache()

    cached = threading.Event()
    release = threading.Event()

    def worker():
        _image(1024)
        cached.set()
        assert release.wait(timeout=30)

    thread = threading.Thread(target=worker)
    thread.start()
    original_limit = cvcuda.get_cache_limit_inbytes()
    try:
        assert cached.wait(timeout=30)
        _image(512)
        # Either cache may supply the evicted item; the global result is what
        # the per-device limit promises.
        cvcuda.set_cache_limit_inbytes(worker_nbytes)
        assert cvcuda.current_cache_size_inbytes() <= worker_nbytes
    finally:
        release.set()
        thread.join(timeout=30)
        cvcuda.set_cache_limit_inbytes(original_limit)
        cvcuda.clear_cache()


@requires_multi_gpu
@pytest.mark.usefixtures("_restore_device_and_limits")
def test_per_device_cache_limits():
    """Cache limits, size accounting, and eviction must be independent per device."""
    cvcuda.clear_cache()

    # 1. Default limit is per-device: half of each GPU's total memory.
    for gpu_id in range(NUM_GPUS):
        cudart.cudaSetDevice(gpu_id)
        total = cupy.cuda.Device().mem_info[1]
        assert cvcuda.get_cache_limit_inbytes() == total // 2

    # 2. Setting limit on device 0 does not affect device 1.
    cudart.cudaSetDevice(0)
    cvcuda.set_cache_limit_inbytes(12345)
    assert cvcuda.get_cache_limit_inbytes() == 12345

    cudart.cudaSetDevice(1)
    total_1 = cupy.cuda.Device().mem_info[1]
    assert cvcuda.get_cache_limit_inbytes() == total_1 // 2

    # 3. Size accounting is per-device.
    cudart.cudaSetDevice(0)
    total_0 = cupy.cuda.Device().mem_info[1]
    cvcuda.set_cache_limit_inbytes(total_0 // 2)
    cvcuda.clear_cache()

    cudart.cudaSetDevice(0)
    img0 = cvcuda.Image.zeros((32, 32), cvcuda.Format.RGB8)
    size0 = cvcuda.current_cache_size_inbytes()
    assert size0 > 0

    cudart.cudaSetDevice(1)
    assert cvcuda.current_cache_size_inbytes() == 0

    img1 = cvcuda.Image.zeros((32, 32), cvcuda.Format.RGB8)
    size1 = cvcuda.current_cache_size_inbytes()
    assert size1 > 0

    cudart.cudaSetDevice(0)
    assert cvcuda.current_cache_size_inbytes() == size0

    # 4. Eviction on device 0 does not affect device 1.
    del img0
    cudart.cudaSetDevice(0)
    img_size = cvcuda.internal.nbytes_in_cache(
        cvcuda.Image.zeros((32, 32), cvcuda.Format.RGB8)
    )
    cvcuda.set_cache_limit_inbytes(img_size - 1)
    assert cvcuda.current_cache_size_inbytes() == 0

    cudart.cudaSetDevice(1)
    assert cvcuda.current_cache_size_inbytes() == size1

    del img1


def _device_ptr(tensor):
    return tensor.cuda().__cuda_array_interface__["data"][0]


@pytest.mark.parametrize("held_count", [1, 8, 64])
def test_cache_never_hands_back_an_in_use_buffer(held_count):
    """A cached item is only reusable once nothing else holds it.

    The cache looks up by key and returns the first item that is not in use.
    Handing back one that is still referenced would alias two Python objects
    onto the same device buffer, so every live tensor must have a distinct
    pointer no matter how many same-key items the lookup has to walk past.
    """
    cvcuda.clear_cache()
    gc.collect()

    shape, dtype = (1, 8, 8, 3), cvcuda.Type.U8
    held = [
        cvcuda.Tensor(shape, dtype, cvcuda.TensorLayout.NHWC) for _ in range(held_count)
    ]
    pointers = {_device_ptr(t) for t in held}
    assert len(pointers) == held_count, "cache aliased two live tensors onto one buffer"

    # Every held tensor is in use, so this must not reuse any of them.
    fresh = cvcuda.Tensor(shape, dtype, cvcuda.TensorLayout.NHWC)
    assert (
        _device_ptr(fresh) not in pointers
    ), "cache returned a buffer that was still in use"

    del held, fresh
    gc.collect()
    cvcuda.clear_cache()


def test_cache_reuses_released_buffers():
    """Released items must come back, otherwise the lookup is not a cache.

    Guards the other direction from the test above: skipping in-use items must
    not turn into skipping everything and allocating afresh each time.
    """
    cvcuda.clear_cache()
    gc.collect()

    shape, dtype = (1, 16, 16, 3), cvcuda.Type.U8
    first = cvcuda.Tensor(shape, dtype, cvcuda.TensorLayout.NHWC)
    first_ptr = _device_ptr(first)
    del first
    gc.collect()

    reused = cvcuda.Tensor(shape, dtype, cvcuda.TensorLayout.NHWC)
    assert _device_ptr(reused) == first_ptr, "released buffer was not reused"

    del reused
    gc.collect()
    cvcuda.clear_cache()
