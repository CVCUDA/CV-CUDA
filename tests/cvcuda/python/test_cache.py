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
import sys
import threading
import time

import numpy as np
import pytest

import cuda.bindings.runtime as cudart

import cvcuda
import cvcuda_util as util
import cupy

RNG = np.random.default_rng(12345)


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
