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

import os
import threading
import time
import nvcv
import cvcuda
import torch
import pytest
import numpy as np

import nvcv_util as util

RNG = np.random.default_rng(12345)


def test_cache_limit_get_set():
    nvcv.clear_cache()

    # Verify initial cache limit (half of total gpu mem)
    total = torch.cuda.mem_get_info()[1]
    assert nvcv.get_cache_limit_inbytes() == total // 2

    # Verify we can also set the cache limit
    nvcv.set_cache_limit_inbytes(total)
    assert nvcv.get_cache_limit_inbytes() == total


def test_cache_current_byte_size():
    nvcv.clear_cache()

    nvcv_cache_size = 0
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(img_create)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    image_batch_create = nvcv.ImageBatchVarShape(5)
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(image_batch_create)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    stream = nvcv.cuda.Stream()
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(stream)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    tensor_create = nvcv.Tensor(2, (37, 7), nvcv.Format.RGB8, rowalign=1)
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(tensor_create)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    tensor_batch_create = nvcv.TensorBatch(10)
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(tensor_batch_create)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size


def test_cache_external_cacheitem():
    nvcv.clear_cache()

    input_tensor = torch.rand(2, 30, 16, 1).cuda()
    input_tensor = input_tensor * 255
    input_tensor = input_tensor.to(dtype=torch.uint8)
    frames_cvcuda = cvcuda.as_tensor(input_tensor, "NHWC")
    assert nvcv.current_cache_size_inbytes() == 0

    frames_cvcuda_out = cvcuda.advcvtcolor(
        frames_cvcuda, cvcuda.ColorConversion.YUV2RGB_NV12, cvcuda.ColorSpec.BT2020
    )
    assert (
        nvcv.current_cache_size_inbytes()
        == nvcv.internal.nbytes_in_cache(frames_cvcuda_out)
    ) and (nvcv.internal.nbytes_in_cache(frames_cvcuda_out) > 0)


def test_cache_limit_clearing():
    nvcv.clear_cache()

    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    img_cache_size = nvcv.internal.nbytes_in_cache(img_create)

    # Cache should be emptied if new set limit is smaller than current cache size
    nvcv.set_cache_limit_inbytes(img_cache_size - 1)
    assert nvcv.current_cache_size_inbytes() == 0
    del img_create

    # Element should not be added to Cache, if its size exceeds cache limit
    nvcv.set_cache_limit_inbytes(img_cache_size - 1)
    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    assert nvcv.current_cache_size_inbytes() == 0
    del img_create

    # If cache grows too large, cache should be emptied and new element should be added
    nvcv.set_cache_limit_inbytes(img_cache_size)
    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    assert nvcv.current_cache_size_inbytes() == img_cache_size
    img_create2 = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    assert nvcv.current_cache_size_inbytes() == img_cache_size
    del img_create
    del img_create2


def test_cache_zero_cache_limit():
    nvcv.set_cache_limit_inbytes(0)

    assert nvcv.get_cache_limit_inbytes() == 0

    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    assert nvcv.internal.nbytes_in_cache(img_create) > 0
    assert nvcv.current_cache_size_inbytes() == 0


def test_cache_negative_cache_limit():
    with pytest.raises(ValueError):
        nvcv.set_cache_limit_inbytes(-1)


def test_parallel_cache_size():
    """Check that the cache size is properly synced accross threads."""

    def create_tensors(thread_no: int, h: int, w: int):
        N = items_per_thread[thread_no]

        for _ in range(N):
            tensor = nvcv.Tensor((h, w), np.uint8)
            tensors.append(tensor)

        assert nvcv.cache_size(nvcv.ThreadScope.LOCAL) == N
        assert N <= nvcv.cache_size(nvcv.ThreadScope.GLOBAL) <= nb_items * nb_threads

        # Keep all threads alive until the assertions
        barrier.wait()

    # Ensure that the cache limit was not altered by another test
    nvcv.set_cache_limit_inbytes(torch.cuda.mem_get_info()[1] // 2)
    nvcv.clear_cache()

    nb_threads = len(os.sched_getaffinity(0))
    items_per_thread = RNG.integers(50, 200, size=nb_threads)
    nb_items = items_per_thread.sum()
    tensors = []
    barrier = threading.Barrier(nb_threads)
    util.run_parallel(create_tensors, 16, 32)

    assert nvcv.cache_size(nvcv.ThreadScope.LOCAL) == 0
    # Wait a bit for worker thread C++ Cache destructors to run and update the global state
    time.sleep(1)  # 1 second is enough for now in our case.

    # Other threads have been destroyed - the cache is empty again
    assert (
        nvcv.cache_size(nvcv.ThreadScope.GLOBAL)
        == nvcv.current_cache_size_inbytes()
        == 0
    )


def test_parallel_clear_cache():
    """Make sure that nvcv.clear_cache clears the cache for all threads."""

    def clear_cache():
        done_event.wait()  # wait for the main thread to be ready
        nvcv.clear_cache()
        clear_event.set()  # notify that the cache has been cleared

    # Ensure that the cache limit was not altered by another test
    nvcv.set_cache_limit_inbytes(torch.cuda.mem_get_info()[1] // 2)
    nvcv.clear_cache()

    done_event = threading.Event()
    clear_event = threading.Event()
    clear_thread = threading.Thread(target=clear_cache, daemon=True)
    clear_thread.start()

    h, w = 16, 32
    nvcv.Tensor((h, w), np.uint8)
    size_inbytes = nvcv.current_cache_size_inbytes()
    assert nvcv.cache_size() == 1
    assert size_inbytes > 0

    done_event.set()
    clear_event.wait()

    assert nvcv.cache_size() == nvcv.current_cache_size_inbytes() == 0
    nvcv.Tensor((h, w), np.uint8)
    assert nvcv.cache_size() == 1
    assert nvcv.current_cache_size_inbytes() == size_inbytes
