# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re

import numpy as np
import pytest as t

import cvcuda
import cvcuda_util as util
import cupy


def rand_shape(rank, low=1, high=10):
    return np.random.randint(low=low, high=high, size=rank)


def rand_cuda_buffer(dtype, rank):
    return cupy.asarray(np.random.random(size=rand_shape(rank)).astype(dtype))


def random_tensors(n, dtype, rank, layout):
    return [
        cvcuda.as_tensor(rand_cuda_buffer(dtype, rank), layout=layout) for _ in range(n)
    ]


def test_tensorbatch_creation_works():
    batch = cvcuda.TensorBatch(15)
    assert batch.capacity == 15
    assert len(batch) == 0
    assert batch.layout is None
    assert batch.dtype is None
    assert batch.ndim == -1

    # range must be empty
    cnt = 0
    for i in batch:
        cnt += 1
    assert cnt == 0


def test_tensorbatch_one_tensor():
    batch = cvcuda.TensorBatch(15)

    tensor = cvcuda.as_tensor(cvcuda.Image((64, 32), cvcuda.Format.RGBA8))
    batch.pushback(tensor)
    assert len(batch) == 1
    assert batch.layout == "NHWC"
    assert batch.dtype == np.uint8
    assert batch.ndim == 4
    assert list(batch) == [tensor]

    # range must contain one
    cnt = 0
    for elem in batch:
        assert elem is tensor
        cnt += 1
    assert cnt == 1

    # remove added tensor
    batch.popback()

    # check if its indeed removed
    assert len(batch) == 0
    assert list(batch) == []


def test_tensorbatch_change_layout():
    batch = cvcuda.TensorBatch(10)
    tensorsA = random_tensors(5, np.float32, 3, "HWC")
    batch.pushback(tensorsA)
    assert list(batch) == tensorsA
    assert batch.layout == "HWC"
    assert batch.dtype == np.float32
    assert batch.ndim == 3

    batch.popback(len(tensorsA))
    assert list(batch) == []
    assert batch.layout is None
    assert batch.dtype is None
    assert batch.ndim == -1

    tensorsB = [
        cvcuda.as_tensor(cvcuda.Image(rand_shape(2), cvcuda.Format.RGBA8))
        for _ in range(7)
    ]
    batch.pushback(tensorsB)
    assert list(batch) == tensorsB
    assert batch.layout == "NHWC"
    assert batch.dtype == np.uint8
    assert batch.ndim == 4

    batch.clear()
    assert list(batch) == []
    assert batch.layout is None
    assert batch.dtype is None
    assert batch.ndim == -1


def test_tensorbatch_multiply_tensors():
    N = 10
    tensorsA = random_tensors(5, np.int16, 3, "HWC")
    batch = cvcuda.TensorBatch(len(tensorsA) * N)
    for _ in range(N):
        batch.pushback(tensorsA)

    assert list(batch) == tensorsA * N
    assert batch.layout == "HWC"
    assert batch.dtype == np.int16
    assert batch.ndim == 3


def test_tensorbatch_subscript():
    tensorsA = random_tensors(10, np.float32, 3, "HWC")
    batch = cvcuda.TensorBatch(10)
    batch.pushback(tensorsA)

    # test get item
    for i in range(len(batch)):
        assert batch[i] is tensorsA[i]

    # out of bounds subscript
    with t.raises(
        RuntimeError,
        match=f"Cannot get tensor at index {len(tensorsA)}. Batch has only {len(tensorsA)} elements.",
    ):
        batch[len(tensorsA)]

    # test set item
    tensorsB = random_tensors(5, np.float32, 3, "HWC")
    for i in range(len(tensorsB)):
        batch[i] = tensorsB[i]

    for i in range(len(batch)):
        if i < len(tensorsB):
            assert batch[i] is tensorsB[i]
        else:
            assert batch[i] is tensorsA[i]


def test_tensorbatch_wrap_buffers():
    # from cuda buffer, without layout
    buffers = [
        util.to_cuda_buffer(np.ones(rand_shape(3), dtype=np.int32)) for _ in range(10)
    ]
    batch = cvcuda.as_tensors(buffers)
    assert batch.capacity == len(buffers)
    assert len(batch) == len(buffers)
    assert batch.dtype == np.int32
    assert batch.layout is None
    assert batch.ndim == 3

    # from CUDA buffer, with layout
    buffers = [rand_cuda_buffer(np.int16, 4) for i in range(5)]
    batch = cvcuda.as_tensors(buffers, layout="NHWC")
    assert batch.capacity == len(buffers)
    assert len(batch) == len(buffers)
    assert batch.dtype == np.int16
    assert batch.layout == "NHWC"
    assert batch.ndim == 4

    # mismatching rank
    with t.raises(
        RuntimeError,
        match="NVCV_ERROR_INVALID_ARGUMENT: "
        "Trying to add a tensor to a tensor batch with an inconsistent rank.",
    ):
        buffers = [rand_cuda_buffer(np.int16, 3), rand_cuda_buffer(np.int16, 4)]
        cvcuda.as_tensors(buffers)

    # mismatching dtype
    with t.raises(
        RuntimeError,
        match="NVCV_ERROR_INVALID_ARGUMENT: "
        "Trying to add a tensor to a tensor batch with an inconsistent type.",
    ):
        buffers = [rand_cuda_buffer(np.int16, 3), rand_cuda_buffer(np.int32, 3)]
        cvcuda.as_tensors(buffers)

    # invalid types
    with t.raises(
        RuntimeError,
        match="Input buffer doesn't provide cuda_array_interface or DLPack interfaces.",
    ):
        buffers = [[1, 2, 3]]
        cvcuda.as_tensors(buffers)


def test_tensorbatch_errors():
    with t.raises(
        RuntimeError,
        match=re.escape(
            "NVCV_ERROR_OVERFLOW: Adding 2 tensors to a tensor batch would exceed its capacity (2) by 1"
        ),
    ):
        batch = cvcuda.TensorBatch(2)
        batch.pushback(random_tensors(1, np.int16, 3, ""))
        batch.pushback(random_tensors(2, np.int16, 3, ""))

    with t.raises(
        RuntimeError,
        match="NVCV_ERROR_UNDERFLOW: Trying to pop 3 tensors from a tensor batch with 2 tensors.",
    ):
        batch = cvcuda.TensorBatch(5)
        batch.pushback(random_tensors(2, np.int16, 3, ""))
        batch.popback(3)

    with t.raises(
        RuntimeError,
        match="NVCV_ERROR_INVALID_ARGUMENT: "
        "Trying to add a tensor to a tensor batch with an inconsistent layout.",
    ):
        batch = cvcuda.TensorBatch(10)
        batch.pushback(random_tensors(2, np.int16, 4, "NHWC"))
        batch.pushback(random_tensors(3, np.int16, 4, "FHWC"))


def test_tensorbatch_size_in_bytes():
    """
    Checks if the computation of the TensorBatch size in bytes is correct
    """
    batch_create = cvcuda.TensorBatch(10)
    assert cvcuda.internal.nbytes_in_cache(batch_create) > 0

    pt_img = cupy.asarray(np.ndarray((16, 32, 4), dtype=np.float32))
    batch_as_tensors = cvcuda.as_tensors([pt_img])
    assert cvcuda.internal.nbytes_in_cache(batch_as_tensors) > 0
