# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvcv
import pytest as t
import numba
import numpy as np
from numba import cuda
import torch

assert numba.cuda.is_available()


@t.mark.parametrize(
    "n,size,fmt,gold_layout,gold_shape,gold_dtype",
    [
        (
            5,
            (32, 16),
            nvcv.Format.RGBA8,
            nvcv.TensorLayout.NHWC,
            [5, 16, 32, 4],
            np.uint8,
        ),
        (
            2,
            (38, 7),
            nvcv.Format.RGB8p,
            nvcv.TensorLayout.NCHW,
            [2, 3, 7, 38],
            np.uint8,
        ),
    ],
)
def test_tensor_creation_imagebatch_works(
    n, size, fmt, gold_layout, gold_shape, gold_dtype
):
    tensor = nvcv.Tensor(n, size, fmt)
    assert tensor.shape == gold_shape
    assert tensor.layout == gold_layout
    assert tensor.dtype == gold_dtype
    assert tensor.ndim == len(gold_shape)

    tensor = nvcv.Tensor(nimages=n, imgsize=size, format=fmt)
    assert tensor.shape == gold_shape
    assert tensor.layout == gold_layout
    assert tensor.dtype == gold_dtype
    assert tensor.ndim == len(gold_shape)


@t.mark.parametrize(
    "shape, dtype,layout",
    [
        ([5, 16, 32, 4], np.float32, nvcv.TensorLayout.NHWC),
        ([7, 3, 33, 11], np.complex64, nvcv.TensorLayout.NCHW),
        ([3, 11], np.int16, None),
        ([16, 32, 4], np.float32, nvcv.TensorLayout.HWC),
        ([32, 4], np.float32, nvcv.TensorLayout.WC),
        ([4, 32], np.float32, nvcv.TensorLayout.CW),
        ([32], np.float32, nvcv.TensorLayout.W),
    ],
)
def test_tensor_creation_shape_works(shape, dtype, layout):
    tensor = nvcv.Tensor(shape, dtype, layout)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout == layout
    assert tensor.ndim == len(shape)

    tensor = nvcv.Tensor(layout=layout, shape=shape, dtype=dtype)
    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tensor.shape == shape
    assert tensor.ndim == len(shape)


@t.mark.parametrize(
    "shape,dtype",
    [
        ([3, 5, 7, 1], np.uint8),
        ([3, 5, 7, 1], np.int8),
        ([3, 5, 7, 1], np.uint16),
        ([3, 5, 7, 1], np.int16),
        ([3, 5, 7, 1], np.float32),
        ([3, 5, 7, 1], np.float64),
        ([3, 5, 7, 2], np.float32),
        ([3, 5, 7, 3], np.uint8),
        ([3, 5, 7, 4], np.uint8),
        ([3, 5, 7], np.csingle),
        ([3], np.int8),
    ],
)
def test_wrap_numba_buffer(shape, dtype):
    tensor = nvcv.as_tensor(cuda.device_array(shape, dtype))
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout is None
    assert tensor.ndim == len(shape)


@t.mark.parametrize(
    "shape,dtype,layout",
    [
        ([3, 5, 7, 1], np.uint8, "NHWC"),
        ([3, 5, 7], np.uint8, "HWC"),
        ([3, 5, 7, 2], np.int16, "NHWC"),
        ([3, 5, 7, 2, 4, 2, 5], np.int16, "abcdefg"),
        ([3, 5], np.uint8, "HW"),
        ([5], np.uint8, "W"),
    ],
)
def test_wrap_numba_buffer_with_layout(shape, dtype, layout):
    tensor = nvcv.as_tensor(cuda.device_array(shape, dtype), layout)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout == layout
    assert tensor.ndim == len(shape)


@t.mark.parametrize(
    "size, fmt, gold_layout,gold_shape,gold_dtype",
    [
        (
            (32, 16),
            nvcv.Format.RGBA8,
            nvcv.TensorLayout.NHWC,
            [1, 16, 32, 4],
            np.uint8,
        ),
        (
            (38, 7),
            nvcv.Format.RGB8p,
            nvcv.TensorLayout.NCHW,
            [1, 3, 7, 38],
            np.uint8,
        ),
    ],
)
def test_tensor_wrap_image_works(size, fmt, gold_layout, gold_shape, gold_dtype):
    img = nvcv.Image(size, fmt)

    tensor = nvcv.as_tensor(img)
    assert tensor.shape == gold_shape
    assert tensor.layout == gold_layout
    assert tensor.dtype == gold_dtype


@t.mark.parametrize(
    "shape,dtype",
    [
        ([1, 23, 65, 3], np.uint8),
        ([5, 23, 65, 3], np.int8),
        ([65, 3], np.int16),
        ([243, 65, 3], np.uint16),
        ([1, 1], np.uint16),
        ([10], np.uint8),
    ],
)
def test_tensor_export_cuda_buffer(shape, dtype):

    hostGold = np.random.randint(0, 127, shape, dtype)

    devGold = cuda.to_device(hostGold)

    tensor = nvcv.as_tensor(devGold)

    devMem = tensor.cuda()
    assert devMem.dtype == dtype
    assert devMem.shape == shape

    assert (hostGold == cuda.as_cuda_array(devMem).copy_to_host()).all()


def test_tensor_hold_reference_of_wrapped_buffer():
    ttensor = torch.as_tensor(np.ndarray([10], np.int8), device="cuda")
    ptr0 = ttensor.data_ptr()

    cvtensor = nvcv.as_tensor(ttensor)  # noqa: F841 assigned but never used

    del ttensor  # cvtensor must have held ttensor object

    ttensor = torch.as_tensor(np.ndarray([10], np.int8), device="cuda")

    # since "cvtensor" must have held the reference to the first "ttensor",
    # the second "ttensor" must be a different buffer
    assert ptr0 != ttensor.data_ptr()
