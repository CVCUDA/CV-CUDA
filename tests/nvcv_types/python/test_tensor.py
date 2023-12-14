# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import torch


@t.mark.parametrize(
    "n,size,fmt,gold_layout,gold_shape,gold_dtype",
    [
        (
            5,
            (32, 16),
            nvcv.Format.RGBA8,
            nvcv.TensorLayout.NHWC,
            (5, 16, 32, 4),
            np.uint8,
        ),
        (
            2,
            (38, 7),
            nvcv.Format.RGB8p,
            nvcv.TensorLayout.NCHW,
            (2, 3, 7, 38),
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
        ((5, 16, 32, 4), np.float32, nvcv.TensorLayout.NHWC),
        ((7, 3, 33, 11), np.complex64, nvcv.TensorLayout.NCHW),
        ((3, 11), np.int16, None),
        ((16, 32, 4), np.float32, nvcv.TensorLayout.HWC),
        ((32, 4), np.float32, nvcv.TensorLayout.WC),
        ((4, 32), np.float32, nvcv.TensorLayout.CW),
        ((32,), np.float32, nvcv.TensorLayout.W),
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


params_wrap_torch = [
    ((3, 5, 7, 1), np.uint8),
    ((3, 5, 7, 1), np.int8),
    ((3, 5, 7, 1), np.int16),
    ((3, 5, 7, 1), np.float32),
    ((3, 5, 7, 1), np.float64),
    ((3, 5, 7, 2), np.float32),
    ((3, 5, 7, 3), np.uint8),
    ((3, 5, 7, 4), np.uint8),
    ((3, 5, 7), np.csingle),
    ((3, 5, 7), np.cdouble),
    ((3,), np.int8),
]


@t.mark.parametrize("shape,dtype", params_wrap_torch)
def test_wrap_torch_buffer(shape, dtype):
    tensor = nvcv.as_tensor(
        torch.as_tensor(np.ndarray(shape, dtype=dtype), device="cuda")
    )
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout is None
    assert tensor.ndim == len(shape)


@t.mark.parametrize("shape,dtype", params_wrap_torch)
def test_wrap_torch_buffer_dlpack(shape, dtype):
    ttensor = torch.as_tensor(np.ndarray(shape, dtype=dtype), device="cuda")

    # Since nvcv.as_tensor can understand both dlpack and cuda_array_interface,
    # and we don't know a priori which interfaces it'll use (torch provides both),
    # let's create one object with only the dlpack interface.
    class DLPackObject:
        pass

    o = DLPackObject()
    o.__dlpack__ = ttensor.__dlpack__
    o.__dlpack_device__ = ttensor.__dlpack_device__

    tensor = nvcv.as_tensor(o)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout is None
    assert tensor.ndim == len(shape)


@t.mark.parametrize("shape,dtype", params_wrap_torch)
def test_wrap_torch_buffer_cuda_array_interface(shape, dtype):
    ttensor = torch.as_tensor(np.ndarray(shape, dtype=dtype), device="cuda")

    # Since nvcv.as_tensor can understand both dlpack and cuda_array_interface,
    # and we don't know a priori which interfaces it'll use (torch provides both),
    # let's create one object with only the cuda_array_interface.
    class CudaArrayInterfaceObject:
        pass

    o = CudaArrayInterfaceObject()
    o.__cuda_array_interface__ = ttensor.__cuda_array_interface__

    tensor = nvcv.as_tensor(o)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout is None
    assert tensor.ndim == len(shape)


@t.mark.parametrize(
    "shape,dtype,layout",
    [
        ((3, 5, 7, 1), np.uint8, "NHWC"),
        ((3, 5, 7), np.uint8, "HWC"),
        ((3, 5, 7, 2), np.int16, "NHWC"),
        ((3, 5, 7, 2, 4, 2, 5), np.int16, "abcdefg"),
        ((3, 5), np.uint8, "HW"),
        ((5,), np.uint8, "W"),
    ],
)
def test_wrap_torch_buffer_with_layout(shape, dtype, layout):
    tensor = nvcv.as_tensor(
        torch.as_tensor(np.ndarray(shape, dtype=dtype), device="cuda"), layout
    )
    assert tensor.shape == shape
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
            (1, 16, 32, 4),
            np.uint8,
        ),
        (
            (38, 7),
            nvcv.Format.RGB8p,
            nvcv.TensorLayout.NCHW,
            (1, 3, 7, 38),
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


export_cuda_buffer_params = [
    ((1, 23, 65, 3), np.uint8),
    ((5, 23, 65, 3), np.int8),
    ((65, 3), np.int16),
    ((243, 65, 3), np.int16),
    ((1, 1), np.int16),
    ((10,), np.uint8),
]


@t.mark.parametrize(
    "shape,dtype",
    export_cuda_buffer_params,
)
def test_tensor_export_cuda_buffer(shape, dtype):
    rng = np.random.default_rng()
    hostGold = rng.integers(0, 128, shape, dtype)

    devGold = torch.as_tensor(hostGold, device="cuda")

    tensor = nvcv.as_tensor(devGold)

    devMem = tensor.cuda()
    assert devMem.dtype == dtype
    assert devMem.shape == shape

    assert (hostGold == torch.as_tensor(devMem).cpu().numpy()).all()


@t.mark.parametrize(
    "shape,dtype",
    export_cuda_buffer_params,
)
def test_tensor_export_cuda_buffer_dlpack(shape, dtype):
    rng = np.random.default_rng()
    hostGold = rng.integers(0, 128, shape, dtype)

    devGold = torch.as_tensor(hostGold, device="cuda")

    tensor = nvcv.as_tensor(devGold)

    devMem = tensor.cuda()
    assert devMem.dtype == dtype
    assert devMem.shape == shape

    assert (hostGold == torch.from_dlpack(devMem).cpu().numpy()).all()


def test_tensor_hold_reference_of_wrapped_buffer():
    ttensor = torch.as_tensor(np.ndarray([10], np.int8), device="cuda")
    ptr0 = ttensor.data_ptr()

    cvtensor = nvcv.as_tensor(ttensor)  # noqa: F841 assigned but never used

    del ttensor  # cvtensor must have held ttensor object

    ttensor = torch.as_tensor(np.ndarray([10], np.int8), device="cuda")

    # since "cvtensor" must have held the reference to the first "ttensor",
    # the second "ttensor" must be a different buffer
    assert ptr0 != ttensor.data_ptr()


def test_tensor_is_kept_alive_by_cuda_array_interface():
    nvcv.clear_cache()

    tensor1 = nvcv.Tensor((480, 640, 3), np.uint8)

    iface1 = tensor1.cuda()

    data_buffer1 = iface1.__cuda_array_interface__["data"][0]

    del tensor1

    tensor2 = nvcv.Tensor((480, 640, 3), np.uint8)
    assert tensor2.cuda().__cuda_array_interface__["data"][0] != data_buffer1

    del tensor2
    # remove tensor2 from cache, but not tensor1, as it's being
    # held by iface
    nvcv.clear_cache()

    # now tensor1 is free for reuse
    del iface1

    tensor3 = nvcv.Tensor((480, 640, 3), np.uint8)
    assert tensor3.cuda().__cuda_array_interface__["data"][0] == data_buffer1


def test_tensor_create_packed():
    tensor = nvcv.Tensor((37, 11, 3), np.uint8, rowalign=1)
    assert tensor.cuda().strides == (11 * 3, 3, 1)


def test_tensor_create_for_imgbatch_packed():
    tensor = nvcv.Tensor(2, (37, 7), nvcv.Format.RGB8, rowalign=1)
    assert tensor.cuda().strides == (37 * 7 * 3, 37 * 3, 3, 1)
