# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest as t

import cvcuda
import cupy


@t.mark.parametrize(
    "n,size,fmt,gold_layout,gold_shape,gold_dtype",
    [
        (
            5,
            (32, 16),
            cvcuda.Format.RGBA8,
            cvcuda.TensorLayout.NHWC,
            (5, 16, 32, 4),
            np.uint8,
        ),
        (
            2,
            (38, 7),
            cvcuda.Format.RGB8p,
            cvcuda.TensorLayout.NCHW,
            (2, 3, 7, 38),
            np.uint8,
        ),
    ],
)
def test_tensor_creation_imagebatch_works(
    n, size, fmt, gold_layout, gold_shape, gold_dtype
):
    tensor = cvcuda.Tensor(n, size, fmt)
    assert tensor.shape == gold_shape
    assert tensor.layout == gold_layout
    assert tensor.dtype == gold_dtype
    assert tensor.ndim == len(gold_shape)

    tensor = cvcuda.Tensor(nimages=n, imgsize=size, format=fmt)
    assert tensor.shape == gold_shape
    assert tensor.layout == gold_layout
    assert tensor.dtype == gold_dtype
    assert tensor.ndim == len(gold_shape)


@t.mark.parametrize(
    "shape, dtype,layout",
    [
        ((5, 16, 32, 4), np.float32, cvcuda.TensorLayout.NHWC),
        ((7, 3, 33, 11), np.complex64, cvcuda.TensorLayout.NCHW),
        ((3, 11), np.int16, None),
        ((16, 32, 4), np.float32, cvcuda.TensorLayout.HWC),
        ((32, 4), np.float32, cvcuda.TensorLayout.WC),
        ((4, 32), np.float32, cvcuda.TensorLayout.CW),
        ((32,), np.float32, cvcuda.TensorLayout.W),
    ],
)
def test_tensor_creation_shape_works(shape, dtype, layout):
    tensor = cvcuda.Tensor(shape, dtype, layout)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout == layout
    assert tensor.ndim == len(shape)

    tensor = cvcuda.Tensor(layout=layout, shape=shape, dtype=dtype)
    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tensor.shape == shape
    assert tensor.ndim == len(shape)


params_wrap_cuda_buffer = [
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


@t.mark.parametrize("shape,dtype", params_wrap_cuda_buffer)
def test_wrap_cuda_buffer(shape, dtype):
    tensor = cvcuda.as_tensor(cupy.asarray(np.ndarray(shape, dtype=dtype)))
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout is None
    assert tensor.ndim == len(shape)


def _make_dlpack_capsule(cupy_array):
    """Build a DLPack v0 PyCapsule from a cupy array using ctypes.

    This replicates the capsule format that cvcuda's DLPack consumer expects,
    bypassing cupy's v1.0 __dlpack__ protocol which triggers an abort in
    cvcuda's C++ consumer for certain dtypes.
    """
    import ctypes

    kDLCUDA = 2
    kDLInt, kDLUInt, kDLFloat, kDLComplex = 0, 1, 2, 5

    dtype = cupy_array.dtype
    if dtype.kind == "u":
        code = kDLUInt
    elif dtype.kind == "i":
        code = kDLInt
    elif dtype.kind == "f":
        code = kDLFloat
    elif dtype.kind == "c":
        code = kDLComplex
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

    class _DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class _DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class _DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", _DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", _DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class _DLManagedTensor(ctypes.Structure):
        pass

    _DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
    _DLManagedTensor._fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DELETER),
    ]

    ndim = cupy_array.ndim
    shape_arr = (ctypes.c_int64 * ndim)(*cupy_array.shape)
    strides_list = [1]
    for i in range(ndim - 1, 0, -1):
        strides_list.insert(0, strides_list[0] * cupy_array.shape[i])
    strides_arr = (ctypes.c_int64 * ndim)(*strides_list)

    mt = _DLManagedTensor()
    mt.dl_tensor.data = ctypes.c_void_p(cupy_array.data.ptr)
    mt.dl_tensor.device = _DLDevice(kDLCUDA, cupy_array.device.id)
    mt.dl_tensor.ndim = ndim
    mt.dl_tensor.dtype = _DLDataType(code, dtype.itemsize * 8, 1)
    mt.dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    mt.dl_tensor.strides = ctypes.cast(strides_arr, ctypes.POINTER(ctypes.c_int64))
    mt.dl_tensor.byte_offset = 0
    mt.manager_ctx = None
    mt.deleter = _DELETER(0)

    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    PyCapsule_New.restype = ctypes.py_object

    capsule = PyCapsule_New(ctypes.addressof(mt), b"dltensor", None)

    # Return the capsule and the prevent-GC refs (caller must keep them alive)
    return capsule, (mt, shape_arr, strides_arr, cupy_array)


@t.mark.parametrize("shape,dtype", params_wrap_cuda_buffer)
def test_wrap_cuda_buffer_dlpack(shape, dtype):
    cuda_buffer = cupy.asarray(np.ndarray(shape, dtype=dtype))

    # Create an object with only __dlpack__ (no __cuda_array_interface__)
    # to force cvcuda.as_tensor to use the DLPack path.
    class DLPackObject:
        def __init__(self, src):
            self._src = src
            self._prevent_gc = None

        def __dlpack__(self, *args, **kwargs):
            capsule, refs = _make_dlpack_capsule(self._src)
            self._prevent_gc = refs
            return capsule

        def __dlpack_device__(self):
            return (2, self._src.device.id)  # kDLCUDA

    o = DLPackObject(cuda_buffer)

    tensor = cvcuda.as_tensor(o)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout is None
    assert tensor.ndim == len(shape)


@t.mark.parametrize("shape,dtype", params_wrap_cuda_buffer)
def test_wrap_cuda_buffer_dlpack_v1(shape, dtype):
    """Test consuming DLPack v1.0 capsules from cupy's native __dlpack__."""
    cuda_buffer = cupy.asarray(np.ndarray(shape, dtype=dtype))

    class DLPackV1Object:
        def __init__(self, src):
            self._src = src

        def __dlpack__(self, *args, **kwargs):
            return self._src.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return self._src.__dlpack_device__()

    o = DLPackV1Object(cuda_buffer)

    tensor = cvcuda.as_tensor(o)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout is None
    assert tensor.ndim == len(shape)


@t.mark.parametrize("shape,dtype", params_wrap_cuda_buffer)
def test_wrap_cuda_buffer_cuda_array_interface(shape, dtype):
    cuda_buffer = cupy.asarray(np.ndarray(shape, dtype=dtype))

    # Since cvcuda.as_tensor can understand both dlpack and cuda_array_interface,
    # and we don't know a priori which interfaces it'll use (some CUDA libraries provide both),
    # let's create one object with only the cuda_array_interface.
    class CudaArrayInterfaceObject:
        pass

    o = CudaArrayInterfaceObject()
    o.__cuda_array_interface__ = cuda_buffer.__cuda_array_interface__

    tensor = cvcuda.as_tensor(o)
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
def test_wrap_cuda_buffer_with_layout(shape, dtype, layout):
    tensor = cvcuda.as_tensor(cupy.asarray(np.ndarray(shape, dtype=dtype)), layout)
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
            cvcuda.Format.RGBA8,
            cvcuda.TensorLayout.NHWC,
            (1, 16, 32, 4),
            np.uint8,
        ),
        (
            (38, 7),
            cvcuda.Format.RGB8p,
            cvcuda.TensorLayout.NCHW,
            (1, 3, 7, 38),
            np.uint8,
        ),
    ],
)
def test_tensor_wrap_image_works(size, fmt, gold_layout, gold_shape, gold_dtype):
    img = cvcuda.Image(size, fmt)

    tensor = cvcuda.as_tensor(img)
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
    rng = np.random.default_rng(0)
    hostGold = rng.integers(0, 128, shape, dtype)

    devGold = cupy.asarray(hostGold)

    tensor = cvcuda.as_tensor(devGold)

    devMem = tensor.cuda()
    assert devMem.dtype == dtype
    assert devMem.shape == shape

    devMemWrapped = cupy.asarray(devMem)
    assert (hostGold == devMemWrapped.get()).all()


@t.mark.parametrize(
    "shape,dtype",
    export_cuda_buffer_params,
)
def test_tensor_export_cuda_buffer_dlpack(shape, dtype):
    rng = np.random.default_rng(0)
    hostGold = rng.integers(0, 128, shape, dtype)

    devGold = cupy.asarray(hostGold)

    tensor = cvcuda.as_tensor(devGold)

    devMem = tensor.cuda()
    assert devMem.dtype == dtype
    assert devMem.shape == shape

    # Use from_dlpack to import the DLPack tensor
    devMemWrapped = cupy.from_dlpack(devMem)
    assert (hostGold == devMemWrapped.get()).all()


@t.mark.parametrize(
    "shape,dtype",
    export_cuda_buffer_params,
)
def test_tensor_export_cuda_buffer_dlpack_v0(shape, dtype):
    """Test that cvcuda produces a v0 'dltensor' capsule when max_version is not passed."""
    import ctypes

    rng = np.random.default_rng(0)
    hostGold = rng.integers(0, 128, shape, dtype)

    devGold = cupy.asarray(hostGold)
    tensor = cvcuda.as_tensor(devGold)
    devMem = tensor.cuda()

    # Call __dlpack__ without max_version to get a legacy v0 capsule
    capsule = devMem.__dlpack__()

    # Verify it's a v0 capsule named "dltensor"
    PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
    PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
    PyCapsule_IsValid.restype = ctypes.c_int
    assert PyCapsule_IsValid(capsule, b"dltensor") or PyCapsule_IsValid(
        capsule, b"used_dltensor"
    )


def test_tensor_hold_reference_of_wrapped_buffer():
    cuda_buffer = cupy.asarray(np.ndarray([10], np.int8))
    ptr0 = cuda_buffer.data.ptr

    cvtensor = cvcuda.as_tensor(cuda_buffer)  # noqa: F841 assigned but never used

    del cuda_buffer  # cvtensor must have held cuda_buffer object

    cuda_buffer = cupy.asarray(np.ndarray([10], np.int8))

    # since "cvtensor" must have held the reference to the first "cuda_buffer",
    # the second "cuda_buffer" must be a different buffer
    assert ptr0 != cuda_buffer.data.ptr


def test_tensor_is_kept_alive_by_cuda_array_interface():
    cvcuda.clear_cache()

    tensor1 = cvcuda.Tensor((480, 640, 3), np.uint8)

    iface1 = tensor1.cuda()

    data_buffer1 = iface1.__cuda_array_interface__["data"][0]

    del tensor1

    tensor2 = cvcuda.Tensor((480, 640, 3), np.uint8)
    assert tensor2.cuda().__cuda_array_interface__["data"][0] != data_buffer1

    del tensor2
    # remove tensor2 from cache, but not tensor1, as it's being
    # held by iface
    cvcuda.clear_cache()

    # now tensor1 is free for reuse
    del iface1

    tensor3 = cvcuda.Tensor((480, 640, 3), np.uint8)
    assert tensor3.cuda().__cuda_array_interface__["data"][0] == data_buffer1


def test_tensor_create_packed():
    tensor = cvcuda.Tensor((37, 11, 3), np.uint8, rowalign=1)
    assert tensor.cuda().strides == (11 * 3, 3, 1)


def test_tensor_create_for_imgbatch_packed():
    tensor = cvcuda.Tensor(2, (37, 7), cvcuda.Format.RGB8, rowalign=1)
    assert tensor.cuda().strides == (37 * 7 * 3, 37 * 3, 3, 1)


@t.mark.parametrize(
    "orig_shape, orig_layout, dtype, shape_arg, layout_arg",
    [
        ((1, 23, 65, 3), "NHWC", np.uint8, (23, 65, 3), "HWC"),
        ((5, 23, 65, 3), None, np.int8, (5, 23 * 65, 3), None),
        ((5, 23, 65, 3), None, np.int8, (5, 23 * 65, 3), "ABC"),
        ((1,), "A", np.float32, (1, 1, 1, 1, 1, 1), "ABCDEF"),
    ],
)
def test_tensor_reshape(orig_shape, orig_layout, dtype, shape_arg, layout_arg):
    tensor = cvcuda.Tensor(orig_shape, dtype, layout=orig_layout, rowalign=1)

    def strides(shape):
        out = [0] * len(shape)
        for d in range(len(shape)):
            out[d] = 1
            for d2 in range(d + 1, len(shape)):
                out[d] = out[d] * shape[d2]
        return tuple(out)

    assert tensor.dtype == dtype
    assert tensor.shape == orig_shape
    assert tensor.cuda().strides == strides(orig_shape)

    new_tensors = [
        tensor.reshape(shape_arg, layout=layout_arg),
        cvcuda.reshape(tensor, shape_arg, layout=layout_arg),
    ]
    for new_tensor in new_tensors:
        assert new_tensor.dtype == dtype
        assert new_tensor.shape == shape_arg
        assert new_tensor.cuda().strides == strides(shape_arg)


@t.mark.parametrize(
    "orig_shape, orig_layout, dtype, shape_arg, layout_arg",
    [
        # wrong number of dims in layout
        ((1, 23, 65, 3), "NHWC", np.uint8, (23, 65, 3), "ABCD"),
        # wrong number of dims in layout
        ((1, 23, 65, 3), None, np.uint8, (23, 65, 3), "ABCD"),
        # dims in current layout
        ((5, 23, 65, 3), "NHWC", np.int8, (5, 23 * 65, 3), None),
        # volume mismatch
        ((5, 23, 65, 3), "NHWC", np.int8, (100, 100), "AB"),
        # 0-dim tensors not supported
        ((1,), "A", np.int8, tuple(), ""),
    ],
)
def test_tensor_reshape_error(orig_shape, orig_layout, dtype, shape_arg, layout_arg):
    tensor = cvcuda.Tensor(orig_shape, dtype, layout=orig_layout, rowalign=1)

    with t.raises(RuntimeError):
        tensor.reshape(shape_arg, layout=layout_arg),

    with t.raises(RuntimeError):
        cvcuda.reshape(tensor, shape_arg, layout=layout_arg)


def test_tensor_reshape_lifetime_ref_obj():
    tensor1 = cvcuda.Tensor((20, 10, 3), np.uint8, layout="HWC", rowalign=1)
    tensor2 = tensor1.reshape((200, 3), layout="WC")

    # tensor2 increased the reference count of the underlying handle,
    # so it should be kept alive after tensor1 is deleted
    del tensor1

    assert tensor2.dtype == np.uint8
    assert tensor2.shape == (200, 3)
    assert tensor2.cuda().strides == (3, 1)


@t.mark.parametrize(
    "shape_arg, layout_arg, expected_strides",
    [
        ((1, 10, 10, 3), "XHWC", (320, 32, 3, 1)),
        ((10, 10, 3, 1), "HWCX", (32, 3, 1, 1)),
        ((10, 1, 10, 3), "HXWC", (32, 32, 3, 1)),
        ((10, 2, 5, 3), "HABC", (32, 15, 3, 1)),
        ((2, 5, 10, 3), "ABWC", (160, 32, 3, 1)),
    ],
)
def test_tensor_reshape_strided(shape_arg, layout_arg, expected_strides):
    tensor = cvcuda.Tensor((10, 10, 3), np.uint8, layout="HWC")
    assert tensor.cuda().strides == (32, 3, 1)  # strided rows

    new_tensors = [
        tensor.reshape(shape_arg, layout=layout_arg),
        cvcuda.reshape(tensor, shape_arg, layout=layout_arg),
    ]
    for new_tensor in new_tensors:
        assert new_tensor.cuda().strides == expected_strides


@t.mark.parametrize(
    "shape_arg, layout_arg",
    [((300,), "A")],
)
def test_tensor_reshape_strided_error(shape_arg, layout_arg):
    tensor = cvcuda.Tensor((10, 10, 3), np.uint8, layout="HWC")
    assert tensor.cuda().strides == (32, 3, 1)  # strided rows

    with t.raises(RuntimeError):
        tensor.reshape(shape_arg, layout=layout_arg)

    with t.raises(RuntimeError):
        cvcuda.reshape(tensor, shape_arg, layout=layout_arg)


@t.mark.parametrize(
    "shape_arg, dtype_arg, layout_arg",
    [
        ((3, 5, 7), np.dtype("2f4"), "NHW"),
        ((3, 5, 3), np.dtype("4f8"), "NHW"),
        ((3, 5, 2), np.dtype("2i1"), "NHW"),
    ],
)
def test_tensor_wrap_cuda_array_interface(shape_arg, dtype_arg, layout_arg):
    tensor = cvcuda.Tensor(shape_arg, dtype_arg, layout_arg)

    tcuda = tensor.cuda()
    cai = tcuda.__cuda_array_interface__
    assert cai["typestr"] == dtype_arg.str
    assert cai["shape"] == shape_arg

    wrapped = cvcuda.as_tensor(tcuda, layout_arg)

    assert wrapped.shape == shape_arg
    assert wrapped.dtype == dtype_arg
    assert wrapped.layout == layout_arg


def test_tensor_size_in_bytes():
    """
    Checks if the computation of the Tensor size in bytes is correct
    """
    tensor_create_for_image_batch = cvcuda.Tensor(
        2, (37, 7), cvcuda.Format.RGB8, rowalign=1
    )
    assert cvcuda.internal.nbytes_in_cache(tensor_create_for_image_batch) > 0

    tensor_create = cvcuda.Tensor((5, 16, 32, 4), np.float32, cvcuda.TensorLayout.NHWC)
    assert cvcuda.internal.nbytes_in_cache(tensor_create) > 0

    tensor_wrap = cvcuda.as_tensor(
        cupy.asarray(np.ndarray((5, 16, 32, 4), dtype=np.float32))
    )
    assert cvcuda.internal.nbytes_in_cache(tensor_wrap) == 0

    img = cvcuda.Image((32, 16), cvcuda.Format.RGBA8)
    tensor_wrap_image = cvcuda.as_tensor(img)
    assert cvcuda.internal.nbytes_in_cache(tensor_wrap_image) == 0

    tensor_reshape = cvcuda.reshape(
        tensor_create, (5, 32, 16, 4), cvcuda.TensorLayout.NHWC
    )
    assert cvcuda.internal.nbytes_in_cache(tensor_reshape) == 0
