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

import ctypes
import gc
import os

import cupy
import numpy as np
import pytest

import cvcuda
from test_resource_lifetime import _DELAY_KERNEL, _external_stream


_NVCV_TENSOR_MAX_RANK = 15
_NVCV_TENSOR_BUFFER_STRIDED_CUDA = 1


@pytest.fixture
def _cache_room():
    previous_limit = cvcuda.get_cache_limit_inbytes()
    try:
        cvcuda.set_cache_limit_inbytes(cupy.cuda.Device().mem_info[1] // 2)
        yield
    finally:
        cvcuda.set_cache_limit_inbytes(previous_limit)


class _NVCVTensorLayout(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_char * (_NVCV_TENSOR_MAX_RANK + 1)),
        ("rank", ctypes.c_int32),
    ]


class _NVCVTensorBufferStrided(ctypes.Structure):
    _fields_ = [
        ("strides", ctypes.c_int64 * _NVCV_TENSOR_MAX_RANK),
        ("base_ptr", ctypes.c_void_p),
    ]


class _NVCVTensorBuffer(ctypes.Union):
    _fields_ = [("strided", _NVCVTensorBufferStrided)]


class _NVCVTensorData(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_uint64),
        ("layout", _NVCVTensorLayout),
        ("rank", ctypes.c_int32),
        ("shape", ctypes.c_int64 * _NVCV_TENSOR_MAX_RANK),
        ("buffer_type", ctypes.c_int),
        ("buffer", _NVCVTensorBuffer),
    ]


class _CAPIThroughTensorGetHandle(ctypes.Structure):
    _fields_ = [
        ("datatype_to_python", ctypes.c_void_p),
        ("datatype_from_python", ctypes.c_void_p),
        ("imageformat_to_python", ctypes.c_void_p),
        ("imageformat_from_python", ctypes.c_void_p),
        ("resource_submit_sync", ctypes.c_void_p),
        ("stream_hold_resources", ctypes.c_void_p),
        ("stream_get_current", ctypes.c_void_p),
        ("stream_get_cuda_handle", ctypes.c_void_p),
        ("tensor_get_handle", ctypes.c_void_p),
    ]


def _tensor_get_handle(tensor):
    capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    capsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule_get_pointer.restype = ctypes.c_void_p
    capi = capsule_get_pointer(cvcuda._C_API, b"cvcuda._C_API")
    assert capi
    function_pointer = ctypes.cast(
        capi, ctypes.POINTER(_CAPIThroughTensorGetHandle)
    ).contents.tensor_get_handle
    assert function_pointer
    return ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)(function_pointer)(
        tensor
    )


def _load_nvcv_types():
    try:
        return ctypes.CDLL("libnvcv_types.so.0")
    except OSError as soname_error:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            loaded_paths = {
                line.rstrip().rsplit(" ", 1)[-1]
                for line in maps
                if os.path.basename(line.rstrip().rsplit(" ", 1)[-1]).startswith(
                    "libnvcv_types"
                )
            }
        assert len(loaded_paths) == 1, (
            "could not uniquely locate the loaded NVCV types library after "
            f"the public soname failed to load: {soname_error}; candidates: "
            f"{sorted(loaded_paths)}"
        )
        return ctypes.CDLL(loaded_paths.pop())


@pytest.mark.parametrize("switch_device", [False, True])
def test_python_tensor_allocator_does_not_alias_its_backing_storage(switch_device):
    original_device = cupy.cuda.runtime.getDevice()
    if switch_device and cupy.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("allocator device-affinity regression requires two GPUs")
    calling_device = (
        (original_device + 1) % cupy.cuda.runtime.getDeviceCount()
        if switch_device
        else original_device
    )
    tensor = cvcuda.Tensor((1024,), cvcuda.Type.U8, "W", rowalign=256)
    tensor_ptr = tensor.cuda().__cuda_array_interface__["data"][0]
    library = _load_nvcv_types()

    library.nvcvTensorGetAllocator.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.nvcvTensorGetAllocator.restype = ctypes.c_int
    library.nvcvAllocatorAllocCudaMemory.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int64,
        ctypes.c_int32,
    ]
    library.nvcvAllocatorAllocCudaMemory.restype = ctypes.c_int
    library.nvcvAllocatorFreeCudaMemory.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int32,
    ]
    library.nvcvAllocatorFreeCudaMemory.restype = ctypes.c_int
    library.nvcvAllocatorDecRef.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.nvcvAllocatorDecRef.restype = ctypes.c_int

    allocator = ctypes.c_void_p()
    assert (
        library.nvcvTensorGetAllocator(
            _tensor_get_handle(tensor), ctypes.byref(allocator)
        )
        == 0
    )
    assert allocator.value
    del tensor
    gc.collect()
    cvcuda.clear_cache()

    allocations = []
    try:
        for _ in range(2):
            allocation = ctypes.c_void_p()
            with cupy.cuda.Device(calling_device):
                assert (
                    library.nvcvAllocatorAllocCudaMemory(
                        allocator, ctypes.byref(allocation), 256, 256
                    )
                    == 0
                )
                assert allocation.value
                allocations.append(allocation.value)
                assert cupy.cuda.runtime.getDevice() == calling_device
                assert (
                    cupy.cuda.runtime.pointerGetAttributes(allocation.value).device
                    == original_device
                )

        assert len({tensor_ptr, *allocations}) == 3
    finally:
        for allocation in allocations:
            with cupy.cuda.Device(calling_device):
                assert (
                    library.nvcvAllocatorFreeCudaMemory(
                        allocator, ctypes.c_void_p(allocation), 256, 256
                    )
                    == 0
                )
                assert cupy.cuda.runtime.getDevice() == calling_device
        with cupy.cuda.Device(calling_device):
            assert library.nvcvAllocatorDecRef(allocator, None) == 0
            assert cupy.cuda.runtime.getDevice() == calling_device


class _NativeTensorAlias:
    def __init__(self, tensor):
        self._library = _load_nvcv_types()
        self._library.nvcvTensorIncRef.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.nvcvTensorIncRef.restype = ctypes.c_int
        self._library.nvcvTensorDecRef.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.nvcvTensorDecRef.restype = ctypes.c_int
        self._library.nvcvTensorExportData.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NVCVTensorData),
        ]
        self._library.nvcvTensorExportData.restype = ctypes.c_int

        self._handle = _tensor_get_handle(tensor)
        assert self._handle
        refcount = ctypes.c_int()
        assert self._library.nvcvTensorIncRef(self._handle, ctypes.byref(refcount)) == 0
        assert refcount.value >= 2

        try:
            data = _NVCVTensorData()
            assert (
                self._library.nvcvTensorExportData(self._handle, ctypes.byref(data))
                == 0
            )
            assert data.buffer_type == _NVCV_TENSOR_BUFFER_STRIDED_CUDA
            self._shape = tuple(data.shape[: data.rank])
            self._strides = tuple(data.buffer.strided.strides[: data.rank])
            self._base_ptr = data.buffer.strided.base_ptr
            assert self._base_ptr
        except BaseException:
            self._library.nvcvTensorDecRef(self._handle, None)
            self._handle = None
            raise

    def as_cupy(self):
        itemsize = np.dtype(np.uint8).itemsize
        size = itemsize + sum(
            (extent - 1) * stride for extent, stride in zip(self._shape, self._strides)
        )
        memory = cupy.cuda.UnownedMemory(self._base_ptr, size, self)
        memory_pointer = cupy.cuda.MemoryPointer(memory, 0)
        return cupy.ndarray(
            self._shape,
            dtype=cupy.uint8,
            memptr=memory_pointer,
            strides=self._strides,
        )

    def close(self):
        assert self._handle
        assert self._library.nvcvTensorDecRef(self._handle, None) == 0
        self._handle = None


@pytest.mark.usefixtures("_cache_room")
def test_cache_does_not_reuse_tensor_retained_through_native_handle():
    cvcuda.clear_cache()
    stream = cvcuda.Stream()
    first_values = cupy.arange(64 * 64 * 3, dtype=cupy.uint8).reshape(1, 64, 64, 3)
    second_values = cupy.arange(32 * 32 * 3, dtype=cupy.uint8).reshape(1, 32, 32, 3)
    first_src = cvcuda.as_tensor(first_values, "NHWC")
    second_src = cvcuda.as_tensor(second_values, "NHWC")

    first = cvcuda.flip(first_src, 0, stream=stream)
    stream.sync()
    cvcuda.internal.drain_pending_releases()
    first_id = first.id
    alias = _NativeTensorAlias(first)
    del first
    gc.collect()

    try:
        second = cvcuda.flip(second_src, 0, stream=stream)
        stream.sync()
        np.testing.assert_array_equal(
            alias.as_cupy().get(), cupy.flip(first_values, axis=1).get()
        )
        assert (
            second.id != first_id
        ), "cache reused an allocation retained by native code"
    finally:
        alias.close()
    del alias
    gc.collect()
    third = cvcuda.flip(first_src, 0, stream=stream)
    assert (
        third.id == first_id
    ), "released native alias left the allocation non-reusable"
    stream.sync()


@pytest.mark.usefixtures("_cache_room")
@pytest.mark.parametrize("pending", [False, True], ids=["ready", "pending"])
def test_pending_pressure_does_not_reuse_tensor_retained_through_native_handle(pending):
    cvcuda.clear_cache()
    producer = cvcuda.Stream()
    consumer = cvcuda.Stream()
    first_values = cupy.arange(256 * 256 * 3, dtype=cupy.uint8).reshape(1, 256, 256, 3)
    second_values = 255 - first_values
    cupy.cuda.get_current_stream().synchronize()
    first_src = cvcuda.as_tensor(first_values, "NHWC")
    second_src = cvcuda.as_tensor(second_values, "NHWC")

    alias = None
    try:
        with _external_stream(int(producer.handle)):
            _DELAY_KERNEL((1,), (1,), (1,))
        producer.sync()

        warmed = cvcuda.flip(first_src, 0, stream=producer)
        producer.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id
        del warmed
        gc.collect()
        limit = cvcuda.current_cache_size_inbytes()
        assert limit > 0
        cvcuda.set_cache_limit_inbytes(limit)

        if pending:
            with _external_stream(int(producer.handle)):
                _DELAY_KERNEL((1,), (1,), (2_000_000_000,))

        first = cvcuda.flip(first_src, 0, stream=producer)
        assert first.id == warmed_id
        alias = _NativeTensorAlias(first)
        del first
        gc.collect()

        if pending:
            assert cupy.cuda.runtime.streamQuery(int(producer.handle)) != 0
        else:
            # Complete GPU work without retiring its host release record.
            cupy.cuda.runtime.streamSynchronize(int(producer.handle))
        assert cvcuda.internal._pending_release_batch_count() > 0
        second = cvcuda.flip(second_src, 0, stream=consumer)
        producer.sync()
        consumer.sync()
        assert (
            second.id != warmed_id
        ), "pending-pressure reuse aliased a live native Tensor handle"
        np.testing.assert_array_equal(
            alias.as_cupy().get(), cupy.flip(first_values, axis=1).get()
        )
        np.testing.assert_array_equal(
            cupy.asarray(second.cuda()).get(), cupy.flip(second_values, axis=1).get()
        )
        alias.close()
        alias = None
        gc.collect()
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() <= limit
        third = cvcuda.flip(first_src, 0, stream=producer)
        producer.sync()
        # Pressure may evict the original cache entry while its native alias
        # keeps the allocation alive; subsequent output need not reuse its ID.
        np.testing.assert_array_equal(
            cupy.asarray(third.cuda()).get(), cupy.flip(first_values, axis=1).get()
        )
        cvcuda.internal.drain_pending_releases()
        assert cvcuda.current_cache_size_inbytes() <= limit
    finally:
        try:
            producer.sync()
            consumer.sync()
        finally:
            try:
                if alias is not None:
                    alias.close()
            finally:
                cvcuda.internal.drain_pending_releases()
