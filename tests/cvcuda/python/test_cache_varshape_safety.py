# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety boundaries for reusing variable-shape cache entries."""

import ctypes
import gc
import sysconfig

import cupy
import numpy as np
import pytest

import cvcuda
from test_cache_native_alias import (
    _cache_room,
    _CAPIThroughTensorGetHandle,
    _load_nvcv_types,
)
from test_resource_lifetime import _park_gpu, _run_repro


class _CAPIThroughImageGetHandle(_CAPIThroughTensorGetHandle):
    _fields_ = [
        ("tensor_create", ctypes.c_void_p),
        ("tensor_create_for_image_batch", ctypes.c_void_p),
        ("array_get_handle", ctypes.c_void_p),
        ("array_create", ctypes.c_void_p),
        ("image_batch_create", ctypes.c_void_p),
        ("image_batch_get_handle", ctypes.c_void_p),
        ("image_batch_push_back", ctypes.c_void_p),
        ("image_batch_pop_back", ctypes.c_void_p),
        ("image_batch_clear", ctypes.c_void_p),
        ("cache_add", ctypes.c_void_p),
        ("cache_fetch", ctypes.c_void_p),
        ("image_create", ctypes.c_void_p),
        ("image_get_handle", ctypes.c_void_p),
    ]


def _capi_table():
    capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    capsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule_get_pointer.restype = ctypes.c_void_p
    capi = capsule_get_pointer(cvcuda._C_API, b"cvcuda._C_API")
    assert capi
    return ctypes.cast(capi, ctypes.POINTER(_CAPIThroughImageGetHandle)).contents


def _capi_handle(field, obj):
    function_pointer = getattr(_capi_table(), field)
    assert function_pointer
    handle = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)(function_pointer)(obj)
    assert handle
    return handle


class _NativeImageAlias:
    def __init__(self, image):
        exported = image.cuda()
        interface = exported.__cuda_array_interface__
        self._shape = tuple(interface["shape"])
        self._dtype = np.dtype(interface["typestr"])
        self._strides = interface.get("strides")
        if self._strides is None:
            stride = self._dtype.itemsize
            contiguous = []
            for extent in reversed(self._shape):
                contiguous.append(stride)
                stride *= extent
            self._strides = tuple(reversed(contiguous))
        self._base_ptr = interface["data"][0]
        assert self._base_ptr

        self._library = _load_nvcv_types()
        self._library.nvcvImageIncRef.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.nvcvImageIncRef.restype = ctypes.c_int
        self._library.nvcvImageDecRef.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.nvcvImageDecRef.restype = ctypes.c_int
        self._handle = _capi_handle("image_get_handle", image)
        refcount = ctypes.c_int()
        assert self._library.nvcvImageIncRef(self._handle, ctypes.byref(refcount)) == 0
        assert refcount.value >= 2
        del exported

    def as_cupy(self):
        size = self._dtype.itemsize + sum(
            (extent - 1) * stride for extent, stride in zip(self._shape, self._strides)
        )
        memory = cupy.cuda.UnownedMemory(self._base_ptr, size, self)
        memory_pointer = cupy.cuda.MemoryPointer(memory, 0)
        return cupy.ndarray(
            self._shape,
            dtype=self._dtype,
            memptr=memory_pointer,
            strides=self._strides,
        )

    def close(self):
        assert self._handle
        assert self._library.nvcvImageDecRef(self._handle, None) == 0
        self._handle = None


class _NativeImageBatchAlias:
    def __init__(self, batch):
        self._library = _load_nvcv_types()
        self._library.nvcvImageBatchIncRef.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.nvcvImageBatchIncRef.restype = ctypes.c_int
        self._library.nvcvImageBatchDecRef.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.nvcvImageBatchDecRef.restype = ctypes.c_int
        self._library.nvcvImageBatchVarShapeGetImages.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
        ]
        self._library.nvcvImageBatchVarShapeGetImages.restype = ctypes.c_int
        self._library.nvcvImageDecRef.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.nvcvImageDecRef.restype = ctypes.c_int

        self._handle = _capi_handle("image_batch_get_handle", batch)
        self._num_images = len(batch)
        refcount = ctypes.c_int()
        assert (
            self._library.nvcvImageBatchIncRef(self._handle, ctypes.byref(refcount))
            == 0
        )
        assert refcount.value >= 2
        self.original_image_handles = self.image_handles()

    def image_handles(self):
        handles = (ctypes.c_void_p * self._num_images)()
        assert (
            self._library.nvcvImageBatchVarShapeGetImages(
                self._handle, 0, handles, self._num_images
            )
            == 0
        )
        values = tuple(handle for handle in handles)
        for handle in handles:
            assert self._library.nvcvImageDecRef(handle, None) == 0
        return values

    def close(self):
        assert self._handle
        assert self._library.nvcvImageBatchDecRef(self._handle, None) == 0
        self._handle = None


@pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="Free-threaded argument wrappers still retain consumed intermediates",
)
def test_pending_varshape_reuse_respects_alias_descriptors_and_streams():
    _run_repro(
        """
        import gc
        import numpy as np

        sizes = [(40, 32), (48, 36)]
        batch_size = len(sizes)

        def reset_cache():
            gc.collect()
            cvcuda.internal.drain_pending_releases()
            cvcuda.clear_cache()

        def set_room_for_misses():
            baseline = cvcuda.current_cache_size_inbytes()
            assert baseline > 0
            cvcuda.set_cache_limit_inbytes(baseline * 32)

        def make_codes():
            return cvcuda.as_tensor(cupy.zeros((batch_size, 1), dtype=cupy.int32), "NC")

        def make_external_batch(image_sizes, fmt, channels, seed):
            result = cvcuda.ImageBatchVarShape(batch_size)
            expected = []
            for index, (width, height) in enumerate(image_sizes):
                color = (np.arange(channels) + seed + 11 * index).astype(np.uint8)
                values = np.broadcast_to(color, (height, width, channels)).copy()
                result.pushback(cvcuda.as_image(cupy.asarray(values), fmt))
                expected.append(values)
            return result, expected

        def make_native_batch(image_sizes, fmt):
            result = cvcuda.ImageBatchVarShape(batch_size)
            for image_size in image_sizes:
                result.pushback(cvcuda.Image(image_size, fmt))
            return result

        def check_batch(result, image_sizes, fmt, expected):
            images = list(result)
            assert len(images) == len(image_sizes)
            for image, image_size, values in zip(images, image_sizes, expected):
                assert image.size == image_size
                assert image.format == fmt
                np.testing.assert_array_equal(cupy.asnumpy(cupy.asarray(image.cuda())), values)

        def first_child(result):
            iterator = iter(result)
            child = next(iterator)
            del iterator
            return child

        def make_alias(kind, image):
            if kind == "child":
                return image
            if kind == "tensor":
                return cvcuda.as_tensor(image)
            assert kind == "export"
            return image.cuda()

        def check_alias(kind, alias, expected):
            values = cupy.asarray(alias if kind == "export" else alias.cuda())
            if values.ndim == 4:
                values = values[0]
            np.testing.assert_array_equal(cupy.asnumpy(values), expected)

        # Compile the finite park once; all negative cases share this process.
        compile_stream = cvcuda.Stream()
        _park_gpu(compile_stream, 0)
        compile_stream.sync()
        del compile_stream

        def run_alias_case(kind, pending):
            reset_cache()
            stream = cvcuda.Stream()
            codes = make_codes()
            first_source, first_values = make_external_batch(sizes, cvcuda.Format.RGB8, 3, 17)
            second_source, second_values = make_external_batch(sizes, cvcuda.Format.RGB8, 3, 91)
            first_prime = make_native_batch(sizes, cvcuda.Format.RGB8)
            second_prime = make_native_batch(sizes, cvcuda.Format.RGB8)
            cupy.cuda.get_current_stream().synchronize()
            cvcuda.flip_into(first_prime, first_source, codes, stream=stream)
            cvcuda.flip_into(second_prime, second_source, codes, stream=stream)
            warmed = cvcuda.flip(first_source, codes, stream=stream)
            stream.sync()
            cvcuda.internal.drain_pending_releases()
            warmed_id = warmed.id

            if pending:
                del warmed
                gc.collect()
                set_room_for_misses()
                _park_gpu(stream, 300_000_000)
                first = cvcuda.flip(first_source, codes, stream=stream)
                assert first.id == warmed_id
            else:
                first = warmed
                del warmed
                set_room_for_misses()

            first_id = first.id
            child = first_child(first)
            child_id = child.id
            alias = make_alias(kind, child)
            del child, first
            gc.collect()
            if pending:
                assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0

            second = cvcuda.flip(second_source, codes, stream=stream)
            second_child_ids = {image.id for image in second}
            assert child_id not in second_child_ids, (kind, pending, child_id, second_child_ids)
            if pending:
                assert second.id != first_id, "an in-flight batch descriptor was cleared and recycled"
            else:
                assert second.id == first_id, "ready batch descriptor reuse was unnecessarily lost"

            stream.sync()
            check_alias(kind, alias, first_values[0])
            check_batch(second, sizes, cvcuda.Format.RGB8, second_values)
            del alias, second
            gc.collect()
            cvcuda.internal.drain_pending_releases()

        for alias_kind in ("child", "tensor", "export"):
            run_alias_case(alias_kind, pending=False)
            run_alias_case(alias_kind, pending=True)

        def run_generic_constructor_case():
            reset_cache()
            stream = cvcuda.Stream()
            codes = make_codes()
            source, expected = make_external_batch(sizes, cvcuda.Format.RGB8, 3, 29)
            prime = make_native_batch(sizes, cvcuda.Format.RGB8)
            snapshot = make_native_batch(sizes, cvcuda.Format.RGB8)
            cupy.cuda.get_current_stream().synchronize()
            cvcuda.flip_into(prime, source, codes, stream=stream)
            cvcuda.flip_into(snapshot, source, codes, stream=stream)
            warmed = cvcuda.flip(source, codes, stream=stream)
            stream.sync()
            cvcuda.internal.drain_pending_releases()
            warmed_id = warmed.id
            del warmed
            gc.collect()
            set_room_for_misses()

            _park_gpu(stream, 300_000_000)
            pending_output = cvcuda.flip(source, codes, stream=stream)
            assert pending_output.id == warmed_id
            cvcuda.flip_into(snapshot, pending_output, codes, stream=stream)
            pending_id = pending_output.id
            del pending_output
            gc.collect()
            assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0

            empty = cvcuda.ImageBatchVarShape(batch_size)
            assert empty.id != pending_id
            assert len(empty) == 0
            stream.sync()
            check_batch(snapshot, sizes, cvcuda.Format.RGB8, expected)
            cvcuda.internal.drain_pending_releases()

        run_generic_constructor_case()

        def run_descriptor_case(case):
            reset_cache()
            stream = cvcuda.Stream()
            codes = make_codes()
            first_sizes = sizes
            first_format = cvcuda.Format.RGB8
            first_channels = 3
            if case == "size_order":
                second_sizes = list(reversed(sizes))
                second_format = cvcuda.Format.RGB8
                second_channels = 3
            else:
                assert case == "format"
                second_sizes = sizes
                second_format = cvcuda.Format.RGBA8
                second_channels = 4

            first_source, first_values = make_external_batch(
                first_sizes, first_format, first_channels, 33
            )
            second_source, second_values = make_external_batch(
                second_sizes, second_format, second_channels, 101
            )
            first_prime = make_native_batch(first_sizes, first_format)
            second_prime = make_native_batch(second_sizes, second_format)
            first_snapshot = make_native_batch(first_sizes, first_format)
            cupy.cuda.get_current_stream().synchronize()
            cvcuda.flip_into(first_prime, first_source, codes, stream=stream)
            cvcuda.flip_into(second_prime, second_source, codes, stream=stream)
            cvcuda.flip_into(first_snapshot, first_source, codes, stream=stream)
            warmed = cvcuda.flip(first_source, codes, stream=stream)
            stream.sync()
            cvcuda.internal.drain_pending_releases()
            warmed_id = warmed.id
            del warmed
            gc.collect()
            set_room_for_misses()

            _park_gpu(stream, 300_000_000)
            first = cvcuda.flip(first_source, codes, stream=stream)
            assert first.id == warmed_id
            cvcuda.flip_into(first_snapshot, first, codes, stream=stream)
            first_id = first.id
            del first
            gc.collect()
            assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0

            second = cvcuda.flip(second_source, codes, stream=stream)
            assert second.id != first_id, f"pending descriptor mismatch reused for {case}"
            stream.sync()
            check_batch(first_snapshot, first_sizes, first_format, first_values)
            check_batch(second, second_sizes, second_format, second_values)
            cvcuda.internal.drain_pending_releases()

        run_descriptor_case("size_order")
        run_descriptor_case("format")

        def run_independent_stream_case():
            reset_cache()
            producer = cvcuda.Stream()
            consumer = cvcuda.Stream()
            assert int(producer.handle) != int(consumer.handle)
            producer_codes = make_codes()
            consumer_codes = make_codes()
            first_source, first_values = make_external_batch(sizes, cvcuda.Format.RGB8, 3, 41)
            second_source, second_values = make_external_batch(sizes, cvcuda.Format.RGB8, 3, 113)
            producer_prime = make_native_batch(sizes, cvcuda.Format.RGB8)
            consumer_prime = make_native_batch(sizes, cvcuda.Format.RGB8)
            first_snapshot = make_native_batch(sizes, cvcuda.Format.RGB8)
            cupy.cuda.get_current_stream().synchronize()
            cvcuda.flip_into(producer_prime, first_source, producer_codes, stream=producer)
            cvcuda.flip_into(consumer_prime, second_source, consumer_codes, stream=consumer)
            cvcuda.flip_into(first_snapshot, first_source, producer_codes, stream=producer)
            warmed = cvcuda.flip(first_source, producer_codes, stream=producer)
            producer.sync()
            consumer.sync()
            cvcuda.internal.drain_pending_releases()
            warmed_id = warmed.id
            del warmed
            gc.collect()
            set_room_for_misses()

            _park_gpu(producer, 300_000_000)
            first = cvcuda.flip(first_source, producer_codes, stream=producer)
            assert first.id == warmed_id
            cvcuda.flip_into(first_snapshot, first, producer_codes, stream=producer)
            first_id = first.id
            del first
            gc.collect()
            assert cupy.cuda.runtime.streamQuery(int(producer.handle)) != 0

            second = cvcuda.flip(second_source, consumer_codes, stream=consumer)
            assert second.id != first_id, "below-limit reuse crossed independent streams"
            consumer.sync()
            producer.sync()
            check_batch(first_snapshot, sizes, cvcuda.Format.RGB8, first_values)
            check_batch(second, sizes, cvcuda.Format.RGB8, second_values)
            cvcuda.internal.drain_pending_releases()

        run_independent_stream_case()
        print("PASS", flush=True)
        """,
        timeout=45,
        cuda_prelude=True,
    )


def _native_batch(sizes):
    batch = cvcuda.ImageBatchVarShape(len(sizes))
    for size in sizes:
        batch.pushback(cvcuda.Image(size, cvcuda.Format.RGB8))
    return batch


def _external_batch(sizes, seed):
    batch = cvcuda.ImageBatchVarShape(len(sizes))
    expected = []
    for index, (width, height) in enumerate(sizes):
        color = (np.arange(3) + seed + 11 * index).astype(np.uint8)
        values = np.broadcast_to(color, (height, width, 3)).copy()
        batch.pushback(cvcuda.as_image(cupy.asarray(values), cvcuda.Format.RGB8))
        expected.append(values)
    return batch, expected


def _check_native_batch(batch, sizes, expected):
    images = list(batch)
    assert len(images) == len(sizes)
    for image, size, values in zip(images, sizes, expected):
        assert image.size == size
        np.testing.assert_array_equal(cupy.asnumpy(cupy.asarray(image.cuda())), values)


@pytest.mark.usefixtures(_cache_room.__name__)
@pytest.mark.parametrize("alias_kind", ["image", "batch"])
@pytest.mark.parametrize("pending", [False, True], ids=["ready", "pending"])
def test_varshape_cache_rejects_native_aliases(alias_kind, pending):
    cvcuda.clear_cache()
    stream = cvcuda.Stream()
    sizes = [(40, 32), (48, 36)]
    codes = cvcuda.as_tensor(cupy.zeros((len(sizes), 1), dtype=cupy.int32), "NC")
    first_source, first_values = _external_batch(sizes, 17)
    second_source, second_values = _external_batch(sizes, 91)
    first_prime = _native_batch(sizes)
    second_prime = _native_batch(sizes)
    snapshot = _native_batch(sizes)
    alias = None

    try:
        cupy.cuda.get_current_stream().synchronize()
        cvcuda.flip_into(first_prime, first_source, codes, stream=stream)
        cvcuda.flip_into(second_prime, second_source, codes, stream=stream)
        cvcuda.flip_into(snapshot, first_source, codes, stream=stream)
        warmed = cvcuda.flip(first_source, codes, stream=stream)
        stream.sync()
        cvcuda.internal.drain_pending_releases()
        warmed_id = warmed.id

        if pending:
            del warmed
            gc.collect()
            baseline = cvcuda.current_cache_size_inbytes()
            assert baseline > 0
            cvcuda.set_cache_limit_inbytes(baseline * 32)
            _park_gpu(stream, 1_000_000_000)
            first = cvcuda.flip(first_source, codes, stream=stream)
            assert first.id == warmed_id
            cvcuda.flip_into(snapshot, first, codes, stream=stream)
        else:
            first = warmed
            del warmed
            baseline = cvcuda.current_cache_size_inbytes()
            assert baseline > 0
            cvcuda.set_cache_limit_inbytes(baseline * 32)

        first_id = first.id
        if alias_kind == "image":
            iterator = iter(first)
            child = next(iterator)
            del iterator
            child_id = child.id
            alias = _NativeImageAlias(child)
            del child
        else:
            alias = _NativeImageBatchAlias(first)

        del first
        gc.collect()
        if pending:
            assert cupy.cuda.runtime.streamQuery(int(stream.handle)) != 0

        second = cvcuda.flip(second_source, codes, stream=stream)
        if alias_kind == "image":
            second_child_ids = {image.id for image in second}
            assert child_id not in second_child_ids
        else:
            assert second.id != first_id

        stream.sync()
        _check_native_batch(snapshot, sizes, first_values)
        _check_native_batch(second, sizes, second_values)
        if alias_kind == "image":
            np.testing.assert_array_equal(alias.as_cupy().get(), first_values[0])
        else:
            assert alias.image_handles() == alias.original_image_handles
    finally:
        try:
            stream.sync()
        finally:
            if alias is not None:
                alias.close()
            gc.collect()
            cvcuda.internal.drain_pending_releases()
