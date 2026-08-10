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
import cvcuda_util as util
import cupy

import cvcuda_types as cv_types


def test_image_is_cached():
    created_ids = set()

    pt_img = cupy.asarray(np.random.rand(1, 1).astype(np.float32))
    img = cvcuda.as_image(pt_img)
    created_ids.add(img.id)
    del img  # delete img such that only cache has a reference to it

    for i in range(50):
        pt_img = cupy.asarray(np.random.rand(1 + i, i + 2).astype(np.float32))
        img = cvcuda.as_image(pt_img)
        assert img.id in created_ids
        del img


def test_failed_image_rebind_preserves_stream_state():
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """\
        import gc

        import cupy
        import cvcuda


        class DLPackOnly:
            def __init__(self, array):
                self.array = array

            def __dlpack_device__(self):
                return self.array.__dlpack_device__()

            def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
                kwargs = {}
                if stream is not None:
                    kwargs["stream"] = stream
                if max_version is not None:
                    kwargs["max_version"] = max_version
                if dl_device is not None:
                    kwargs["dl_device"] = dl_device
                if copy is not None:
                    kwargs["copy"] = copy
                return self.array.__dlpack__(**kwargs)


        cvcuda.clear_cache()
        source = cupy.zeros((6, 8), dtype=cupy.uint8)
        image = cvcuda.as_image(source)

        producer = cupy.cuda.Stream(non_blocking=True)
        stream = cvcuda.as_stream(producer)
        handle = stream.handle
        image.submitStreamSync(stream)

        # The cached Image's Resource is now the sole owner keeping the external
        # producer stream alive.
        del stream, producer, image
        gc.collect()
        assert cupy.cuda.runtime.streamQuery(handle) == 0

        managed_mem = cupy.cuda.malloc_managed(3 * 4 * 2)
        managed_plane = cupy.ndarray((3, 4, 2), dtype=cupy.uint8, memptr=managed_mem)

        try:
            try:
                cvcuda.as_image([DLPackOnly(source), DLPackOnly(managed_plane)])
            except RuntimeError as exc:
                assert "All buffers must belong to the same device" in str(exc)
            else:
                raise AssertionError("mixed-device image planes unexpectedly accepted")

            # Failed validation must leave the cached wrapper's stream ownership
            # unchanged. A premature reset destroys the otherwise-unreferenced stream.
            assert cupy.cuda.runtime.streamQuery(handle) == 0
        finally:
            cvcuda.clear_cache()
            gc.collect()

        print("PASS")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, (
        f"rebind subprocess exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("PASS")


def test_image_creation_works():
    img = cvcuda.Image((7, 5), cvcuda.Format.NV12)
    assert img.width == 7
    assert img.height == 5
    assert img.size == (7, 5)
    assert img.format == cvcuda.Format.NV12


def test_image_creation_arg_keywords():
    img = cvcuda.Image(size=(7, 5), format=cvcuda.Format.NV12)
    assert img.width == 7
    assert img.height == 5
    assert img.size == (7, 5)
    assert img.format == cvcuda.Format.NV12


buffmt_common = [
    # packed formats
    ([5, 7, 1], np.uint8, cvcuda.Format.U8),
    ([5, 7, 1], np.uint8, cvcuda.Format.U8),
    ([5, 7, 1], np.uint8, cvcuda.Format.U8),
    ([5, 7], np.uint8, cvcuda.Format.U8),
    ([5, 7, 1], np.int8, cvcuda.Format.S8),
    ([5, 7, 1], np.int16, cvcuda.Format.S16),
    ([5, 7, 1], np.float16, cvcuda.Format.F16),
    ([5, 7, 2], np.int16, cvcuda.Format._2S16),
    ([5, 7, 2], np.float16, cvcuda.Format._2F16),
    ([5, 7, 1], np.float32, cvcuda.Format.F32),
    ([5, 7, 1], np.float64, cvcuda.Format.F64),
    ([5, 7, 2], np.float32, cvcuda.Format._2F32),
    ([5, 7, 3], np.uint8, cvcuda.Format.RGB8),
    ([5, 7, 4], np.uint8, cvcuda.Format.RGBA8),
    ([1, 5, 7], np.uint8, cvcuda.Format.U8),
    ([1, 5, 7, 4], np.uint8, cvcuda.Format.RGBA8),
    ([5, 7], np.csingle, cvcuda.Format.C64),
    ([5, 7], np.cdouble, cvcuda.Format.C128),
    ([5, 7], np.dtype("2f"), cvcuda.Format._2F32),
    ([5, 7], np.dtype("2e"), cvcuda.Format._2F16),
    ([5, 7, 1], np.uint16, cvcuda.Format.U16),
]


@t.mark.parametrize("shape,dt,format", buffmt_common)
def test_wrap_host_buffer_infer_imgformat(shape, dt, format):
    img = cvcuda.Image(np.ndarray(shape, dt))
    assert img.width == 7
    assert img.height == 5
    assert img.format == format

    img = cvcuda.as_image(util.to_cuda_buffer(np.ndarray(shape, dt)))
    assert img.width == 7
    assert img.height == 5
    assert img.format == format


@t.mark.parametrize(
    "shape,dt,format",
    buffmt_common
    + [
        ([5, 7, 1], np.uint8, cvcuda.Format.Y8),
        ([5, 7, 3], np.uint8, cvcuda.Format.BGR8),
        ([5, 7, 4], np.uint8, cvcuda.Format.BGRA8),
    ],
)
def test_wrap_host_buffer_explicit_format(shape, dt, format):
    img = cvcuda.Image(np.ndarray(shape, dt), format)
    assert img.width == 7
    assert img.height == 5
    assert img.format == format

    img = cvcuda.as_image(util.to_cuda_buffer(np.ndarray(shape, dt)), format)
    assert img.width == 7
    assert img.height == 5
    assert img.format == format


buffmt2_common = [
    # packed formats
    (
        [((6, 8), np.uint8, np.uint8), ((3, 4, 2), np.uint8, np.uint8)],
        cvcuda.Format.NV12_ER,
    )
]


@t.mark.parametrize("buffers,format", buffmt2_common)
def test_wrap_host_buffer_infer_imgformat_multiple_planes(buffers, format):
    img = cvcuda.Image([np.ndarray(buf[0], buf[1]) for buf in buffers])
    assert img.width == 8
    assert img.height == 6
    assert img.format == format

    img = cvcuda.as_image(
        [cupy.asarray(np.zeros(buf[0], dtype=buf[2])) for buf in buffers]
    )
    assert img.width == 8
    assert img.height == 6
    assert img.format == format


@t.mark.parametrize("buffers,format", buffmt2_common)
def test_wrap_host_buffer_explicit_format2(buffers, format):
    img = cvcuda.Image([np.ndarray(buf[0], buf[1]) for buf in buffers], format)
    assert img.width == 8
    assert img.height == 6
    assert img.format == format

    img = cvcuda.as_image(
        [cupy.asarray(np.zeros(buf[0], dtype=buf[2])) for buf in buffers],
        format,
    )
    assert img.width == 8
    assert img.height == 6
    assert img.format == format


@t.mark.parametrize(
    "shape,dt,planes,height,width,channels",
    [
        ([2, 7, 6], np.uint8, 2, 7, 6, 2),
        ([1, 2, 7, 6], np.uint8, 2, 7, 6, 2),
        ([2, 7, 3], np.uint8, 1, 2, 7, 3),
        ([1, 7, 3], np.uint8, 1, 1, 7, 3),
        ([7, 3], np.uint8, 1, 7, 3, 1),
        ([7, 1], np.uint8, 1, 7, 1, 1),
        ([1, 3], np.uint8, 1, 1, 3, 1),
        ([1, 1], np.uint8, 1, 1, 1, 1),
        ([5, 7, 3], np.uint8, 1, 5, 7, 3),
    ],
)
def test_wrap_host_buffer_infer_format_geometry(
    shape, dt, planes, height, width, channels
):
    img = cvcuda.Image(np.ndarray(shape, dt))
    assert img.width == width
    assert img.height == height
    assert img.format.planes == planes
    assert img.format.channels == channels

    img = cvcuda.as_image(util.to_cuda_buffer(np.ndarray(shape, dt)))
    assert img.width == width
    assert img.height == height
    assert img.format.planes == planes
    assert img.format.channels == channels


@t.mark.parametrize("dtype", [np.float32, np.float16])
def test_wrap_host_buffer_arg_keywords(dtype):
    fmt = cvcuda.Format.F32 if dtype == np.float32 else cvcuda.Format.F16
    img = cvcuda.Image(buffer=np.ndarray([5, 7], dtype), format=fmt)
    assert img.size == (7, 5)
    assert img.format == fmt

    img = cvcuda.as_image(
        buffer=util.to_cuda_buffer(np.ndarray([5, 7], dtype)),
        format=fmt,
    )
    assert img.size == (7, 5)
    assert img.format == fmt


@t.mark.parametrize("dtype", [np.float32, np.float16])
def test_wrap_host_buffer_infer_format_arg_keywords(dtype):
    fmt = cvcuda.Format.F32 if dtype == np.float32 else cvcuda.Format.F16
    img = cvcuda.Image(buffer=np.ndarray([5, 7], dtype))
    assert img.size == (7, 5)
    assert img.format == fmt

    img = cvcuda.as_image(buffer=util.to_cuda_buffer(np.ndarray([5, 7], dtype)))
    assert img.size == (7, 5)
    assert img.format == fmt


def test_wrap_host_image_with_format__buffer_has_unsupported_type():
    with t.raises(ValueError):
        cvcuda.Image(np.array([1 + 2j, 4 + 7j]), cvcuda.Format._2F32)


def test_wrap_host_image__buffer_has_unsupported_type():
    with t.raises(ValueError):
        cvcuda.Image(np.array([1 + 2j, 4 + 7j]))


def test_wrap_host_image__format_and_buffer_type_mismatch():
    with t.raises(ValueError):
        cvcuda.Image(np.array([1.4, 2.85]), cvcuda.Format.U8)


def test_wrap_host_image__only_pitch_linear():
    with t.raises(ValueError):
        cvcuda.Image(np.ndarray([6, 4], np.uint8), cvcuda.Format.Y8_BL)


def test_wrap_host_image__css_with_one_plane_failure():
    with t.raises(ValueError):
        cvcuda.Image(np.ndarray([6, 4], np.uint8), cvcuda.Format.NV12)


@t.mark.parametrize(
    "shape",
    [
        (5, 3, 4),  # Buffer shape HCW not supported
        (5, 7, 4),  # Buffer shape doesn't correspond to image format
    ],
)
def test_wrap_host_image_with_format__invalid_shape(shape):
    with t.raises(ValueError):
        cvcuda.Image(np.ndarray(shape, np.uint8), cvcuda.Format.RGB8)


@t.mark.parametrize(
    "shape",
    [
        # When buffer's number of dimensions is 4, first dimension must be 1, not 2
        (
            2,
            3,
            4,
            5,
        ),
        # Number of dimensions must be between 1 and 4, not 5
        (1, 1, 15, 7, 1),
        # Number of dimensions must be between 1 and 4, not 0
        (0,),
        # Buffer shape not supported
        (8, 7, 9),
    ],
)
def test_wrap_host_invalid_dims(shape):
    with t.raises(ValueError):
        cvcuda.Image(np.ndarray(shape))


@t.mark.parametrize(
    "s",
    [
        # Fastest changing dimension must be packed, i.e.,
        # have stride equal to 1 bytes(s), not 2
        (2 * 2, 2),
        # Buffer strides must all be >= 0
        (0, 1),
    ],
)
def test_wrap_host_invalid_strides(s):
    with t.raises(ValueError):
        cvcuda.Image(
            np.ndarray(
                shape=(3, 2), strides=s, buffer=bytearray(s[0] * 3), dtype=np.uint8
            )
        )


@t.mark.parametrize(
    "shapes",
    [
        # When wrapping multiple buffers, buffers with 4
        # dimensions must have first dimension == 1, not 2
        [
            (3, 4),
            (2, 2, 3, 1),
        ],
        # Number of buffer#1's dimensions must be
        # between 1 and 4, not 5
        [
            (3, 4),
            (5, 2, 2, 3, 1),
        ],
    ],
)
def test_wrap_host_multiplane_invalid_dims(shapes):
    buffers = []
    for shape in shapes:
        buffers.append(np.ndarray(shape, np.uint8))

    with t.raises(ValueError):
        cvcuda.Image(buffers)


def test_image_wrap_invalid_cuda_buffer():
    class NonCudaMemory(object):
        pass

    obj = NonCudaMemory()
    obj.__cuda_array_interface__ = {
        "shape": (1, 1),
        "typestr": "i",
        "data": (419, True),
        "version": 3,
    }

    with t.raises(RuntimeError):
        cvcuda.as_image(obj)


def test_image_create_packed():
    img = cvcuda.Image((37, 11), cvcuda.Format.U8, rowalign=1)
    assert img.cuda().strides == (37, 1)


def test_image_create_zeros_packed():
    img = cvcuda.Image.zeros((37, 11), cvcuda.Format.U8, rowalign=1)
    assert img.cuda().strides == (37, 1)


def test_image_create_from_host_packed():
    img = cvcuda.Image(np.ndarray((11, 37), np.uint8), rowalign=1)
    assert img.cuda().strides == (37, 1)


@t.mark.parametrize(
    "size,format,layout,out_dtype, out_shape, simple_layout",
    [
        ((257, 231), cvcuda.Format.U8, None, np.uint8, (231, 257), "HWC"),
        ((257, 231), cvcuda.Format.U8, "HWC", np.uint8, (231, 257, 1), "HWC"),
        ((257, 231), cvcuda.Format.U8, "CHW", np.uint8, (1, 231, 257), "CHW"),
        (
            (257, 231),
            cvcuda.Format.U8,
            "xyCrodHlimaWab",
            np.uint8,
            (1, 1, 1, 1, 1, 1, 231, 1, 1, 1, 1, 257, 1, 1),
            "CHW",
        ),
        ((257, 231), cvcuda.Format.RGBAf16, None, np.float16, (231, 257, 4), "HWC"),
        ((257, 231), cvcuda.Format.RGBAf32, None, np.float32, (231, 257, 4), "HWC"),
        ((257, 231), cvcuda.Format.RGBA8, "HWC", np.uint8, (231, 257, 4), "HWC"),
        ((257, 231), cvcuda.Format.RGBA8p, None, np.uint8, (4, 231, 257), "CHW"),
        ((257, 231), cvcuda.Format.RGBAf16p, None, np.float16, (4, 231, 257), "CHW"),
        ((257, 231), cvcuda.Format.RGBAf32p, "CHW", np.float32, (4, 231, 257), "CHW"),
        (
            (258, 232),
            cvcuda.Format.NV12,
            None,
            [np.uint8, np.uint8],
            [(232, 258, 1), (232 // 2, 258 // 2, 2)],
            "HWC",
        ),
        (
            (258, 232),
            cvcuda.Format.NV12,
            "HWC",
            [np.uint8, np.uint8],
            [(232, 258, 1), (232 // 2, 258 // 2, 2)],
            "HWC",
        ),
        # For YUYV and friends things get a bit funky
        ((258, 232), cvcuda.Format.YUYV, None, np.uint8, (232, 258, 2), "HWC"),
        ((258, 232), cvcuda.Format.YUYV, "HWC", np.uint8, (232, 258, 2), "HWC"),
    ],
)
def test_image_export_cuda_buffer(
    size, format, layout, out_dtype, out_shape, simple_layout
):
    img = cvcuda.Image(size, format)

    mem = img.cuda(layout)
    if type(mem) is list:
        for i in range(0, len(mem)):
            assert mem[i].dtype == out_dtype[i]
            assert mem[i].shape == out_shape[i]
    else:
        assert mem.dtype == out_dtype
        assert mem.shape == out_shape

    # external buffer must not be reused
    assert img.cuda(layout) is not mem
    if layout is not None:
        newmem = img.cuda()
        assert newmem is not mem
        assert newmem is not img.cuda()
        assert newmem is not img.cuda(layout)

    cuda_buffer = img.cuda(simple_layout)
    if type(cuda_buffer) is not list:
        cuda_buffer = [cuda_buffer]

    rng = np.random.default_rng(0)

    gold_buffer = list()
    for buf in cuda_buffer:
        gold_data = (rng.random(size=buf.shape) * 255).astype(buf.dtype)
        gold_buffer.append(gold_data)

        cuda_buf = cupy.asarray(buf)
        cuda_buf[:] = cupy.asarray(gold_data)

    # Get values back on cpu
    host_buffer = img.cpu(simple_layout)
    if type(host_buffer) is not list:
        host_buffer = [host_buffer]

    if type(out_dtype) is not list:
        out_dtype = [out_dtype]

    # compare to see if they are correct
    for b in range(0, len(host_buffer)):
        np.testing.assert_array_equal(
            host_buffer[b], gold_buffer[b], "buffer #" + str(b) + " mismatch"
        )


def test_image_export_cuda_buffer_strides():
    # cupy returns packed buffers
    cuda_img = cupy.asarray(np.zeros((11, 37), dtype=np.uint8))

    img = cvcuda.as_image(cuda_img)

    data = img.cuda()

    assert data.strides == (37, 1)


@t.mark.parametrize(
    "fmt",
    [
        cvcuda.Format.U8,
        cvcuda.Format.U16,
        cvcuda.Format.U32,
        cvcuda.Format.F16,
        cvcuda.Format.F32,
    ],
)
def test_image_zeros(fmt):
    img = cvcuda.Image.zeros((67, 34), fmt)
    assert (img.cpu() == np.zeros((34, 67), cv_types.as_np_dtype(fmt))).all()


def test_image_is_kept_alive_by_cuda_array_interface():
    cvcuda.clear_cache()

    img1 = cvcuda.Image((640, 480), cvcuda.Format.U8)

    iface1 = img1.cuda()

    data_buffer1 = iface1.__cuda_array_interface__["data"][0]

    del img1

    img2 = cvcuda.Image((640, 480), cvcuda.Format.U8)
    assert img2.cuda().__cuda_array_interface__["data"][0] != data_buffer1

    del img2
    # remove img2 from cache, but not img1, as it's being
    # held by iface1
    cvcuda.clear_cache()

    # now img1 is free for reuse
    del iface1

    img3 = cvcuda.Image((640, 480), cvcuda.Format.U8)
    assert img3.cuda().__cuda_array_interface__["data"][0] == data_buffer1


def test_image_wrapper_nodeletion():
    """
    Check if image wrappers deletes memory that's not ours.
    """
    # run twice, first run is without cache re-usage, second is with cache re-usage
    for i in range(2):
        np_img = np.random.rand(1 + i, 2 + i).astype(np.float32)
        pt_img = cupy.asarray(np_img)

        nv_img = cvcuda.as_image(pt_img)
        del nv_img

        try:
            assert (pt_img.get() == np_img).all()
        except RuntimeError:
            assert False, "Invalid memory"


def test_image_size_in_bytes():
    """
    Checks if the computation of the image size in bytes is correct
    """
    img_create = cvcuda.Image.zeros((1, 1), cvcuda.Format.F32)
    assert cvcuda.internal.nbytes_in_cache(img_create) > 0

    np_img = np.random.rand(1, 1).astype(np.float32)
    img_create_host = cvcuda.Image(np_img)
    assert cvcuda.internal.nbytes_in_cache(img_create_host) > 0

    np_img = np.random.rand(1, 1).astype(np.float32)
    img_create_host_vector = cvcuda.Image([np_img, np_img])
    assert cvcuda.internal.nbytes_in_cache(img_create_host_vector) > 0

    pt_img = cupy.asarray(np_img)

    img_wrap_external_buffer = cvcuda.as_image(pt_img)
    assert cvcuda.internal.nbytes_in_cache(img_wrap_external_buffer) == 0

    img_wrap_external_buffer_vector = cvcuda.as_image([pt_img, pt_img])
    assert cvcuda.internal.nbytes_in_cache(img_wrap_external_buffer_vector) == 0


def test_as_image_does_not_leak_wrappers():
    """
    Regression test for GitHub issue #258 (as_image leaks memory).

    Each call to `cvcuda.as_image` made while an earlier result is still
    alive forces a *new* wrapper Image into the cache (the existing one
    is "in use").  Each cached wrapper keeps its wrapped GPU buffer alive
    via its ExternalBuffer.  Once the caller drops those Images the
    wrappers are no longer in use, but they used to stay in the cache
    indefinitely — every one still pinning its external buffer — until
    explicitly reused or `clear_cache` was called.

    The fix makes every as_image call run `removeAllNotInUseMatching` on
    the wrapper key (matching what `Tensor::WrapExternalBuffer` already
    does) so stale wrappers are freed promptly.
    """
    import gc

    cvcuda.clear_cache()
    gc.collect()
    assert cvcuda.cache_size() == 0

    N = 8
    tensors = [cupy.random.rand(16, 16).astype(cupy.float32) for _ in range(N)]

    # List comprehension: each as_image call sees earlier results as
    # "in use", so N distinct wrappers end up in the cache.
    imgs = [cvcuda.as_image(t) for t in tensors]
    assert cvcuda.cache_size() == N

    # Drop the Python handles.  Wrappers transition to "not in use" but
    # remain in the cache with their external-buffer references.
    del imgs
    gc.collect()
    assert cvcuda.cache_size() == N  # nothing has cleaned them up yet

    # One more as_image call.  With the fix this reuses one wrapper and
    # evicts the rest.  Without the fix the cache keeps growing.
    extra = cupy.random.rand(16, 16).astype(cupy.float32)
    img = cvcuda.as_image(extra)

    assert cvcuda.cache_size() == 1, (
        f"Expected 1 wrapper in cache after cleanup, got {cvcuda.cache_size()}. "
        "Stale as_image wrappers are accumulating (GitHub issue #258)."
    )
    del img
