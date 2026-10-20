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

import cupy as cp
import cvcuda
import cvcuda_tools as cv_tools
import cvcuda_types as cv_types
import numpy as np
import pytest


FAMILIES = [
    (cvcuda.Interp.NEAREST, False),
    (cvcuda.Interp.NEAREST_FLOOR, True),
    (cvcuda.Interp.LINEAR, False),
    (cvcuda.Interp.LINEAR, True),
    (cvcuda.Interp.CUBIC, False),
    (cvcuda.Interp.CUBIC, True),
]
DTYPES = [np.uint8, np.uint16, np.int16, np.float16, np.float32]


def _shape(layout, h, w, c=3, n=2):
    extents = dict(N=n, H=h, W=w, C=c)
    return tuple(extents[axis] for axis in layout)


def _host(tensor):
    cvcuda.Stream.current.sync()
    return cp.asnumpy(cp.asarray(tensor.cuda()))


def _batch(sizes, fmt=cvcuda.Format.RGB8):
    batch = cvcuda.ImageBatchVarShape(max(2, len(sizes)))
    for h, w in sizes:
        batch.pushback(cvcuda.Image((w, h), fmt))
    return batch


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "layout,channels", [("HWC", 1), ("NHWC", 2), ("HWC", 4), ("CHW", 5), ("NCHW", 17)]
)
@pytest.mark.parametrize("interp,aa", FAMILIES)
def test_op_tvresize_tensor_api(dtype, layout, channels, interp, aa):
    src = cvcuda.Tensor(_shape(layout, 11, 17, channels), dtype, layout)
    stream = cvcuda.Stream()
    out = cvcuda.tvresize(
        src=src, size=(7, 9), interp=interp, antialias=aa, stream=stream
    )
    assert out.shape == _shape(layout, 7, 9, channels)
    assert out.layout == src.layout
    assert out.dtype == src.dtype
    dst = cvcuda.Tensor(out.shape, dtype, layout)
    assert cvcuda.tvresize_into(dst, src, interp, aa, stream=stream) is dst
    stream.sync()


@pytest.mark.parametrize(
    "h,w,size,maximum,expected",
    [
        (7, 13, 5, None, (5, 9)),
        (13, 7, [5], None, (9, 5)),
        (11, 11, (7,), None, (7, 7)),
        (7, 13, [5], 8, (4, 8)),
        (13, 7, [5], 8, (8, 4)),
        (7, 13, None, 26, (14, 26)),
        (13, 7, None, 26, (26, 14)),
        (7, 13, None, 8, (4, 8)),
        (7, 13, [4, 6], None, (4, 6)),
    ],
)
@pytest.mark.parametrize("layout", ["HWC", "NHWC", "CHW", "NCHW"])
def test_op_tvresize_size_binding(h, w, size, maximum, expected, layout):
    src = cvcuda.Tensor(_shape(layout, h, w), np.uint8, layout)
    out = cvcuda.tvresize(src, size, max_size=maximum)
    assert out.shape == _shape(layout, *expected)


@pytest.mark.parametrize("interp,aa", FAMILIES)
@pytest.mark.parametrize(
    "fmt",
    [
        cvcuda.Format.U8,
        cvcuda.Format.U16,
        cvcuda.Format.S16,
        cvcuda.Format.F16,
        cvcuda.Format.F32,
        cvcuda.Format.RGB8,
        cvcuda.Format.RGBA8,
    ],
)
def test_op_tvresize_varshape_api(interp, aa, fmt):
    src = _batch([(11, 17), (1, 7), (7, 1)], fmt)
    stream = cvcuda.Stream()
    out = cvcuda.tvresize(src, [5, 9], interp, None, aa, stream=stream)
    assert len(out) == len(src)
    assert out.capacity == src.capacity
    assert all(image.size == (9, 5) and image.format == fmt for image in out)
    dst = _batch([(3, 7), (6, 2), (1, 4)], fmt)
    assert cvcuda.tvresize_into(dst, src, interp, aa, stream=stream) is dst
    stream.sync()


@pytest.mark.parametrize(
    "size,maximum,expected",
    [(5, 8, [(8, 4), (4, 8)]), (None, 26, [(26, 14), (14, 26)])],
)
def test_op_tvresize_varshape_independent_size(size, maximum, expected):
    src = _batch([(7, 13), (13, 7)])
    out = cvcuda.tvresize(src, size, max_size=maximum)
    assert [image.size for image in out] == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"size": 0},
        {"size": -1},
        {"size": []},
        {"size": [1, 2, 3]},
        {"size": [0, 3]},
        {"size": [2, -3]},
        {"max_size": 0},
        {"size": 3, "max_size": 3},
        {"size": 3, "max_size": 2},
        {"size": [2, 3], "max_size": 9},
        {"size": 2**80},
        {"size": [2**31, 2]},
        {"max_size": 2**80},
    ],
)
@pytest.mark.parametrize("varshape", [False, True])
def test_op_tvresize_invalid_size(kwargs, varshape):
    src = _batch([(7, 13)]) if varshape else cvcuda.Tensor((7, 13, 3), np.uint8, "HWC")
    with pytest.raises(ValueError):
        cvcuda.tvresize(src, **kwargs)


@pytest.mark.parametrize("size", [1.5, "5", [2.5], ["5"], object()])
def test_op_tvresize_size_types(size):
    src = cvcuda.Tensor((7, 13, 3), np.uint8, "HWC")
    with pytest.raises(TypeError):
        cvcuda.tvresize(src, size)


def test_op_tvresize_zero_or_overflow_result():
    src = cvcuda.Tensor((1, 64, 3), np.uint8, "HWC")
    for kwargs in [{"max_size": 5}, {"size": 1, "max_size": 5}, {"size": 2**31 - 1}]:
        with pytest.raises(ValueError):
            cvcuda.tvresize(src, **kwargs)


@pytest.mark.parametrize("kwargs", [{"size": [4, 5]}, {"max_size": 5}])
@pytest.mark.parametrize("varshape", [False, True])
def test_op_tvresize_into_rejects_size(kwargs, varshape):
    src = _batch([(7, 13)]) if varshape else cvcuda.Tensor((7, 13, 3), np.uint8, "HWC")
    dst = _batch([(4, 5)]) if varshape else cvcuda.Tensor((4, 5, 3), np.uint8, "HWC")
    with pytest.raises(TypeError):
        cvcuda.tvresize_into(dst, src, **kwargs)


def test_op_tvresize_stream_is_keyword_only():
    src = cvcuda.Tensor((7, 13, 3), np.uint8, "HWC")
    stream = cvcuda.Stream()
    with pytest.raises(TypeError):
        cvcuda.tvresize(src, 4, cvcuda.Interp.LINEAR, None, True, stream)
    dst = cvcuda.Tensor((4, 7, 3), np.uint8, "HWC")
    with pytest.raises(TypeError):
        cvcuda.tvresize_into(dst, src, cvcuda.Interp.LINEAR, True, stream)


@pytest.mark.parametrize(
    "interp", [cvcuda.Interp.AREA, cvcuda.Interp.LANCZOS, cvcuda.Interp.GAUSSIAN]
)
def test_op_tvresize_unsupported_mode(interp):
    src = cvcuda.Tensor((7, 13, 3), np.uint8, "HWC")
    with pytest.raises(RuntimeError):
        cvcuda.tvresize(src, (4, 5), interp)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int32, np.float64, cvcuda.Type._3U8, cvcuda.Type._2F16]
)
def test_op_tvresize_unsupported_tensor_dtype(dtype):
    src = cvcuda.Tensor((7, 13, 3), dtype, "HWC")
    with pytest.raises(RuntimeError):
        cvcuda.tvresize(src, (4, 5))


@pytest.mark.parametrize(
    "shape,layout", [((7, 13, 5), "HWC"), ((7, 13), "HW"), ((3, 7, 13), "CWH")]
)
def test_op_tvresize_unsupported_tensor_layout_channels(shape, layout):
    src = cvcuda.Tensor(shape, np.uint8, layout)
    with pytest.raises(RuntimeError):
        cvcuda.tvresize(src, (4, 5))


@pytest.mark.parametrize(
    "fault",
    ["mixed_source", "mixed_output", "count", "two_channels", "multiplane", "format"],
)
def test_op_tvresize_varshape_rejection(fault):
    src = _batch([(7, 13), (9, 11)])
    dst = _batch([(4, 5), (3, 6)])
    if fault == "mixed_source":
        src = _batch([(7, 13)])
        src.pushback(cvcuda.Image((11, 9), cvcuda.Format.RGBA8))
    elif fault == "mixed_output":
        dst = _batch([(4, 5)])
        dst.pushback(cvcuda.Image((6, 3), cvcuda.Format.RGBA8))
    elif fault == "count":
        dst = _batch([(4, 5)])
    elif fault == "two_channels":
        src = _batch([(7, 13), (9, 11)], cvcuda.Format._2S16)
        dst = _batch([(4, 5), (3, 6)], cvcuda.Format._2S16)
    elif fault == "multiplane":
        src = _batch([(7, 13), (9, 11)], cvcuda.Format.RGB8p)
    else:
        dst = _batch([(4, 5), (3, 6)], cvcuda.Format.BGR8)
    with pytest.raises(RuntimeError):
        cvcuda.tvresize_into(dst, src)
    if fault in {"mixed_source", "two_channels", "multiplane"}:
        with pytest.raises(RuntimeError):
            cvcuda.tvresize(src, (4, 5))


@pytest.mark.parametrize(
    "source_type,destination_type",
    [(np.uint8, np.float32), (np.float32, np.uint8), (np.uint16, np.int16)],
)
@pytest.mark.parametrize(
    "interp,aa",
    [
        (cvcuda.Interp.LINEAR, True),
        (cvcuda.Interp.CUBIC, True),
        (cvcuda.Interp.NEAREST, False),
        (cvcuda.Interp.LINEAR, False),
    ],
)
def test_op_tvresize_dtype_mismatch_preserves_destination(
    source_type, destination_type, interp, aa
):
    src_mem = (
        cp.arange(11 * 17 * 3, dtype=cp.float32).astype(source_type).reshape(11, 17, 3)
    )
    storage = cp.full((7, 11, 3), 37, dtype=destination_type)
    dst_mem = storage[1:6, 1:8, :]
    src = cvcuda.as_tensor(src_mem, "HWC")
    dst = cvcuda.as_tensor(dst_mem, "HWC")
    cp.cuda.get_current_stream().synchronize()
    before = cp.asnumpy(storage)
    with pytest.raises(RuntimeError):
        cvcuda.tvresize_into(dst, src, interp, aa)
    cvcuda.Stream.current.sync()
    np.testing.assert_array_equal(cp.asnumpy(storage), before)


@pytest.mark.parametrize(
    "source_type,destination_type,source_format,destination_format",
    [
        (np.uint8, np.float32, cvcuda.Format.U8, cvcuda.Format.F32),
        (np.float32, np.uint8, cvcuda.Format.F32, cvcuda.Format.U8),
        (np.uint16, np.int16, cvcuda.Format.U16, cvcuda.Format.S16),
    ],
)
@pytest.mark.parametrize(
    "interp,aa",
    [
        (cvcuda.Interp.LINEAR, True),
        (cvcuda.Interp.CUBIC, True),
        (cvcuda.Interp.NEAREST, False),
        (cvcuda.Interp.LINEAR, False),
    ],
)
def test_op_tvresize_varshape_dtype_mismatch_preserves_destination(
    source_type, destination_type, source_format, destination_format, interp, aa
):
    src_mem = (
        cp.arange(11 * 17, dtype=cp.float32).astype(source_type).reshape(11, 17, 1)
    )
    storage = cp.full((7, 11, 1), 37, dtype=destination_type)
    dst_mem = storage[1:6, 1:8, :]
    src = cvcuda.ImageBatchVarShape(1)
    dst = cvcuda.ImageBatchVarShape(1)
    src.pushback(cvcuda.as_image(src_mem, source_format))
    dst.pushback(cvcuda.as_image(dst_mem, destination_format))
    cp.cuda.get_current_stream().synchronize()
    before = cp.asnumpy(storage)
    with pytest.raises(RuntimeError):
        cvcuda.tvresize_into(dst, src, interp, aa)
    cvcuda.Stream.current.sync()
    np.testing.assert_array_equal(cp.asnumpy(storage), before)


@pytest.mark.parametrize("interp,aa", FAMILIES)
def test_op_tvresize_empty_varshape(interp, aa):
    src = _batch([])
    dst = _batch([])
    out = cvcuda.tvresize(src, (4, 5), interp, antialias=aa)
    assert len(out) == 0
    assert cvcuda.tvresize_into(dst, src, interp, aa) is dst


@pytest.mark.parametrize("interp", [cvcuda.Interp.LINEAR, cvcuda.Interp.CUBIC])
def test_op_tvresize_antialias_defaults_and_none(interp):
    checker = (np.indices((17, 23)).sum(axis=0) % 2).astype(np.float32)[..., None]
    data = cp.asarray(checker)
    cp.cuda.get_current_stream().synchronize()
    src = cvcuda.as_tensor(data, "HWC")
    default = cvcuda.tvresize(src, (7, 11), interp)
    enabled = cvcuda.tvresize(src, (7, 11), interp, antialias=True)
    disabled = cvcuda.tvresize(src, (7, 11), interp, antialias=False)
    none = cvcuda.tvresize(src, (7, 11), interp, antialias=None)
    np.testing.assert_array_equal(_host(default), _host(enabled))
    np.testing.assert_array_equal(_host(disabled), _host(none))
    assert not np.array_equal(_host(default), _host(disabled))
    dst = cvcuda.Tensor(default.shape, np.float32, "HWC")
    cvcuda.tvresize_into(dst, src, interp)
    np.testing.assert_array_equal(_host(dst), _host(enabled))
    cvcuda.tvresize_into(dst, src, interp, None)
    np.testing.assert_array_equal(_host(dst), _host(disabled))


@pytest.mark.parametrize("interp", [cvcuda.Interp.NEAREST, cvcuda.Interp.NEAREST_FLOOR])
@pytest.mark.parametrize("dtype", DTYPES)
def test_op_tvresize_nearest_aa_invariance(dtype, interp):
    values = np.arange(11 * 17, dtype=np.float32).reshape(11, 17, 1).astype(dtype)
    data = cp.asarray(values)
    cp.cuda.get_current_stream().synchronize()
    src = cvcuda.as_tensor(data, "HWC")
    plain = cvcuda.tvresize(src, (7, 9), interp, antialias=False)
    aa = cvcuda.tvresize(src, (7, 9), interp, antialias=True)
    np.testing.assert_array_equal(_host(plain).view(np.uint8), _host(aa).view(np.uint8))


@pytest.mark.parametrize("h,w", [(1, 7), (7, 1), (1, 1)])
def test_op_tvresize_plain_linear_singletons(h, w):
    data = cp.full((h, w, 3), 3.25, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    src = cvcuda.as_tensor(data, "HWC")
    out = cvcuda.tvresize(src, (5, 9), antialias=False)
    np.testing.assert_allclose(_host(out), 3.25, rtol=0, atol=1e-5)


@pytest.mark.parametrize("interp,aa", FAMILIES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_op_tvresize_identity_exact(dtype, interp, aa):
    values = np.arange(7 * 13 * 3).reshape(7, 13, 3).astype(dtype)
    data = cp.asarray(values)
    cp.cuda.get_current_stream().synchronize()
    src = cvcuda.as_tensor(data, "HWC")
    dst = cvcuda.tvresize(src, (7, 13), interp, antialias=aa)
    np.testing.assert_array_equal(_host(dst).view(np.uint8), values.view(np.uint8))


@pytest.mark.parametrize(
    "shape,layout", [((0, 7, 13, 3), "NHWC"), ((0, 5, 7, 13), "NCHW")]
)
def test_op_tvresize_empty_tensor_rejected_by_constructor(shape, layout):
    storage = cp.empty((1, *shape[1:]), dtype=cp.uint8)
    cp.cuda.get_current_stream().synchronize()

    class CudaArrayInterfaceObject:
        pass

    # Keep the valid allocation pointer so zero-extent validation, not NULL-buffer validation, is tested.
    buffer = CudaArrayInterfaceObject()
    buffer.__cuda_array_interface__ = dict(
        storage.__cuda_array_interface__, shape=shape
    )
    assert buffer.__cuda_array_interface__["data"][0] != 0
    with pytest.raises(RuntimeError, match="NVCV_ERROR_INVALID_ARGUMENT"):
        cvcuda.as_tensor(buffer, layout)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("interp,aa", FAMILIES)
def test_op_tvresize_native_planar_matches_interleaved(dtype, interp, aa):
    values = (np.arange(2 * 11 * 17 * 3) % 251).reshape(2, 11, 17, 3).astype(dtype)
    data = cp.asarray(values)
    cp.cuda.get_current_stream().synchronize()
    interleaved = cvcuda.as_tensor(data, "NHWC")
    planar = cvcuda.reformat(interleaved, "NCHW")
    result = cvcuda.tvresize(interleaved, (7, 9), interp, antialias=aa)
    planar_result = cvcuda.tvresize(planar, (7, 9), interp, antialias=aa)
    actual = cvcuda.reformat(planar_result, "NHWC")
    np.testing.assert_array_equal(
        _host(actual).view(np.uint8), _host(result).view(np.uint8)
    )


@pytest.mark.parametrize("varshape", [False, True])
def test_op_tvresize_resource_overflow_before_allocation(varshape):
    src = _batch([(7, 13)]) if varshape else cvcuda.Tensor((7, 13, 3), np.uint8, "HWC")
    with pytest.raises(RuntimeError, match="NVCV_ERROR_INVALID_ARGUMENT"):
        cvcuda.tvresize(src, (2**31 - 1, 2**31 - 1))


def _tvresize_params(dtype, layout, channels):
    return {"size": (7, 9), "interp": cvcuda.Interp.LINEAR}


_supported_dtypes = {
    cvcuda.Type.U8,
    cvcuda.Type.U16,
    cvcuda.Type.S16,
    cvcuda.Type.F16,
    cvcuda.Type.F32,
}

globals().update(
    cv_tools.make_op_tests(
        name="tvresize",
        runner_info=[("tensor", cvcuda.tvresize, _tvresize_params)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes=_supported_dtypes,
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels=cv_types.CHANNELS,
        # Only planar tensors support more than four channels.
        exclude_dlc=[
            (None, layout, channels)
            for layout in ("NHWC", "HWC")
            for channels in cv_types.CHANNELS - {1, 2, 3, 4}
        ],
    )
)

globals().update(
    cv_tools.make_op_tests(
        name="tvresize_varshape",
        runner_info=[("image_batch", cvcuda.tvresize, _tvresize_params)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes=_supported_dtypes,
        supported_layouts={"NHWC", "HWC"},
        supported_channels={1, 3, 4},
    )
)
