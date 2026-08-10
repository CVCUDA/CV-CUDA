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

import cvcuda

import pytest
import numpy as np

import cvcuda_util as util
import cvcuda_tools as cv_tools

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "input_args,base_args,scale_args,globalscale,globalshift,epsilon,flags",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            ((1, 1), np.float32, "HW"),
            ((1, 1), np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            ((16, 1), np.float32, "HW"),
            ((16, 1), np.float32, "HW"),
            1,
            2,
            3,
            cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            ((1, 23), np.float32, "HW"),
            ((1, 23), np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            ((16, 23), np.float32, "HW"),
            ((16, 23), np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
    ],
)
def test_op_normalize(
    input_args, base_args, scale_args, globalscale, globalshift, epsilon, flags
):
    input = cvcuda.Tensor(*input_args)
    base = cvcuda.Tensor(*base_args)
    scale = cvcuda.Tensor(*scale_args)
    out = cvcuda.normalize(input, base, scale)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.normalize_into(out, input, base, scale)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.normalize(
        src=input,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.normalize_into(
        src=input,
        dst=out,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@pytest.mark.parametrize(
    "nimages,format,max_size,max_pixel,base_args,scale_args,globalscale,globalshift,epsilon,flags",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            ((1, 1, 1, 5), np.float32, "NHWC"),
            ((1, 1, 1, 5), np.float32, "NHWC"),
            1,
            2,
            3,
            None,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            ((1, 1, 1, 5), np.float32, "NHWC"),
            ((1, 1, 1, 5), np.float32, "NHWC"),
            1,
            2,
            3,
            cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        ),
    ],
)
def test_op_rotatevarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    base_args,
    scale_args,
    globalscale,
    globalshift,
    epsilon,
    flags,
):
    base = cvcuda.Tensor(*base_args)
    scale = cvcuda.Tensor(*scale_args)

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    out = cvcuda.normalize(input, base, scale)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    out = util.clone_image_batch(input)
    tmp = cvcuda.normalize_into(out, input, base, scale)
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()
    out = cvcuda.normalize(
        src=input,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    tmp = cvcuda.normalize_into(
        src=input,
        dst=out,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def _get_base_and_std(layout, channels):
    if layout == "NHWC":
        shape = (1, 1, 1, channels)
    elif layout == "HWC":
        shape = (1, 1, channels)
    elif layout == "NCHW":
        shape = (1, channels, 1, 1)
    elif layout == "CHW":
        shape = (channels, 1, 1)
    else:
        shape = (*[1] * (len(layout) - 1), channels)
    base = cvcuda.Tensor(shape, cvcuda.Type.F32, layout)
    std = cvcuda.Tensor(shape, cvcuda.Type.F32, layout)
    return base, std


def _normalize_params(dtype, layout, channels):
    base, std = _get_base_and_std(layout, channels)
    return {
        "base": base,
        "scale": std,
    }


def _normalize_list_params(dtype, layout, channels):
    # list[float] (tensor-free / by-value) base & scale: one value per channel, so the
    # by-value overload is exercised by the same support matrix as the tensor path.
    # dtype and layout are unused -- the values are plain floats regardless of input.
    base = [0.40 + 0.05 * i for i in range(channels)]
    scale = [0.20 + 0.03 * i for i in range(channels)]
    return {"base": base, "scale": scale}


def _normalize_varshape_params(dtype, layout, channels):
    """Build base/scale tensors for image_batch normalize tests.

    layout is unused: image_batch wrappers operate on image formats, not tensor
    layouts, so base/scale are always created as NHWC-shaped (batch,1,1,C) tensors.
    """
    base_data = np.zeros((2, 1, 1, channels), dtype=np.float32)
    scale_data = np.ones((2, 1, 1, channels), dtype=np.float32)
    return {
        "base": util.to_cvcuda_tensor(base_data, "NHWC"),
        "scale": util.to_cvcuda_tensor(scale_data, "NHWC"),
    }


def test_op_normalize_varshape_planar():
    nimages = 2
    img_format = cvcuda.Format.RGBA8p
    base = util.to_cvcuda_tensor(np.zeros((1, 4, 1, 1), dtype=np.float32), "NCHW")
    scale = util.to_cvcuda_tensor(np.ones((1, 4, 1, 1), dtype=np.float32), "NCHW")

    input = cvcuda.ImageBatchVarShape(nimages)
    for i in range(nimages):
        input.pushback(cvcuda.Image((16 + i * 3, 23 + i * 2), img_format))

    out = cvcuda.normalize(input, base, scale)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    out = util.clone_image_batch(input)
    tmp = cvcuda.normalize_into(out, input, base, scale)
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def _scalar_input(dtype, layout, channels):
    dims = {"N": 2, "H": 9, "W": 13, "C": channels}
    shape = tuple(dims[t] for t in layout)
    if np.issubdtype(np.dtype(dtype), np.floating):
        data = RNG.random(shape).astype(dtype)
    else:
        info = np.iinfo(dtype)
        data = RNG.integers(info.min, info.max, size=shape, endpoint=True).astype(dtype)
    return util.to_cvcuda_tensor(data, layout)


def _scalar_values(channels, mode):
    n = 1 if mode == "scalar" else channels
    base = [0.40 + 0.05 * i for i in range(n)]
    scale = [0.20 + 0.03 * i for i in range(n)]
    return base, scale


def _scalar_param_tensor(vals, layout):
    n = len(vals)
    # Parameter tensor mirrors the input layout with the channel axis = n (all others 1), so the
    # tensor-path oracle broadcasts base/scale per channel exactly like the by-value path.
    shapes = {
        "NHWC": (1, 1, 1, n),
        "HWC": (1, 1, n),
        "NCHW": (1, n, 1, 1),
        "CHW": (n, 1, 1),
    }
    return util.to_cvcuda_tensor(
        np.array(vals, np.float32).reshape(shapes[layout]), layout
    )


_STDDEV = cvcuda.NormalizeFlags.SCALE_IS_STDDEV
_DEFAULT_GLOBALS = (1.0, 0.0, 0.0)
_NONDEFAULT_GLOBALS = (2.0, 5.0, 1e-4)


@pytest.mark.parametrize(
    "dtype,channels,layout,flags,mode,global_profile",
    [
        (np.uint8, 1, "NHWC", None, "scalar", _DEFAULT_GLOBALS),
        (np.int8, 3, "HWC", _STDDEV, "per_channel", _NONDEFAULT_GLOBALS),
        (np.uint16, 4, "NCHW", None, "per_channel", _DEFAULT_GLOBALS),
        (np.int16, 3, "CHW", _STDDEV, "per_channel", _NONDEFAULT_GLOBALS),
        (np.int32, 1, "NCHW", None, "scalar", _NONDEFAULT_GLOBALS),
        (np.float32, 4, "NHWC", _STDDEV, "scalar", _DEFAULT_GLOBALS),
    ],
    ids=["u8-nhwc", "s8-hwc", "u16-nchw", "s16-chw", "s32-nchw", "f32-nhwc"],
)
def test_op_normalize_list_matches_tensor(
    dtype, channels, layout, flags, mode, global_profile
):
    globalscale, globalshift, epsilon = global_profile
    inp = _scalar_input(dtype, layout, channels)
    base_vals, scale_vals = _scalar_values(channels, mode)
    base_t = _scalar_param_tensor(base_vals, layout)
    scale_t = _scalar_param_tensor(scale_vals, layout)

    kw = dict(
        flags=flags, globalscale=globalscale, globalshift=globalshift, epsilon=epsilon
    )

    # Allocating overload: list result must equal tensor result byte-for-byte.
    out_ref = cvcuda.normalize(inp, base_t, scale_t, **kw)
    out_lst = cvcuda.normalize(inp, base_vals, scale_vals, **kw)
    assert out_lst.layout == inp.layout
    assert out_lst.shape == inp.shape
    assert out_lst.dtype == inp.dtype
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(out_ref.cuda()),
        util.to_cpu_numpy_buffer(out_lst.cuda()),
    )

    # _into overload: same equivalence.
    dst_ref = cvcuda.Tensor(inp.shape, inp.dtype, inp.layout)
    dst_lst = cvcuda.Tensor(inp.shape, inp.dtype, inp.layout)
    cvcuda.normalize_into(dst_ref, inp, base_t, scale_t, **kw)
    ret = cvcuda.normalize_into(dst_lst, inp, base_vals, scale_vals, **kw)
    assert ret is dst_lst
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(dst_ref.cuda()),
        util.to_cpu_numpy_buffer(dst_lst.cuda()),
    )


def test_op_normalize_list_accepts_tuple():
    """A tuple is accepted just like a list (both convert to std::vector<float>)."""
    inp = _scalar_input(np.float32, "HWC", 3)
    out_list = cvcuda.normalize(inp, [0.4, 0.45, 0.5], [0.2, 0.23, 0.26])
    out_tuple = cvcuda.normalize(inp, (0.4, 0.45, 0.5), (0.2, 0.23, 0.26))
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(out_list.cuda()),
        util.to_cpu_numpy_buffer(out_tuple.cuda()),
    )


@pytest.mark.parametrize(
    "bad_base,expected_exception",
    [
        ([], ValueError),
        ([0.5, 0.5], RuntimeError),
        ([0.1, 0.2, 0.3, 0.4, 0.5], ValueError),
    ],
    ids=["empty", "channel-mismatch", "too-many"],
)
def test_op_normalize_list_wrong_length_raises(bad_base, expected_exception):
    """The binding rejects invalid vector lengths; the operator rejects a valid
    vector length that is neither scalar nor equal to the input channel count."""
    inp = _scalar_input(np.float32, "NHWC", 3)
    with pytest.raises(expected_exception):
        cvcuda.normalize(inp, bad_base, [0.2] * max(len(bad_base), 1))


globals().update(
    cv_tools.make_op_tests(
        name="normalize",
        runner_info=[
            ("tensor", cvcuda.normalize, _normalize_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.S8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3, 4},
    )
)


globals().update(
    cv_tools.make_op_tests(
        name="normalize_varshape",
        runner_info=[
            ("image_batch", cvcuda.normalize, _normalize_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC"},
        supported_channels={1, 3, 4},
    )
)


# Tensor-free (list[float] / by-value) base & scale run through the same auto
# input-validation matrix as the tensor path. It supports interleaved and planar
# layouts and channels {1, 3, 4}, matching the tensor overload's support set.
globals().update(
    cv_tools.make_op_tests(
        name="normalize_list",
        runner_info=[
            ("tensor", cvcuda.normalize, _normalize_list_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.S8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3, 4},
        # The by-value overload rejects unsupported channel counts at the operator
        # (RuntimeError); the binding additionally rejects list lengths > 4 (ValueError),
        # which the negative-channels test hits for the 5- and 6-channel cases.
        negative_exceptions=[RuntimeError, ValueError],
    )
)
