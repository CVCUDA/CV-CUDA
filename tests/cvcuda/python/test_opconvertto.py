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
import cupy
import cvcuda_tools as cv_tools
import cvcuda_util as util


@pytest.mark.parametrize(
    "input_args,dtype,scale,offset",
    [
        (((5, 16, 23, 4), np.uint8, "NHWC"), np.float32, 1.2, 10.2),
        (((16, 23, 2), np.uint8, "HWC"), np.int32, -1.2, -5.5),
    ],
)
def test_op_convertto(input_args, dtype, scale, offset):
    input = cvcuda.Tensor(*input_args)
    out = cvcuda.convertto(input, dtype)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = cvcuda.Tensor(input.shape, dtype, input.layout)
    tmp = cvcuda.convertto_into(out, input)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = cvcuda.convertto(input, dtype, scale)
    out = cvcuda.convertto(input, dtype, scale, offset)

    out = cvcuda.Tensor(input.shape, dtype, input.layout)
    tmp = cvcuda.convertto_into(out, input, scale)
    tmp = cvcuda.convertto_into(out, input, scale, offset)

    stream = cvcuda.Stream()
    out = cvcuda.convertto(
        src=input, dtype=dtype, scale=scale, offset=offset, stream=stream
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    tmp = cvcuda.convertto_into(
        dst=out, src=input, scale=scale, offset=offset, stream=stream
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype


def test_op_convertto_round_mode():
    # NEAREST rounds float->int to nearest, TRUNCATE drops the fraction; values
    # (incl. negatives) chosen so the two modes differ.
    src = util.to_cvcuda_tensor(
        np.array([[[2.7], [-2.7], [3.4], [-3.4]]], dtype=np.float32), "HWC"
    )

    def convert(**kw):
        out = cvcuda.convertto(src, cvcuda.Type.S32, **kw)
        return cupy.asarray(out.cuda()).get().reshape(-1)

    np.testing.assert_array_equal(convert(round=cvcuda.Round.NEAREST), [3, -3, 3, -3])
    np.testing.assert_array_equal(convert(round=cvcuda.Round.TRUNCATE), [2, -2, 3, -3])
    np.testing.assert_array_equal(convert(), [3, -3, 3, -3])  # default == NEAREST


@pytest.mark.parametrize(
    "dtype,round_mode,inputs,expected",
    [
        # Values straddling every boundary that matters: in-range, just past the top, a full
        # period past the top, negative, and the half-way ties that distinguish the two modes.
        (
            cvcuda.Type.U16,
            cvcuda.Round.TRUNCATE_WRAP,
            [92414.59, 70000.0, -5000.0, -70000.0, 65535.9, 32768.0, 3.5, 2.5],
            # Unsigned maps negatives to zero, truncates (so 65535.9→65535, 3.5→3, 2.5→2),
            # and reduces positives modulo 65536.
            [26878, 4464, 0, 0, 65535, 32768, 3, 2],
        ),
        (
            cvcuda.Type.S16,
            cvcuda.Round.NEAREST_WRAP,
            [92414.59, 70000.0, -5000.0, -70000.0, 65535.9, 32768.0, 3.5, 2.5],
            # Signed wraps in both directions; nearest ties-to-even (3.5→4, 2.5→2).
            [26879, 4464, -5000, -4464, 0, -32768, 4, 2],
        ),
        (
            cvcuda.Type.U8,
            cvcuda.Round.TRUNCATE_WRAP,
            # Use values outside [0, 255] to exercise wrapping; span=256.
            [300.0, 256.0, -50.0, -300.0, 255.0, 255.9, 3.5, 2.5],
            # 300→44, 256→0, negatives→0, 255→255, 255.9 truncated→255, 3.5→3, 2.5→2.
            [44, 0, 0, 0, 255, 255, 3, 2],
        ),
        (
            cvcuda.Type.S8,
            cvcuda.Round.NEAREST_WRAP,
            # span=256, half=128; 200→-56; -200→56; 100→100; -50→-50; 127→127; -128→-128.
            [200.0, -200.0, 100.0, -50.0, 127.0, -128.0, 3.5, 2.5],
            [-56, 56, 100, -50, 127, -128, 4, 2],
        ),
    ],
)
def test_op_convertto_wrap_round_modes(dtype, round_mode, inputs, expected):
    src = util.to_cvcuda_tensor(
        np.array([[inputs]], dtype=np.float32).reshape(1, len(inputs), 1),
        "HWC",
    )
    out = cvcuda.convertto(src, dtype, round=round_mode)
    np.testing.assert_array_equal(cupy.asarray(out.cuda()).get().reshape(-1), expected)


def test_op_convertto_wrap_saturates_elsewhere():
    """The saturating modes are untouched by the addition of the wrapping ones."""
    src = util.to_cvcuda_tensor(
        np.array([[[92414.59], [-5000.0]]], dtype=np.float32), "HWC"
    )
    out = cvcuda.convertto(src, cvcuda.Type.U16, round=cvcuda.Round.NEAREST)
    np.testing.assert_array_equal(
        cupy.asarray(out.cuda()).get().reshape(-1), [65535, 0]
    )


@pytest.mark.parametrize(
    "round_mode", [cvcuda.Round.NEAREST_WRAP, cvcuda.Round.TRUNCATE_WRAP]
)
def test_op_gamma_contrast_rejects_wrap_round_modes(round_mode):
    """gamma_contrast shares NVCVRoundMode but implements only the saturating modes."""
    src = util.to_cvcuda_tensor(np.zeros((4, 4, 3), dtype=np.uint8), "HWC")
    with pytest.raises(RuntimeError):
        cvcuda.gamma_contrast(src, 1.0, round=round_mode)


@pytest.mark.parametrize(
    "round_mode", [cvcuda.Round.NEAREST_WRAP, cvcuda.Round.TRUNCATE_WRAP]
)
def test_op_convertto_wrap_rejects_s32_destination(round_mode):
    """Wrap modes are rejected for S32: float32 working arithmetic cannot represent the
    2^31 wrap period without quantization error."""
    src = util.to_cvcuda_tensor(np.zeros((1, 4, 1), dtype=np.float32), "HWC")
    with pytest.raises(RuntimeError):
        cvcuda.convertto(src, cvcuda.Type.S32, round=round_mode)


def _is_float(dt):
    return np.issubdtype(dt, np.floating)


_CONVERSION_CASES = [
    # Each supported source and destination dtype appears exactly once. Boundary
    # values exercise integer saturation and float-to-integer truncation; the
    # float16 source values are exactly half-representable.
    (np.uint8, np.float16, cvcuda.Type.F16, [0, 1, 127, 255], 1.0 / 255.0),
    (np.int8, np.uint8, cvcuda.Type.U8, [-128, -1, 0, 127], 1.0),
    (np.uint16, np.int16, cvcuda.Type.S16, [0, 32767, 32768, 65535], 1.0),
    (np.int16, np.uint16, cvcuda.Type.U16, [-32768, -1, 0, 32767], 1.0),
    (np.int32, np.float64, cvcuda.Type.F64, [-16777216, -1, 0, 16777216], 0.5),
    (np.float16, np.float32, cvcuda.Type.F32, [-2.5, -0.5, 0.5, 3.5], 0.5),
    (np.float32, np.int32, cvcuda.Type.S32, [-3.7, -2.5, 2.5, 3.7], 1.0),
    (np.float64, np.int8, cvcuda.Type.S8, [-200.9, -2.7, 2.7, 200.9], 1.0),
]


@pytest.mark.parametrize(
    "in_np,out_np,out_t,values,scale",
    _CONVERSION_CASES,
    ids=[f"{i.__name__}->{o.__name__}" for i, o, _, _, _ in _CONVERSION_CASES],
)
def test_op_convertto_dtype_axes(in_np, out_np, out_t, values, scale):
    arr = np.asarray(values, dtype=in_np).reshape(1, len(values), 1)
    work = arr.astype(np.float64) * scale
    if _is_float(out_np):
        gold = work.astype(out_np)
    else:
        info = np.iinfo(out_np)
        gold = np.clip(np.trunc(work), info.min, info.max).astype(out_np)

    src = util.to_cvcuda_tensor(arr, "HWC")
    allocated = cvcuda.convertto(src, out_t, scale, round=cvcuda.Round.TRUNCATE)

    into = cvcuda.Tensor(src.shape, out_t, src.layout)
    returned = cvcuda.convertto_into(into, src, scale, round=cvcuda.Round.TRUNCATE)
    assert returned is into

    # float16 outputs round once to half after a float32 mul, while the gold scales in
    # float64 before rounding, so allow one F16 ULP at the [0, 1] result magnitude
    # (2^-10); wider float outputs and exact integer conversions keep the tight bounds.
    if out_np == np.float16:
        atol = 1e-3
    elif _is_float(out_np):
        atol = 1e-7
    else:
        atol = 0
    for result in (allocated, into):
        np.testing.assert_allclose(
            cupy.asarray(result.cuda()).get().reshape(-1).astype(np.float64),
            gold.reshape(-1).astype(np.float64),
            atol=atol,
            rtol=0,
        )


def _convertto_params(dtype, layout, channels, **extra):
    return {"dtype": cvcuda.Type.F64, **extra}


globals().update(
    cv_tools.make_op_tests(
        name="convertto",
        runner_info=[("tensor", cvcuda.convertto, _convertto_params)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.S8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F16,
            cvcuda.Type.F32,
            cvcuda.Type.F64,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        extra_params={"round": {cvcuda.Round.NEAREST, cvcuda.Round.TRUNCATE}},
    )
)
