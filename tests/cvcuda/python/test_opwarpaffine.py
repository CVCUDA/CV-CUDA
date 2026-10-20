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

import cupy
import pytest
import numpy as np

import cvcuda_util as util
import cvcuda_tools as cv_tools
from cvcuda_test_utils import as_cupy_stream

RNG = np.random.default_rng(0)

_DELAY_KERNEL = cupy.RawKernel(
    r"""
extern "C" __global__ void cvcuda_warp_affine_delay(
    unsigned long long cycles)
{
    unsigned long long start = clock64();
    while (clock64() - start < cycles) { }
}
""",
    "cvcuda_warp_affine_delay",
)


def test_op_warp_affine_constant_coverage_tensor_only():
    src = cvcuda.Tensor((1, 4, 4, 1), cvcuda.Type.U8, "NHWC")
    xform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    kwargs = {
        "flags": cvcuda.Interp.LINEAR,
        "border_mode": cvcuda.Border.CONSTANT_COVERAGE,
        "border_value": [0.5],
    }
    assert isinstance(cvcuda.warp_affine(src, xform, **kwargs), cvcuda.Tensor)
    dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    assert cvcuda.warp_affine_into(dst, src, xform, **kwargs) is dst

    batch = util.create_image_batch(1, cvcuda.Format.U8, size=(4, 4))
    batch_dst = util.clone_image_batch(batch)
    batch_xform = util.to_cvcuda_tensor(xform.reshape(1, 6), "NC")
    with pytest.raises(RuntimeError):
        cvcuda.warp_affine(batch, batch_xform, **kwargs)
    with pytest.raises(RuntimeError):
        cvcuda.warp_affine_into(batch_dst, batch, batch_xform, **kwargs)


def test_op_warp_affine_host_array_binding_contract():
    source = cupy.zeros((1, 2, 2, 3), dtype=cupy.uint8)
    input = cvcuda.as_tensor(source, "NHWC")
    xform = np.array([[1, 0, 100], [0, 1, 100]], dtype=np.float32)
    xform.setflags(write=False)

    output = cvcuda.warp_affine(
        input,
        xform,
        cvcuda.Interp.NEAREST,
        border_mode=cvcuda.Border.CONSTANT,
        border_value=7,
    )

    np.testing.assert_array_equal(
        cupy.asarray(output.cuda()).get(), np.full(source.shape, 7, dtype=np.uint8)
    )

    with pytest.raises(TypeError, match="incompatible function arguments"):
        cvcuda.warp_affine(
            input,
            cupy.asarray(xform),
            cvcuda.Interp.NEAREST,
            border_mode=cvcuda.Border.CONSTANT,
        )


def test_op_warp_affine_varshape_completion_lease_across_streams():
    """In-flight coefficient scratch must not serialize another stream."""
    cvcuda.clear_cache()
    input_a = util.create_image_batch(
        2,
        cvcuda.Format.RGB8,
        size=(64, 64),
        max_random=255,
        rng=RNG,
    )
    input_b = util.create_image_batch(
        2,
        cvcuda.Format.RGB8,
        size=(64, 64),
        max_random=255,
        rng=RNG,
    )
    identity = util.to_cvcuda_tensor(
        np.tile([1, 0, 0, 0, 1, 0], (len(input_a), 1)).astype(np.float32),
        "NC",
    )
    translated = util.to_cvcuda_tensor(
        np.tile([1, 0, 256, 0, 1, 256], (len(input_b), 1)).astype(np.float32),
        "NC",
    )
    output_a = util.clone_image_batch(input_a)
    output_b = util.clone_image_batch(input_b)

    # Prime two operator states, their coefficient allocations, and the CUDA
    # module before introducing the delay; lazy initialization could otherwise
    # hide whether the second stream was independently submitted.
    warm_stream = cvcuda.Stream()
    cvcuda.warp_affine_into(
        output_a,
        input_a,
        identity,
        cvcuda.Interp.NEAREST,
        border_value=[0, 0, 0],
        stream=warm_stream,
    )
    cvcuda.warp_affine_into(
        output_b,
        input_b,
        translated,
        cvcuda.Interp.NEAREST,
        border_value=[0, 0, 0],
        stream=warm_stream,
    )
    warm_stream.sync()
    cvcuda.internal.drain_pending_releases()

    stream_a = cvcuda.Stream()
    stream_b = cvcuda.Stream()
    external_a = as_cupy_stream(stream_a.handle)
    _DELAY_KERNEL(
        (1,),
        (1,),
        (np.uint64(500_000_000),),
        stream=external_a,
    )
    assert not external_a.done

    cvcuda.warp_affine_into(
        output_a,
        input_a,
        identity,
        cvcuda.Interp.NEAREST,
        border_value=[0, 0, 0],
        stream=stream_a,
    )
    cvcuda.warp_affine_into(
        output_b,
        input_b,
        translated,
        cvcuda.Interp.NEAREST,
        border_mode=cvcuda.Border.CONSTANT,
        border_value=[73, 73, 73],
        stream=stream_b,
    )
    stream_b.sync()
    submission_a_still_pending = not external_a.done
    stream_a.sync()

    assert submission_a_still_pending, (
        "the second WarpAffine cache acquisition waited for the first "
        "stream instead of leasing independent coefficient scratch"
    )
    for source, result in zip(input_a, output_a, strict=True):
        np.testing.assert_array_equal(
            cupy.from_dlpack(result.cuda()).get(),
            cupy.from_dlpack(source.cuda()).get(),
        )
    for result in output_b:
        np.testing.assert_array_equal(
            cupy.from_dlpack(result.cuda()).get(),
            np.full((64, 64, 3), 73, dtype=np.uint8),
        )


@pytest.mark.parametrize(
    "input_args, xform, flags, border_mode, border_value",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            [[1, 0, 0], [0, 1, 0]],
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [],
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            [[1, 0, 0], [0, 1, 0]],
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            [[1, 2, 0], [2, 1, 1]],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [1, 2, 3, 4],
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            [[1, 2, 0], [2, 1, 1]],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
        (
            ((11, 21, 4), np.uint8, "HWC"),
            [[2, 2, 0], [3, 1, 0]],
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            ((11, 21, 4), np.uint8, "HWC"),
            [[2, 2, 1], [3, 1, 2]],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [1, 2, 3, 4],
        ),
        (
            ((11, 21, 4), np.uint8, "HWC"),
            [[1, 2, 0], [2, 1, 1]],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
        (
            ((5, 16, 23, 4), np.float16, "NHWC"),
            [[1, 2, 0], [2, 1, 1]],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
    ],
)
def test_op_warp_affine(input_args, xform, flags, border_mode, border_value):
    input = cvcuda.Tensor(*input_args)
    out = cvcuda.warp_affine(
        input, xform, flags, border_mode=border_mode, border_value=border_value
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.warp_affine_into(
        src=input,
        dst=out,
        xform=xform,
        flags=flags,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@pytest.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_xval, flags, bmode, border_value",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            7,
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [],
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            7,
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [1, 2, 3, 4],
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            5,
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [0],
        ),
        (
            3,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            4,
            cvcuda.Interp.CUBIC,
            cvcuda.Border.REPLICATE,
            [2, 1, 0],
        ),
        (
            4,
            cvcuda.Format.RGBf16,
            (16, 23),
            128.0,
            5,
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [0],
        ),
    ],
)
def test_op_warp_affinevarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    max_xval,
    flags,
    bmode,
    border_value,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    xform = util.create_tensor(
        (nimages, 6), np.float32, "NC", max_random=max_xval, rng=RNG
    )

    out = cvcuda.warp_affine(
        input, xform, flags, border_mode=bmode, border_value=border_value
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)
    tmp = cvcuda.warp_affine_into(
        src=input,
        dst=out,
        xform=xform,
        flags=flags,
        border_mode=bmode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def _warp_affine_params(dtype, layout, channels):
    return {
        "xform": [[1, 0, 0], [0, 1, 0]],
        "flags": cvcuda.Interp.NEAREST,
    }


def _warp_affine_varshape_params(dtype, layout, channels):
    return {
        "xform": util.create_tensor((2, 6), np.float32, "NC", max_random=1, rng=RNG),
        "flags": cvcuda.Interp.NEAREST,
        "border_mode": cvcuda.Border.CONSTANT,
        "border_value": np.array([], dtype=np.float32),
    }


globals().update(
    cv_tools.make_op_tests(
        name="warp_affine",
        runner_info=[
            ("tensor", cvcuda.warp_affine, _warp_affine_params),
            ("image_batch", cvcuda.warp_affine, _warp_affine_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.F16,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3, 4},
    )
)
