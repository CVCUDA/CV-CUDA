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

from random import randint

import cupy
import cvcuda
import pytest
import cvcuda_tools as cv_tools
import cvcuda_types as cv_types


def _zeroed(tensor):
    # cvcuda.Tensor leaves device memory uninitialized; erase validates its anchor, so zero the
    # parameter tensors to give it well-defined (no-op) values instead of whatever the allocator
    # pool happens to hold.
    cupy.asarray(tensor.cuda())[...] = 0
    return tensor


@pytest.mark.parametrize(
    "input_args, erasing_area_num, random, seed",
    [
        (((1, 460, 640, 3), cvcuda.Type.U8, "NHWC"), 1, False, 0),
        (((5, 460, 640, 3), cvcuda.Type.U8, "NHWC"), 1, True, 1),
    ],
)
def test_op_erase(input_args, erasing_area_num, random, seed):
    input = cvcuda.Tensor(*input_args)

    parameter_shape = (erasing_area_num,)
    values_shape = (erasing_area_num * input_args[0][-1],)
    anchor = _zeroed(cvcuda.Tensor(parameter_shape, cvcuda.Type._2S32, "N"))
    erasing = _zeroed(cvcuda.Tensor(parameter_shape, cvcuda.Type._3S32, "N"))
    imgIdx = _zeroed(cvcuda.Tensor(parameter_shape, cvcuda.Type.S32, "N"))
    values = _zeroed(cvcuda.Tensor(values_shape, cvcuda.Type.F32, "N"))

    out = cvcuda.erase(input, anchor, erasing, values, imgIdx)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.erase_into(out, input, anchor, erasing, values, imgIdx)
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.erase(
        src=input,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.erase_into(
        src=input,
        dst=out,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp is out


@pytest.mark.parametrize(
    "shape,dtype,layout,channels",
    [
        ((2, 7, 9, 3), cvcuda.Type.U8, "NHWC", 3),
        ((7, 9, 4), cvcuda.Type.F32, "HWC", 4),
        ((2, 2, 7, 9), cvcuda.Type.U8, "NCHW", 2),
        ((1, 7, 9), cvcuda.Type.F32, "CHW", 1),
    ],
)
def test_op_erase_region_overload(shape, dtype, layout, channels):
    input = cvcuda.Tensor(shape, dtype, layout)
    values = cvcuda.Tensor((channels, 2, 3), cvcuda.Type.F32, "CHW")
    cupy.asarray(input.cuda())[...] = 4
    cupy.asarray(values.cuda())[...] = 7

    out = cvcuda.erase(input, 2, 3, 2, 3, values)
    assert out.shape == input.shape
    assert out.dtype == input.dtype
    assert out.layout == input.layout

    dst = cvcuda.Tensor(shape, dtype, layout)
    tmp = cvcuda.erase_into(dst, input, 2, 3, 2, 3, values)
    assert tmp is dst

    tmp = cvcuda.erase_into(input, input, 2, 3, 2, 3, values)
    assert tmp is input


def test_op_erase_region_rejects_invalid_value_dtype():
    input = cvcuda.Tensor((1, 7, 9, 3), cvcuda.Type.U8, "NHWC")
    values = cvcuda.Tensor((1,), cvcuda.Type.S16, "W")

    with pytest.raises(RuntimeError, match="values must match"):
        cvcuda.erase(input, 2, 3, 2, 3, values)


@pytest.mark.parametrize(
    "dtype",
    [
        cvcuda.Type.U8,
        cvcuda.Type.S8,
        cvcuda.Type.U16,
        cvcuda.Type.S16,
        cvcuda.Type.U32,
        cvcuda.Type.S32,
        cvcuda.Type.U64,
        cvcuda.Type.S64,
    ],
    ids=str,
)
def test_op_erase_region_float32_cast_matches_torch(dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch is required")

    torch_dtype = cv_types.TYPE_TO_TORCH_DTYPE.get(dtype)
    if torch_dtype is None:
        pytest.skip(f"{dtype} is not supported by this PyTorch version")

    values = torch.tensor(
        [
            -float("inf"),
            -1.0e20,
            -256.9,
            -1.9,
            float("nan"),
            0.0,
            1.9,
            255.9,
            256.1,
            1.0e20,
            float("inf"),
        ],
        dtype=torch.float32,
        device="cuda",
    )
    source = torch.zeros((1, 1, 1, values.numel()), dtype=torch_dtype, device="cuda")
    expected = source.clone()
    expected[...] = values

    actual = cvcuda.erase(
        cvcuda.as_tensor(source, "NCHW"),
        0,
        0,
        1,
        values.numel(),
        cvcuda.as_tensor(values, "W"),
    )

    assert torch.equal(torch.as_tensor(actual.cuda(), device=expected.device), expected)


@pytest.mark.parametrize(
    "num_images, format, min_size, max_size, erasing_area_num, random, seed",
    [
        (1, cvcuda.Format.U8, (100, 100), (200, 200), 1, False, 0),
        (5, cvcuda.Format.RGB8, (100, 100), (200, 100), 1, True, 1),
    ],
)
def test_op_erase_varshape(
    num_images, format, min_size, max_size, erasing_area_num, random, seed
):

    parameter_shape = (erasing_area_num,)
    values_shape = (erasing_area_num * format.channels,)
    anchor = _zeroed(cvcuda.Tensor(parameter_shape, cvcuda.Type._2S32, "N"))
    erasing = _zeroed(cvcuda.Tensor(parameter_shape, cvcuda.Type._3S32, "N"))
    imgIdx = _zeroed(cvcuda.Tensor(parameter_shape, cvcuda.Type.S32, "N"))
    values = _zeroed(cvcuda.Tensor(values_shape, cvcuda.Type.F32, "N"))

    input = cvcuda.ImageBatchVarShape(num_images)
    output = cvcuda.ImageBatchVarShape(num_images)
    for i in range(num_images):
        w = randint(min_size[0], max_size[0])
        h = randint(min_size[1], max_size[1])
        img_in = cvcuda.Image([w, h], format)
        input.pushback(img_in)
        img_out = cvcuda.Image([w, h], format)
        output.pushback(img_out)

    tmp = cvcuda.erase(input, anchor, erasing, values, imgIdx)
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.erase_into(
        output, input, anchor, erasing, values, imgIdx, random=random, seed=seed
    )
    assert tmp is output

    stream = cvcuda.Stream()
    tmp = cvcuda.erase(
        src=input,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.erase_into(
        src=input,
        dst=output,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp is output


def _erase_params(dtype, layout, channels):
    return {
        "anchor": _zeroed(cvcuda.Tensor((1,), cvcuda.Type._2S32, "N")),
        "erasing": _zeroed(cvcuda.Tensor((1,), cvcuda.Type._3S32, "N")),
        "values": _zeroed(cvcuda.Tensor((channels,), cvcuda.Type.F32, "N")),
        "imgIdx": _zeroed(cvcuda.Tensor((1,), cvcuda.Type.S32, "N")),
    }


def _erase_region_params(dtype, layout, channels):
    return {
        "i": 0,
        "j": 0,
        "h": 1,
        "w": 1,
        "v": _zeroed(cvcuda.Tensor((channels, 1, 1), dtype, "CHW")),
    }


globals().update(
    cv_tools.make_op_tests(
        name="erase",
        runner_info=[
            ("tensor", cvcuda.erase, _erase_params),
            ("image_batch", cvcuda.erase, _erase_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        exclude_dlc=[(None, "NCHW", 2), (None, "CHW", 2)],
    )
)


globals().update(
    cv_tools.make_op_tests(
        name="erase_region",
        runner_info=[
            ("tensor", cvcuda.erase, _erase_region_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.S8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.U32,
            cvcuda.Type.S32,
            cvcuda.Type.U64,
            cvcuda.Type.S64,
            cvcuda.Type.F16,
            cvcuda.Type.F32,
            cvcuda.Type.F64,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
    )
)
