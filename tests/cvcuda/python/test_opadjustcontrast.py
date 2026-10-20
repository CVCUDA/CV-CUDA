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

import cvcuda

import pytest
import numpy as np

import cvcuda_tools as cv_tools
import cvcuda_util as util

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "tensor_params, contrast_factor",
    [
        (((5, 16, 23, 3), np.uint8, "NHWC"), 1.5),
        (((4, 9, 3), np.uint8, "HWC"), 0.5),
        (((3, 88, 13, 1), np.uint8, "NHWC"), 2.0),
        (((2, 16, 23, 3), np.float16, "NHWC"), 1.5),
        (((2, 3, 16, 23), np.float16, "NCHW"), 0.75),
        (((2, 3, 16, 23), np.float32, "NCHW"), 1.25),
        (((1, 8, 8), np.float32, "CHW"), 0.0),
    ],
)
def test_op_adjust_contrast(tensor_params, contrast_factor):
    src = cvcuda.Tensor(*tensor_params)

    out = cvcuda.adjust_contrast(src, contrast_factor)
    assert (out.layout, out.shape, out.dtype) == (src.layout, src.shape, src.dtype)

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp = cvcuda.adjust_contrast_into(
        src=src, dst=out, contrast_factor=contrast_factor, stream=stream
    )
    assert tmp is out
    assert (out.layout, out.shape, out.dtype) == (src.layout, src.shape, src.dtype)


@pytest.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, contrast_factor",
    [
        (10, cvcuda.Format.RGB8, (123, 321), 256, 1.5),
        (7, cvcuda.Format.RGBf32, (62, 35), 1.0, 0.75),
        (3, cvcuda.Format.RGBf16, (50, 40), 1.0, 1.5),
        (1, cvcuda.Format.U8, (33, 48), 256, 2.0),
    ],
)
def test_op_adjust_contrast_varshape(
    num_images, img_format, img_size, max_pixel, contrast_factor
):
    src = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    out = cvcuda.adjust_contrast(src, contrast_factor)
    assert (len(out), out.capacity, out.uniqueformat, out.maxsize) == (
        len(src),
        src.capacity,
        src.uniqueformat,
        src.maxsize,
    )

    stream = cvcuda.Stream()
    out = util.clone_image_batch(src)
    tmp = cvcuda.adjust_contrast_into(
        src=src, dst=out, contrast_factor=contrast_factor, stream=stream
    )
    assert tmp is out
    assert (len(out), out.capacity) == (len(src), src.capacity)


def _adjust_contrast_params(dtype, layout, channels, contrast_factor=1.5):
    return {"contrast_factor": contrast_factor}


globals().update(
    cv_tools.make_op_tests(
        name="adjust_contrast",
        runner_info=[
            ("tensor", cvcuda.adjust_contrast, _adjust_contrast_params),
            ("image_batch", cvcuda.adjust_contrast, _adjust_contrast_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8, cvcuda.Type.F16, cvcuda.Type.F32},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3},
        extra_params_negative={
            "contrast_factor": {
                -0.5,
                float("nan"),
            }
        },
    )
)


@pytest.mark.parametrize(
    "round_mode,expected", [(cvcuda.Round.NEAREST, 128), (cvcuda.Round.TRUNCATE, 127)]
)
@pytest.mark.parametrize(
    "shape,layout",
    [
        ((1, 2, 1, 1), "NHWC"),
        ((2, 32, 64, 1), "NHWC"),
        ((32, 64, 1), "HWC"),
        ((2, 1, 32, 64), "NCHW"),
        ((1, 32, 64), "CHW"),
    ],
)
def test_op_adjust_contrast_round_mode(round_mode, expected, shape, layout):
    h_src = np.resize(np.array([0, 255], dtype=np.uint8), shape)
    src = util.to_cvcuda_tensor(h_src, layout)

    # factor 0 collapses every pixel onto the mean, which is exactly 127.5.
    out = cvcuda.adjust_contrast(src, 0.0, round=round_mode)
    got = util.to_cpu_numpy_buffer(out.cuda())
    np.testing.assert_array_equal(got, np.full_like(h_src, expected))

    dst = util.to_cvcuda_tensor(np.zeros_like(h_src), layout)
    returned = cvcuda.adjust_contrast_into(
        dst=dst, src=src, contrast_factor=0.0, round=round_mode
    )
    assert returned is dst
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(dst.cuda()), np.full_like(h_src, expected)
    )


def test_op_adjust_contrast_round_mode_varshape():
    hosts = [np.array([[[0]], [[255]]], dtype=np.uint8)]
    src = cvcuda.ImageBatchVarShape(1)
    src.pushback(util.to_cvcuda_image(hosts[0]))

    for mode, expected in [(cvcuda.Round.NEAREST, 128), (cvcuda.Round.TRUNCATE, 127)]:
        out = cvcuda.adjust_contrast(src, 0.0, round=mode)
        np.testing.assert_array_equal(
            util.to_cpu_numpy_buffer(next(iter(out)).cuda()),
            np.full(hosts[0].shape[:2], expected, dtype=np.uint8),
        )
        dst = util.clone_image_batch(src)
        assert cvcuda.adjust_contrast_into(dst, src, 0.0, round=mode) is dst
        np.testing.assert_array_equal(
            util.to_cpu_numpy_buffer(next(iter(dst)).cuda()),
            np.full(hosts[0].shape[:2], expected, dtype=np.uint8),
        )


def test_op_adjust_contrast_rejects_wrap_round_mode():
    src = cvcuda.Tensor((1, 2, 2, 1), np.uint8, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.adjust_contrast(src, 1.0, round=cvcuda.Round.NEAREST_WRAP)


def test_op_adjust_contrast_round_mode_defaults_to_nearest():
    h_src = np.array([[[[0]], [[255]]]], dtype=np.uint8)
    src = util.to_cvcuda_tensor(h_src, "NHWC")

    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(cvcuda.adjust_contrast(src, 0.0).cuda()),
        util.to_cpu_numpy_buffer(
            cvcuda.adjust_contrast(src, 0.0, round=cvcuda.Round.NEAREST).cuda()
        ),
    )
