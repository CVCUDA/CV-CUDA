# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cvcuda_util as util
import numpy as np
import cvcuda_tools as cv_tools

RNG = np.random.default_rng(12345)


@pytest.mark.parametrize(
    "src_args, twist_args",
    [
        (
            ((1, 16, 23, 3), cvcuda.Type.U8, "NHWC"),
            ((3,), cvcuda.Type._4F32, "HW"),
        ),
        (
            ((5, 33, 28), cvcuda.Type._3U8, "NHWC"),
            ((5, 3), cvcuda.Type._4F32, "NHW"),
        ),
        (
            ((16, 23, 3), cvcuda.Type.U8, "HWC"),
            ((3, 4), cvcuda.Type.F32, "HW"),
        ),
        (
            ((33, 28), cvcuda.Type._3U8, "HWC"),
            ((3, 4), cvcuda.Type.F32, "HW"),
        ),
        (
            ((9, 16, 23, 3), cvcuda.Type.U16, "NHWC"),
            ((3,), cvcuda.Type._4F32, "HW"),
        ),
        (
            ((9, 16, 23), cvcuda.Type._4S16, "NHWC"),
            ((3,), cvcuda.Type._4F32, "HW"),
        ),
        (
            ((13, 33, 28), cvcuda.Type._3U16, "NHWC"),
            ((13, 3), cvcuda.Type._4F32, "NHW"),
        ),
        (
            ((9, 16, 23, 4), cvcuda.Type.S32, "NHWC"),
            ((3,), cvcuda.Type._4F64, "HW"),
        ),
        (
            ((13, 33, 28), cvcuda.Type._4S32, "NHWC"),
            ((13, 3), cvcuda.Type._4F64, "NHW"),
        ),
        (
            ((17, 16, 23, 3), cvcuda.Type.F32, "NHWC"),
            ((3, 4), cvcuda.Type.F32, "HW"),
        ),
        (
            ((21, 33, 28), cvcuda.Type._3F32, "NHWC"),
            ((21, 3, 4), cvcuda.Type.F32, "NHW"),
        ),
        (
            ((16, 23, 3), cvcuda.Type.F32, "HWC"),
            ((3,), cvcuda.Type._4F32, "HW"),
        ),
        (
            ((33, 28), cvcuda.Type._3F32, "HWC"),
            ((3,), cvcuda.Type._4F32, "HW"),
        ),
    ],
)
def test_op_remap_api(src_args, twist_args):
    stream = cvcuda.Stream()

    t_src = util.create_tensor(*src_args)
    twist = util.create_tensor(*twist_args)

    t_dst = cvcuda.color_twist(
        src=t_src,
        twist=twist,
        stream=stream,
    )
    assert t_dst.layout == t_src.layout
    assert t_dst.dtype == t_src.dtype
    assert t_dst.shape == t_src.shape

    t_dst = cvcuda.Tensor(t_src.shape, t_src.dtype, t_src.layout)
    t_tmp = cvcuda.color_twist_into(
        t_dst,
        t_src,
        twist=twist,
    )
    assert t_tmp is t_dst


@pytest.mark.parametrize(
    "num_images, dtype, max_size, twist_args",
    [
        (4, np.uint8, (73, 98), ((3,), cvcuda.Type._4F32, "HW")),
        (11, np.uint16, (33, 32), ((11, 3), cvcuda.Type._4F32, "NHW")),
        (8, np.int16, (13, 42), ((3, 4), cvcuda.Type.F32, "HW")),
        (3, np.float32, (53, 68), ((3, 3, 4), cvcuda.Type.F32, "NHW")),
    ],
)
def test_op_colortwistvarshape_api(num_images, dtype, max_size, twist_args):
    stream = cvcuda.Stream()

    b_src = cvcuda.ImageBatchVarShape(num_images)
    for _ in range(num_images):
        w = RNG.integers(1, max_size[0] + 1)
        h = RNG.integers(1, max_size[1] + 1)
        shape = (h, w, 3)
        h_data = util.generate_data(shape, dtype, rng=RNG)
        image = util.to_cvcuda_image(h_data)
        b_src.pushback(image)

    twist = util.create_tensor(*twist_args)
    b_dst = cvcuda.color_twist(
        src=b_src,
        twist=twist,
        stream=stream,
    )

    assert len(b_dst) == len(b_src)
    assert b_dst.capacity == b_src.capacity
    assert b_dst.uniqueformat == b_src.uniqueformat
    assert b_dst.maxsize == b_src.maxsize

    b_dst = util.clone_image_batch(b_src)
    b_tmp = cvcuda.color_twist_into(
        dst=b_dst,
        src=b_src,
        twist=twist,
        stream=stream,
    )
    assert b_dst is b_tmp


def _colortwist_params(dtype, layout, channels):
    twist_dtype = cvcuda.Type.F32
    if dtype in {cvcuda.Type.U32, cvcuda.Type.S32}:
        twist_dtype = cvcuda.Type.F64
    return {"twist": cvcuda.Tensor((3, 4), twist_dtype, "HW")}


def _colortwist_varshape_params(dtype, layout, channels):
    np_dtype = np.float64 if dtype in {cvcuda.Type.U32, cvcuda.Type.S32} else np.float32
    twist_data = np.zeros((2, 3, 4), dtype=np_dtype)
    for i in range(3):
        twist_data[:, i, i] = 1.0
    return {"twist": util.to_cvcuda_tensor(twist_data, "NHW")}


globals().update(
    cv_tools.make_op_tests(
        name="colortwist",
        runner_info=[
            ("tensor", cvcuda.color_twist, _colortwist_params),
            ("image_batch", cvcuda.color_twist, _colortwist_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.U32,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={3, 4},
    )
)
