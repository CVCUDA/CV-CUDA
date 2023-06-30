# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvcv
import cvcuda
import pytest as t
import cvcuda_util as util
import numpy as np

RNG = np.random.default_rng(12345)


@t.mark.parametrize(
    "src_args, dst_dtype, args_setup",
    [
        (
            ((1, 16, 23, 3), nvcv.Type.U8, "NHWC"),
            nvcv.Type.U8,
            (
                nvcv.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((5, 33, 28), nvcv.Type._3U8, "NHWC"),
            nvcv.Type._3U8,
            (
                nvcv.Type.F32,
                (5, 5, 5, 5),
            ),
        ),
        (
            ((16, 23, 3), nvcv.Type.U8, "HWC"),
            nvcv.Type.U8,
            (
                nvcv.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((33, 28), nvcv.Type._3U8, "HWC"),
            nvcv.Type._3U8,
            (
                nvcv.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((9, 16, 23, 3), nvcv.Type.U16, "NHWC"),
            nvcv.Type.U16,
            (
                nvcv.Type.F32,
                (1, 9, 1, 9),
            ),
        ),
        (
            ((9, 16, 23), nvcv.Type._4S16, "NHWC"),
            nvcv.Type._4S16,
            (
                nvcv.Type.F32,
                (9, 9, 9, 9),
            ),
        ),
        (
            ((13, 33, 28), nvcv.Type._3U16, "NHWC"),
            nvcv.Type._3U16,
            (
                nvcv.Type.F32,
                (13, 1, 13, 1),
            ),
        ),
        (
            ((9, 16, 23, 4), nvcv.Type.S32, "NHWC"),
            nvcv.Type.S32,
            (
                nvcv.Type.F64,
                (9, 9, 9, 1),
            ),
        ),
        (
            ((13, 33, 28), nvcv.Type._4S32, "NHWC"),
            nvcv.Type._4S32,
            (
                nvcv.Type.F64,
                (1, 13, 13, 13),
            ),
        ),
        (
            ((17, 16, 23, 3), nvcv.Type.F32, "NHWC"),
            nvcv.Type.F32,
            (
                nvcv.Type.F32,
                (17, 17, 17, 17),
            ),
        ),
        (
            ((21, 33, 28), nvcv.Type._3F32, "NHWC"),
            nvcv.Type._3F32,
            (
                nvcv.Type.F32,
                (21, 21, 1, 21),
            ),
        ),
        (
            ((16, 23, 3), nvcv.Type.F32, "HWC"),
            nvcv.Type.F32,
            (
                nvcv.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((33, 28), nvcv.Type._3F32, "HWC"),
            nvcv.Type._3F32,
            (
                nvcv.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
    ],
)
def test_op_brightness_contrast_api(src_args, dst_dtype, args_setup):
    stream = cvcuda.Stream()

    shape, src_dtype, layout = src_args
    t_src = util.create_tensor(shape, src_dtype, layout)
    arg_dtype, (b_num, c_num, bs_num, cc_num) = args_setup
    brightness = util.create_tensor((b_num,), arg_dtype, "N")
    contrast = util.create_tensor((c_num,), arg_dtype, "N")
    brightness_shift = util.create_tensor((bs_num,), arg_dtype, "N")
    contrast_center = util.create_tensor((cc_num,), arg_dtype, "N")

    all_kwargs = (
        ("brightness", brightness),
        ("contrast", contrast),
        ("brightness_shift", brightness_shift),
        ("contrast_center", contrast_center),
    )
    for i in range(len(all_kwargs) + 1):
        if i < len(all_kwargs):
            kwargs_case = dict([all_kwargs[i]])
        else:
            kwargs_case = dict(all_kwargs)

        t_dst = cvcuda.brightness_contrast(src=t_src, stream=stream, **kwargs_case)
        assert t_dst.layout == t_src.layout
        assert t_dst.dtype == t_src.dtype
        assert t_dst.shape == t_src.shape

        t_dst = util.create_tensor(shape, dst_dtype, layout)
        t_tmp = cvcuda.brightness_contrast_into(
            t_dst,
            t_src,
            **kwargs_case,
        )
        assert t_tmp is t_dst


@t.mark.parametrize(
    "num_images, src_format, src_dtype, max_size, dst_format, args_setup",
    [
        (
            1,
            nvcv.Format.BGR8,
            np.uint8,
            (128, 128),
            nvcv.Format.BGR8,
            (nvcv.Type.F32, (1, 1, 1, 1)),
        ),
        (
            2,
            nvcv.Format.BGR8,
            np.uint8,
            (128, 128),
            nvcv.Format.BGRf32,
            (nvcv.Type.F32, (2, 2, 2, 1)),
        ),
        (
            3,
            nvcv.Format.BGRf32,
            np.float32,
            (128, 128),
            nvcv.Format.BGR8,
            (nvcv.Type.F32, (3, 3, 1, 3)),
        ),
        (
            4,
            nvcv.Format.RGB8p,
            np.uint8,
            (128, 128),
            nvcv.Format.RGB8p,
            (nvcv.Type.F32, (4, 1, 4, 4)),
        ),
        (
            5,
            nvcv.Format.RGB8p,
            np.uint8,
            (128, 128),
            nvcv.Format.RGBf32p,
            (nvcv.Type.F32, (1, 5, 5, 5)),
        ),
        (
            6,
            nvcv.Format.RGBf32p,
            np.float32,
            (128, 128),
            nvcv.Format.RGB8p,
            (nvcv.Type.F32, (6, 6, 6, 6)),
        ),
    ],
)
def test_op_brightnesscontrastvarshape_api(
    num_images, src_format, src_dtype, max_size, dst_format, args_setup
):
    stream = cvcuda.Stream()

    arg_dtype, (b_num, c_num, bs_num, cc_num) = args_setup
    brightness = util.create_tensor((b_num,), arg_dtype, "N")
    contrast = util.create_tensor((c_num,), arg_dtype, "N")
    brightness_shift = util.create_tensor((bs_num,), arg_dtype, "N")
    contrast_center = util.create_tensor((cc_num,), arg_dtype, "N")

    b_src = nvcv.ImageBatchVarShape(num_images)
    for _ in range(num_images):
        h, w = max_size
        h = RNG.integers(1, h + 1)
        w = RNG.integers(1, w + 1)
        if src_format.planes == 1:
            shape = (h, w, src_format.channels)
            h_data = util.generate_data(shape, src_dtype, rng=RNG)
            # image = util.to_nvcv_image(h_data, format=src_format)
            image = nvcv.as_image(util.to_cuda_buffer(h_data))
        else:
            shape = (h, w)
            planes_data = [
                util.generate_data(shape, src_dtype, rng=RNG)
                for _ in range(src_format.planes)
            ]
            # image = util.to_nvcv_image(h_data, format=src_format)
            image = nvcv.as_image(
                [util.to_cuda_buffer(plane_data) for plane_data in planes_data],
                format=src_format,
            )
        b_src.pushback(image)

    all_kwargs = (
        ("brightness", brightness),
        ("contrast", contrast),
        ("brightness_shift", brightness_shift),
        ("contrast_center", contrast_center),
    )
    for i in range(len(all_kwargs) + 1):
        if i < len(all_kwargs):
            kwargs_case = dict([all_kwargs[i]])
        else:
            kwargs_case = dict(all_kwargs)

        b_dst = cvcuda.brightness_contrast(
            src=b_src,
            **kwargs_case,
            stream=stream,
        )

        assert len(b_dst) == len(b_src)
        assert b_dst.capacity == b_src.capacity
        assert b_dst.uniqueformat == b_src.uniqueformat
        assert b_dst.maxsize == b_src.maxsize

        b_dst = util.clone_image_batch(b_src, dst_format)
        b_tmp = cvcuda.brightness_contrast_into(
            src=b_src,
            dst=b_dst,
            **kwargs_case,
            stream=stream,
        )
        assert b_dst is b_tmp
