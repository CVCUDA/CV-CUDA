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
    "src_args, dst_dtype, args_setup",
    [
        (
            ((1, 16, 23, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.Type.U8,
            (
                cvcuda.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((5, 33, 28), cvcuda.Type._3U8, "NHWC"),
            cvcuda.Type._3U8,
            (
                cvcuda.Type.F32,
                (5, 5, 5, 5),
            ),
        ),
        (
            ((16, 23, 3), cvcuda.Type.U8, "HWC"),
            cvcuda.Type.U8,
            (
                cvcuda.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((33, 28), cvcuda.Type._3U8, "HWC"),
            cvcuda.Type._3U8,
            (
                cvcuda.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((2, 3, 16, 23), cvcuda.Type.U8, "NCHW"),
            cvcuda.Type.U8,
            (
                cvcuda.Type.F32,
                (2, 2, 1, 2),
            ),
        ),
        (
            ((3, 16, 23), cvcuda.Type.U8, "CHW"),
            cvcuda.Type.U8,
            (
                cvcuda.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((9, 16, 23, 3), cvcuda.Type.U16, "NHWC"),
            cvcuda.Type.U16,
            (
                cvcuda.Type.F32,
                (1, 9, 1, 9),
            ),
        ),
        (
            ((9, 16, 23), cvcuda.Type._4S16, "NHWC"),
            cvcuda.Type._4S16,
            (
                cvcuda.Type.F32,
                (9, 9, 9, 9),
            ),
        ),
        (
            ((13, 33, 28), cvcuda.Type._3U16, "NHWC"),
            cvcuda.Type._3U16,
            (
                cvcuda.Type.F32,
                (13, 1, 13, 1),
            ),
        ),
        (
            ((9, 16, 23, 4), cvcuda.Type.S32, "NHWC"),
            cvcuda.Type.S32,
            (
                cvcuda.Type.F64,
                (9, 9, 9, 1),
            ),
        ),
        (
            ((13, 33, 28), cvcuda.Type._4S32, "NHWC"),
            cvcuda.Type._4S32,
            (
                cvcuda.Type.F64,
                (1, 13, 13, 13),
            ),
        ),
        (
            ((17, 16, 23, 3), cvcuda.Type.F32, "NHWC"),
            cvcuda.Type.F32,
            (
                cvcuda.Type.F32,
                (17, 17, 17, 17),
            ),
        ),
        (
            ((21, 33, 28), cvcuda.Type._3F32, "NHWC"),
            cvcuda.Type._3F32,
            (
                cvcuda.Type.F32,
                (21, 21, 1, 21),
            ),
        ),
        (
            ((5, 4, 33, 28), cvcuda.Type.F32, "NCHW"),
            cvcuda.Type.F32,
            (
                cvcuda.Type.F32,
                (5, 5, 1, 5),
            ),
        ),
        (
            ((16, 23, 3), cvcuda.Type.F32, "HWC"),
            cvcuda.Type.F32,
            (
                cvcuda.Type.F32,
                (1, 1, 1, 1),
            ),
        ),
        (
            ((33, 28), cvcuda.Type._3F32, "HWC"),
            cvcuda.Type._3F32,
            (
                cvcuda.Type.F32,
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


@pytest.mark.parametrize(
    "num_images, src_format, src_dtype, max_size, dst_format, args_setup",
    [
        (
            1,
            cvcuda.Format.BGR8,
            np.uint8,
            (128, 128),
            cvcuda.Format.BGR8,
            (cvcuda.Type.F32, (1, 1, 1, 1)),
        ),
        (
            2,
            cvcuda.Format.BGR8,
            np.uint8,
            (128, 128),
            cvcuda.Format.BGRf32,
            (cvcuda.Type.F32, (2, 2, 2, 1)),
        ),
        (
            3,
            cvcuda.Format.BGRf32,
            np.float32,
            (128, 128),
            cvcuda.Format.BGR8,
            (cvcuda.Type.F32, (3, 3, 1, 3)),
        ),
        (
            4,
            cvcuda.Format.RGB8p,
            np.uint8,
            (128, 128),
            cvcuda.Format.RGB8p,
            (cvcuda.Type.F32, (4, 1, 4, 4)),
        ),
        (
            5,
            cvcuda.Format.RGB8p,
            np.uint8,
            (128, 128),
            cvcuda.Format.RGBf32p,
            (cvcuda.Type.F32, (1, 5, 5, 5)),
        ),
        (
            6,
            cvcuda.Format.RGBf32p,
            np.float32,
            (128, 128),
            cvcuda.Format.RGB8p,
            (cvcuda.Type.F32, (6, 6, 6, 6)),
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

    b_src = cvcuda.ImageBatchVarShape(num_images)
    for _ in range(num_images):
        h, w = max_size
        h = RNG.integers(1, h + 1)
        w = RNG.integers(1, w + 1)
        if src_format.planes == 1:
            shape = (h, w, src_format.channels)
            h_data = util.generate_data(shape, src_dtype, rng=RNG)
            image = cvcuda.as_image(util.to_cuda_buffer(h_data))
        else:
            shape = (h, w)
            planes_data = [
                util.generate_data(shape, src_dtype, rng=RNG)
                for _ in range(src_format.planes)
            ]
            image = cvcuda.as_image(
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


_SCALAR_ARGS = (0.75, 1.25, 0.125, 0.5)


def _scalar_arg_tensors(dtype=np.float32):
    return tuple(
        util.to_cvcuda_tensor(np.array([value], dtype), "N") for value in _SCALAR_ARGS
    )


@pytest.mark.parametrize(
    "dtype,arg_dtype,layout,shape",
    [
        (np.uint8, np.float32, "NHWC", (2, 3, 5, 3)),
        (np.float32, np.float32, "NCHW", (2, 3, 3, 5)),
        (np.int32, np.float64, "NHWC", (2, 3, 5, 1)),
    ],
)
def test_op_brightness_contrast_scalar_tensor_matches_tensor_parameters(
    dtype, arg_dtype, layout, shape
):
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    if np.issubdtype(dtype, np.floating):
        values /= np.prod(shape)
    src = util.to_cvcuda_tensor(
        values,
        layout,
    )
    arg_tensors = _scalar_arg_tensors(arg_dtype)

    ref = cvcuda.brightness_contrast(src, *arg_tensors)
    out = cvcuda.brightness_contrast(src, *_SCALAR_ARGS)
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(ref.cuda()), util.to_cpu_numpy_buffer(out.cuda())
    )

    ref = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    out = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    cvcuda.brightness_contrast_into(ref, src, *arg_tensors)
    ret = cvcuda.brightness_contrast_into(out, src, *_SCALAR_ARGS)
    assert ret is out
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(ref.cuda()), util.to_cpu_numpy_buffer(out.cuda())
    )


def test_op_brightness_contrast_scalar_varshape_matches_tensor_parameters():
    src = cvcuda.ImageBatchVarShape(2)
    for shape in ((3, 5, 3), (4, 2, 3)):
        src.pushback(
            util.to_cvcuda_image(
                np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
            )
        )
    arg_tensors = _scalar_arg_tensors()

    ref = cvcuda.brightness_contrast(src, *arg_tensors)
    out = cvcuda.brightness_contrast(src, *_SCALAR_ARGS)
    for ref_image, out_image in zip(ref, out):
        np.testing.assert_array_equal(
            util.to_cpu_numpy_buffer(ref_image.cuda()),
            util.to_cpu_numpy_buffer(out_image.cuda()),
        )

    ref = util.clone_image_batch(src)
    out = util.clone_image_batch(src)
    cvcuda.brightness_contrast_into(ref, src, *arg_tensors)
    ret = cvcuda.brightness_contrast_into(out, src, *_SCALAR_ARGS)
    assert ret is out
    for ref_image, out_image in zip(ref, out):
        np.testing.assert_array_equal(
            util.to_cpu_numpy_buffer(ref_image.cuda()),
            util.to_cpu_numpy_buffer(out_image.cuda()),
        )


def test_op_brightness_contrast_scalar_clamp():
    src = util.to_cvcuda_tensor(
        np.array([0, 0.25, 0.75, 1], np.float32).reshape(1, 1, 4, 1), "NHWC"
    )
    args = (2.0, 1.0, -0.25, 0.0)

    unclamped = cvcuda.brightness_contrast(src, *args, clamp=False)
    clamped = cvcuda.brightness_contrast(src, *args, clamp=True)
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(unclamped.cuda()).reshape(-1),
        [-0.25, 0.25, 1.25, 1.75],
    )
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(clamped.cuda()).reshape(-1), [0, 0.25, 1, 1]
    )

    signed_src = util.to_cvcuda_tensor(
        np.array([-10, 10], np.int32).reshape(1, 1, 2, 1), "NHWC"
    )
    signed_unclamped = cvcuda.brightness_contrast(
        signed_src, 1.0, 1.0, 0.0, 0.0, clamp=False
    )
    signed_clamped = cvcuda.brightness_contrast(
        signed_src, 1.0, 1.0, 0.0, 0.0, clamp=True
    )
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(signed_unclamped.cuda()).reshape(-1), [-10, 10]
    )
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(signed_clamped.cuda()).reshape(-1), [0, 10]
    )


def test_op_brightness_contrast_rejects_mixed_parameter_kinds():
    src = cvcuda.Tensor((1, 3, 5, 3), cvcuda.Type.F32, "NHWC")
    with pytest.raises(TypeError):
        cvcuda.brightness_contrast(src, _scalar_arg_tensors()[0], *_SCALAR_ARGS[1:])


def _brightness_contrast_scalar_op(data):
    return cvcuda.brightness_contrast(data, *_SCALAR_ARGS)


globals().update(
    cv_tools.make_op_tests(
        name="brightnesscontrast",
        runner_info=[
            ("tensor", cvcuda.brightness_contrast, None),
            ("image_batch", cvcuda.brightness_contrast, None),
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
        name="brightnesscontrast_scalar",
        runner_info=[
            ("tensor", _brightness_contrast_scalar_op, None),
            ("image_batch", _brightness_contrast_scalar_op, None),
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
