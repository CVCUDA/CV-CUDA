# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvcv
import cvcuda
import torch

# NOTE: The following tests for resize_crop_convert_reformat DO NOT TEST:
#       1. The correctness of the output data
#       2. Whether the format conversion actually worked correctly w.r.t. the data
#       3. Whether the channel swapping actually worked correctly w.r.t. the data


@t.mark.parametrize(
    "tensor_params, resize_dim, resize_interpolation, crop_rect_params, "
    "out_layout, out_dtype, manip, out_expected_shape, scale_norm, offset_norm, "
    "is_positive_test",
    [
        (
            ((4, 512, 512, 3), np.uint8, "NHWC"),  # Basic test
            (256, 256),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 224),
            "NCHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (4, 3, 224, 224),
            1,
            0,
            True,
        ),
        (
            ((4, 512, 512, 3), np.uint8, "NHWC"),
            (256, 256),
            cvcuda.Interp.NEAREST,  # With NEAREST Interpolation
            (0, 0, 224, 224),
            "NCHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (4, 3, 224, 224),
            1,
            0,
            True,
        ),
        (
            ((4, 512, 512, 3), np.uint8, "NHWC"),
            (256, 256),
            cvcuda.Interp.NEAREST,
            (0, 0, 224, 224),
            "",  # Empty output layout means keep the same as input
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (4, 224, 224, 3),
            1,
            0,
            True,
        ),
        (
            ((4, 512, 512, 3), np.uint8, "NHWC"),
            (256, 256),
            cvcuda.Interp.NEAREST,
            (0, 0, 224, 224),
            "",  # Empty output layout means keep the same layout as input
            0,  # Zero means keep the same dtype as input
            cvcuda.ChannelManip.REVERSE,
            (4, 224, 224, 3),
            1,
            0,
            True,
        ),
        (
            ((17, 678, 1027, 3), np.uint8, "NHWC"),  # Odd sizes
            (251, 256),  # Odd sizes
            cvcuda.Interp.LINEAR,
            (0, 0, 200, 22),
            "NCHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (17, 3, 22, 200),
            1,
            0,
            True,
        ),
        (
            ((17, 678, 1027, 3), np.uint8, "NHWC"),
            (251, 256),
            cvcuda.Interp.LINEAR,
            (0, 0, 200, 22),
            "NHWC",  # Same output layout as the input tensor
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (17, 22, 200, 3),
            1,
            0,
            True,
        ),
        (
            ((3, 40, 20, 3), np.uint8, "NHWC"),
            (160, 160),
            cvcuda.Interp.NEAREST,
            (10, 20, 20, 35),
            "NCHW",
            nvcv.Type.U8,  # Same dtype as the input tensor
            cvcuda.ChannelManip.NO_OP,  # No op here
            (3, 3, 35, 20),
            1,
            0,
            True,
        ),
        (
            ((512, 512, 3), np.uint8, "HWC"),  # Single image case.
            (256, 256),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 224),
            "CHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (3, 224, 224),
            1,
            0,
            True,
        ),
        (
            ((3, 512, 512), np.uint8, "CHW"),  # Unsupported input CHW
            (256, 256),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 224),
            "CHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (3, 224, 224),
            1,
            0,
            False,  # Negative test
        ),
        (
            ((512, 1024, 3), np.uint8, "HWC"),  # Large sizes
            (1024, 256),  # Unchanged resize width
            cvcuda.Interp.LINEAR,
            (0, 0, 1024, 224),  # Unchanged crop width
            "CHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.NO_OP,
            (3, 224, 1024),
            1,
            0,
            True,
        ),
        (
            ((4, 678, 1027, 3), np.uint8, "NHWC"),
            (251, 256),
            cvcuda.Interp.LINEAR,
            (0, 0, 200, 22),
            "NCHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (4, 3, 22, 200),
            127.5,  # Normalize output to [-1:1]
            -1,
            True,
        ),
        (
            ((512, 1024, 3), np.float32, "HWC"),  # Unsupported input dtype
            (1024, 256),
            cvcuda.Interp.LINEAR,
            (0, 0, 1024, 224),
            "CHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (3, 224, 1024),
            1,
            0,
            False,  # Negative test
        ),
        (
            ((1, 2, 3), np.uint8, "HWC"),  # Very small sizes
            (60, 5),
            cvcuda.Interp.LINEAR,
            (0, 0, 59, 5),
            "CHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (3, 5, 59),
            1,
            0,
            True,
        ),
        (
            ((1, 2, 3), np.uint8, "HWC"),
            (60, 5),
            cvcuda.Interp.LINEAR,
            (0, 0, 61, 5),  # Out of range crop.
            "CHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (3, 5, 59),
            1,
            0,
            False,  # Negative test
        ),
        (
            ((4, 512, 512, 3), np.uint8, "NHWC"),
            (256, 256),
            cvcuda.Interp.AREA,  # With Area Interpolation
            (0, 0, 224, 224),
            "NCHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (4, 3, 224, 224),
            1,
            0,
            False,  # Negative test
        ),
    ],
)
def test_op_resize_crop_convert_reformat(
    tensor_params,
    resize_dim,
    resize_interpolation,
    crop_rect_params,
    out_layout,
    out_dtype,
    manip,
    out_expected_shape,
    scale_norm,
    offset_norm,
    is_positive_test,
):

    inputTensor = cvcuda.Tensor(*tensor_params)
    out_layout = out_layout if out_layout else str(inputTensor.layout)
    out_dtype = out_dtype if out_dtype else inputTensor.dtype

    try:
        out1 = cvcuda.resize_crop_convert_reformat(
            inputTensor,
            resize_dim,
            resize_interpolation,
            cvcuda.RectI(*crop_rect_params),
            layout=out_layout,
            data_type=out_dtype,
            manip=manip,
            scale=scale_norm,
            offset=offset_norm,
        )
    except Exception as e:
        if is_positive_test:
            raise e
        else:
            # This is pass for a negative test.
            pass

    if is_positive_test:
        assert out1.layout == out_layout
        assert out1.shape == out_expected_shape
        assert out1.dtype == out_dtype

    out2 = cvcuda.Tensor(out_expected_shape, out_dtype, out_layout)

    try:
        tmp = cvcuda.resize_crop_convert_reformat_into(
            out2,
            inputTensor,
            resize_dim,
            resize_interpolation,
            [crop_rect_params[1], crop_rect_params[0]],
            manip=manip,
            scale=scale_norm,
            offset=offset_norm,
        )
    except Exception as e:
        if is_positive_test:
            raise e
        else:
            # This is pass for a negative test.
            pass

    if is_positive_test:
        assert tmp is out2
        assert out2.layout == out_layout
        assert out2.shape == out_expected_shape
        assert out2.dtype == out_dtype

    # Compare the two
    if is_positive_test:
        out1 = torch.as_tensor(out1.cuda())
        out2 = torch.as_tensor(out2.cuda())
        assert torch.equal(out1, out2)


@t.mark.parametrize(
    "num_images, min_size, max_size, resize_dim, resize_interpolation, crop_rect_params, "
    "out_layout, out_dtype, manip, out_expected_shape, scale_norm, offset_norm, is_positive_test",
    [
        (
            10,  # Basic test
            (50, 50),
            (512, 512),
            (256, 256),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 224),
            "NCHW",
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (10, 3, 224, 224),
            1,
            0,
            True,
        ),
        (
            1,  # Only one image
            (500, 300),
            (800, 700),  # Bigger sizes
            (400, 200),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 190),
            "NHWC",  # Same output layout as the input
            nvcv.Type.F32,
            cvcuda.ChannelManip.REVERSE,
            (1, 190, 224, 3),
            1,
            0,
            True,
        ),
        (
            50,  # More images
            (1, 1),  # Very small image
            (50, 70),
            (400, 200),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 190),
            "NHWC",
            nvcv.Type.F32,
            cvcuda.ChannelManip.NO_OP,  # No channels swapping
            (50, 190, 224, 3),
            1,
            0,
            True,
        ),
        (
            50,
            (1, 1),
            (50, 70),
            (400, 200),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 190),
            "NCHW",
            nvcv.Type.U8,  # Same uint8 dtype as the input
            cvcuda.ChannelManip.REVERSE,
            (50, 3, 190, 224),
            1,
            0,
            True,
        ),
        (
            50,
            (1, 1),
            (50, 70),
            (400, 200),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 190),
            "NCHW",
            nvcv.Type.U8,
            cvcuda.ChannelManip.NO_OP,  # NO_OP
            (50, 3, 190, 224),
            1,
            0,
            True,
        ),
        (
            50,
            (1, 1),
            (50, 70),
            (400, 200),
            cvcuda.Interp.LINEAR,
            (0, 0, 224, 190),
            "",  # Same uint8 dtype as the input
            0,  # Same uint8 dtype as the input
            cvcuda.ChannelManip.REVERSE,
            (50, 190, 224, 3),
            1,
            0,
            True,
        ),
    ],
)
def test_op_resize_crop_convert_reformat_varshape(
    num_images,
    min_size,
    max_size,
    resize_dim,
    resize_interpolation,
    crop_rect_params,
    out_layout,
    out_dtype,
    manip,
    out_expected_shape,
    scale_norm,
    offset_norm,
    is_positive_test,
):

    inputVarShape = cvcuda.ImageBatchVarShape(num_images)
    out_layout = out_layout if out_layout else "NHWC"
    out_dtype = out_dtype if out_dtype else nvcv.Type.U8

    inputVarShape.pushback(
        [
            cvcuda.Image(
                (
                    min_size[0] + (max_size[0] - min_size[0]) * i // num_images,
                    min_size[1] + (max_size[1] - min_size[1]) * i // num_images,
                ),
                cvcuda.Format.RGB8,
            )
            for i in range(num_images)
        ]
    )

    try:
        out1 = cvcuda.resize_crop_convert_reformat(
            inputVarShape,
            resize_dim=resize_dim,
            interp=resize_interpolation,
            crop_rect=cvcuda.RectI(*crop_rect_params),
            layout=out_layout,
            data_type=out_dtype,
            manip=manip,
            scale=scale_norm,
            offset=offset_norm,
        )
    except Exception as e:
        if is_positive_test:
            raise e
        else:
            # This is pass for a negative test.
            pass

    if is_positive_test:
        assert out1.layout == out_layout
        assert out1.shape == out_expected_shape
        assert out1.dtype == out_dtype

    out2 = cvcuda.Tensor(out_expected_shape, out_dtype, out_layout)

    try:
        tmp = cvcuda.resize_crop_convert_reformat_into(
            out2,
            inputVarShape,
            resize_dim,
            resize_interpolation,
            [crop_rect_params[1], crop_rect_params[0]],
            manip=manip,
        )
    except Exception as e:
        if is_positive_test:
            raise e
        else:
            # This is pass for a negative test.
            pass

    if is_positive_test:
        assert tmp is out2
        assert out2.layout == out_layout
        assert out2.shape == out_expected_shape
        assert out2.dtype == out_dtype

    if is_positive_test:
        # Compare the two
        out1 = torch.as_tensor(out1.cuda())
        out2 = torch.as_tensor(out2.cuda())
        assert torch.equal(out1, out2)
