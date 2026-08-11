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
import threading
import cvcuda_types as cv_types
import cvcuda_tools as cv_tools
import cupy

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "input_args,out_shape,out_layout",
    [
        (((5, 16, 23, 4), np.uint8, "NHWC"), (5, 4, 16, 23), "NCHW"),
        (((5, 16, 23, 3), np.uint8, "NHWC"), (5, 3, 16, 23), "NCHW"),
        (((5, 3, 16, 23), np.uint8, "NCHW"), (5, 16, 23, 3), "NHWC"),
        (((3, 6, 4), np.uint8, "CHW"), (6, 4, 3), "HWC"),
        (((7, 5, 4), np.uint8, "HWC"), (4, 7, 5), "CHW"),
    ],
)
def test_op_reformat(input_args, out_shape, out_layout):
    input = cvcuda.Tensor(*input_args)
    out = cvcuda.reformat(input, out_layout)
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(out_shape, input.dtype, out_layout)
    tmp = cvcuda.reformat_into(out, input)
    assert tmp is out
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.reformat(src=input, layout=out_layout, stream=stream)
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    tmp = cvcuda.reformat_into(src=input, dst=out, stream=stream)
    assert tmp is out
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype


def test_op_reformat_gpuload():
    src_layout = "NHWC"
    dst_layout = "NCHW"
    src_shape = (2, 720, 1280, 3)
    dst_shape = (src_shape[0], src_shape[3], src_shape[1], src_shape[2])
    src = cvcuda.Tensor(src_shape, np.uint8, src_layout)
    dst = cvcuda.Tensor(dst_shape, np.uint8, dst_layout)

    cuda0 = cupy.asarray(np.zeros(src_shape, dtype=np.int32))
    cuda1 = cupy.asarray(np.zeros(src_shape, dtype=np.int32))

    thread = threading.Thread(
        target=lambda: (
            np.abs(cuda0.get(), out=cuda1.get()),
            np.square(cuda1.get(), out=cuda0.get()),
        )
    )
    thread.start()

    tmp = cvcuda.reformat_into(dst, src)
    assert tmp is dst
    assert dst.layout == dst_layout
    assert dst.dtype == src.dtype
    assert dst.shape == dst_shape

    thread.join()
    assert cuda0.shape == src_shape
    assert cuda1.shape == src_shape


_layout_conversion = {
    "NHWC": "NCHW",
    "NCHW": "NHWC",
    "HWC": "CHW",
    "CHW": "HWC",
}


def _reformat_params(dtype, layout, channels):
    inverse_layout = _layout_conversion.get(layout)
    return {
        "layout": inverse_layout if inverse_layout is not None else layout,
    }


globals().update(
    cv_tools.make_op_tests(
        name="reformat",
        runner_info=[("tensor", cvcuda.reformat, _reformat_params)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.S8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
            cvcuda.Type.F64,
        },
        supported_layouts={"NHWC", "NCHW", "HWC", "CHW"},
        supported_channels=cv_types.CHANNELS,
    )
)
