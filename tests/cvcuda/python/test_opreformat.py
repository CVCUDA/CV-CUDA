# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest as t
import numpy as np
import threading
import torch


RNG = np.random.default_rng(0)


@t.mark.parametrize(
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

    torch0 = torch.zeros(src_shape, dtype=torch.int32, device="cuda")
    torch1 = torch.zeros(src_shape, dtype=torch.int32, device="cuda")

    thread = threading.Thread(
        target=lambda: (torch.abs(torch0, out=torch1), torch.square(torch1, out=torch0))
    )
    thread.start()

    tmp = cvcuda.reformat_into(dst, src)
    assert tmp is dst
    assert dst.layout == dst_layout
    assert dst.dtype == src.dtype
    assert dst.shape == dst_shape

    thread.join()
    assert torch0.shape == src_shape
    assert torch1.shape == src_shape
