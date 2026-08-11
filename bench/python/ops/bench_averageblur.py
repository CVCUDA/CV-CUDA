#!/usr/bin/env python3
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

"""CV-CUDA AverageBlur operator benchmark - Python equivalent of BenchAverageBlur.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_num_channels,
    get_dtype_size,
    get_format_from_dtype,
    get_border_type,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def averageblur(state):
    """AverageBlur operator benchmark matching C++ BenchAverageBlur.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    kernel_size = parse_shape(state.get_string("kernelSize"))
    border = get_border_type(state.get_string("border"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    device_id = state.get_device()

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "AverageBlur benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None
    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) AverageBlur benchmark is tensor-only")
        return None

    # Report memory (match C++)
    dtype_size = get_dtype_size(dtype_str)
    state.add_global_memory_reads(N * H * W * dtype_size)
    state.add_global_memory_writes(N * H * W * dtype_size)

    get_stream = create_stream_cache()

    def do_averageblur(dst, src, stream):
        cvcuda.averageblur_into(
            dst,
            src,
            kernel_size,
            kernel_anchor=(-1, -1),
            border=border,
            stream=stream,
        )

    if is_fake_planar:
        src = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        inter_src = create_tensor(
            (N, H, W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        inter_dst = create_tensor(
            (N, H, W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        dst = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode=0,
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            do_averageblur(inter_dst, inter_src, stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        tensor_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        src = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode="checkerboard",
        )
        dst = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode=0,
        )

        def run(launch):
            do_averageblur(dst, src, get_stream(launch))

        return run

    else:  # ImageBatchVarShape mode (inputKind=VarShape)
        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode=0,
        )
        kernel_size_tensor = create_tensor(
            (N, 2),
            cvcuda.Type.S32,
            device_id,
            layout="NC",
            fill_mode=[kernel_size[0], kernel_size[1]],
        )
        kernel_anchor_tensor = create_tensor(
            (N, 2), cvcuda.Type.S32, device_id, layout="NC", fill_mode=[-1, -1]
        )

        def run(launch):
            cvcuda.averageblur_into(
                dst,
                src,
                kernel_size,
                kernel_size_tensor,
                kernel_anchor_tensor,
                border=border,
                stream=get_stream(launch),
            )

        return run


if __name__ == "__main__":
    run_benchmark("averageblur", averageblur)
