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

"""CV-CUDA ColorTwist operator benchmark - Python equivalent of BenchColorTwist.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_num_channels,
    get_format_from_dtype,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def colortwist(state):
    """ColorTwist operator benchmark matching C++ BenchColorTwist.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    input_kind = get_input_kind(state.get_string("inputKind"))
    twist_mode = state.get_string("twistMode")
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    device_id = state.get_device()

    num_channels = get_num_channels(dtype_str)

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "ColorTwist benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) ColorTwist benchmark is tensor-only")
        return None
    if is_planar and input_kind == "VarShape" and dtype_str == "uchar4":
        state.skip("RGBA8p varshape is unsupported by the Python image API")
        return None

    dtype_size = get_dtype_size(dtype_str)
    image_bytes = N * H * W * dtype_size
    if is_fake_planar:
        state.add_global_memory_reads(3 * image_bytes)
        state.add_global_memory_writes(3 * image_bytes)
    else:
        state.add_global_memory_reads(image_bytes)
        state.add_global_memory_writes(image_bytes)

    if twist_mode == "per_sample":
        twist = create_tensor(
            (N, 3, 4), cvcuda.Type.F32, device_id, layout="NHW", fill_mode="lcg"
        )
    elif twist_mode == "global":
        twist = create_tensor(
            (3, 4), cvcuda.Type.F32, device_id, layout="HW", fill_mode="lcg"
        )
    else:
        state.skip(f"Invalid twistMode: {twist_mode}")
        return None

    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        inter_src = create_tensor(
            (N, H, W, num_channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, num_channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, num_channels, H, W), dtype, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.color_twist_into(inter_dst, inter_src, twist, stream=stream)
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
        dst = create_tensor(tensor_shape, dtype, device_id, layout=layout, fill_mode=0)
    else:  # ImageBatchVarShape mode
        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype_str,
            device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype_str,
            device_id,
            fill_mode=0,
        )

    def run(launch):
        cvcuda.color_twist_into(dst, src, twist, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("colortwist", colortwist)
