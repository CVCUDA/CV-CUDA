#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CV-CUDA AdvCvtColor operator benchmark - Python equivalent of BenchAdvCvtColor.cpp"""

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
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def advcvtcolor(state):
    """AdvCvtColor operator benchmark matching C++ BenchAdvCvtColor.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    rgb_channels = get_num_channels(dtype_str)
    code_str = state.get_string("code")
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "AdvCvtColor benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"

    if input_kind == "VarShape":  # ImageBatchVarShape mode
        state.skip("ImageBatchVarShape not implemented for this benchmark")
        return None

    device_id = state.get_device()

    code_map = {
        "BGR2YUV": (cvcuda.ColorConversion.BGR2YUV, "444"),
        "RGB2YUV": (cvcuda.ColorConversion.RGB2YUV, "444"),
        "YUV2BGR": (cvcuda.ColorConversion.YUV2BGR, "444"),
        "YUV2RGB": (cvcuda.ColorConversion.YUV2RGB, "444"),
        "RGB2YUV_NV12": (cvcuda.ColorConversion.RGB2YUV_NV12, "rgb_to_nv"),
        "BGR2YUV_NV21": (cvcuda.ColorConversion.BGR2YUV_NV21, "rgb_to_nv"),
        "YUV2RGB_NV12": (cvcuda.ColorConversion.YUV2RGB_NV12, "nv_to_rgb"),
        "YUV2BGR_NV21": (cvcuda.ColorConversion.YUV2BGR_NV21, "nv_to_rgb"),
    }
    code, conversion_shape = code_map[code_str]
    spec = cvcuda.ColorSpec.BT2020

    if dtype != cvcuda.Type.U8:
        state.skip("AdvCvtColor benchmark supports uint8 vector types only")
        return None

    src_h = H
    dst_h = H
    src_c = rgb_channels
    dst_c = rgb_channels

    if conversion_shape == "444":
        if rgb_channels != 3:
            state.skip("Interleaved 444 conversion requires uchar3")
            return None
        src_c = 3
        dst_c = 3
    elif conversion_shape == "rgb_to_nv":
        if rgb_channels not in (3, 4):
            state.skip("RGB/BGR to NV conversion requires uchar3 or uchar4")
            return None
        if H % 2 or W % 2:
            state.skip("NV conversion requires even height and width")
            return None
        dst_h = (H * 3) // 2
        dst_c = 1
    else:
        if rgb_channels not in (3, 4):
            state.skip("NV to RGB/BGR conversion requires uchar3 or uchar4")
            return None
        if H % 2 or W % 2:
            state.skip("NV conversion requires even height and width")
            return None
        src_h = (H * 3) // 2
        src_c = 1

    dtype_size = get_dtype_size(dtype)
    src_bytes = N * src_h * W * src_c * dtype_size
    dst_bytes = N * dst_h * W * dst_c * dtype_size
    if is_fake_planar:
        state.add_global_memory_reads(2 * src_bytes + dst_bytes)
        state.add_global_memory_writes(src_bytes + 2 * dst_bytes)
    else:
        state.add_global_memory_reads(src_bytes)
        state.add_global_memory_writes(dst_bytes)

    get_stream = create_stream_cache()

    def tensor_shape(height, channels, tensor_layout):
        if tensor_layout == "NCHW":
            return (N, channels, height, W)
        return (N, height, W, channels)

    if is_fake_planar:
        src = create_tensor(
            tensor_shape(src_h, src_c, "NCHW"),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        inter_src = create_tensor(
            tensor_shape(src_h, src_c, "NHWC"),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        inter_dst = create_tensor(
            tensor_shape(dst_h, dst_c, "NHWC"),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        dst = create_tensor(
            tensor_shape(dst_h, dst_c, "NCHW"),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode=0,
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.advcvtcolor_into(inter_dst, inter_src, code, spec, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    tensor_layout = "NCHW" if is_planar else "NHWC"
    src = create_tensor(
        tensor_shape(src_h, src_c, tensor_layout),
        dtype,
        device_id,
        layout=tensor_layout,
        fill_mode="checkerboard",
    )
    dst = create_tensor(
        tensor_shape(dst_h, dst_c, tensor_layout),
        dtype,
        device_id,
        layout=tensor_layout,
        fill_mode=0,
    )

    def run(launch):
        cvcuda.advcvtcolor_into(dst, src, code, spec, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("advcvtcolor", advcvtcolor)
