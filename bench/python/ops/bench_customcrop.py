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

"""CV-CUDA CustomCrop operator benchmark - Python equivalent of BenchCustomCrop.cpp"""

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
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def customcrop(state):
    """CustomCrop operator benchmark matching C++ BenchCustomCrop.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    crop_mode = state.get_string("cropMode")
    input_kind = get_input_kind(state.get_string("inputKind"))
    layout = state.get_string("layout")

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "CustomCrop benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"

    # CustomCrop is tensor-only; the native and fake planar paths are too.
    if input_kind == "VarShape":  # ImageBatchVarShape mode
        state.skip("ImageBatchVarShape not implemented for this benchmark")
        return None
    if crop_mode == "full":
        crop_x, crop_y, crop_w, crop_h = 0, 0, W, H
    elif crop_mode == "center_half":
        crop_x, crop_y, crop_w, crop_h = W // 4, H // 4, W // 2, H // 2
    else:
        state.skip(f"Invalid cropMode: {crop_mode}")
        return None

    device_id = state.get_device()

    crop_rect = cvcuda.RectI(crop_x, crop_y, crop_w, crop_h)

    dtype_size = get_dtype_size(dtype_str)
    full_bytes = N * H * W * dtype_size
    crop_bytes = N * crop_h * crop_w * dtype_size
    if is_fake_planar:
        # reformat(NCHW->NHWC) + crop + reformat(NHWC->NCHW)
        state.add_global_memory_reads(full_bytes + 2 * crop_bytes)
        state.add_global_memory_writes(full_bytes + 2 * crop_bytes)
    else:
        state.add_global_memory_reads(crop_bytes)
        state.add_global_memory_writes(crop_bytes)

    get_stream = create_stream_cache()

    if is_fake_planar:
        # Tensor-only "fake planar": planar->interleaved->crop->interleaved->planar,
        # all timed, as the comparison baseline for the native planar (NCHW) path.
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
            (N, crop_h, crop_w, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        dst = create_tensor(
            (N, num_channels, crop_h, crop_w),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode=0,
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.customcrop_into(inter_dst, inter_src, crop_rect, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    src_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
    dst_shape = (
        (N, num_channels, crop_h, crop_w)
        if is_planar
        else (N, crop_h, crop_w, num_channels)
    )
    src = create_tensor(
        src_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
    )
    dst = create_tensor(dst_shape, dtype, device_id, layout=layout, fill_mode=0)

    def run(launch):
        cvcuda.customcrop_into(dst, src, crop_rect, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("customcrop", customcrop)
