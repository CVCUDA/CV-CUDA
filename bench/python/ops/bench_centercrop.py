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

"""CV-CUDA CenterCrop operator benchmark - Python equivalent of BenchCenterCrop.cpp"""

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


def centercrop(state):
    """CenterCrop operator benchmark matching C++ BenchCenterCrop.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    crop_type = state.get_string("cropType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "CenterCrop benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"

    # CenterCrop is tensor-only; the native and fake planar paths are too.
    if input_kind == "VarShape":  # ImageBatchVarShape mode
        state.skip("ImageBatchVarShape not implemented for this benchmark")
        return None

    device_id = state.get_device()

    if crop_type == "SAME":
        crop_h, crop_w = H, W
    elif crop_type == "QUARTER":
        crop_h, crop_w = H // 2, W // 2
    else:
        state.skip(f"Invalid cropType: {crop_type}")
        return None

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
            cvcuda.center_crop_into(
                inter_dst, inter_src, (crop_w, crop_h), stream=stream
            )
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
        cvcuda.center_crop_into(dst, src, (crop_w, crop_h), stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("centercrop", centercrop)
