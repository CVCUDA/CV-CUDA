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

"""CV-CUDA PillowResize operator benchmark - Python equivalent of BenchPillowResize.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_resize_output_shape,
    get_num_channels,
    get_format_from_dtype,
    get_interpolation_type,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def pillowresize(state):
    """PillowResize operator benchmark matching C++ BenchPillowResize.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    resize_type = state.get_string("resizeType")
    interp = get_interpolation_type(state.get_string("interpolation"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    device_id = state.get_device()

    # The layout axis is optional; profiles without it default to interleaved NHWC.
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "PillowResize benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    # NCHW_FAKE ("fake planar") is a tensor-only comparison path.
    if is_fake_planar and input_kind != "Tensor":
        state.skip("Fake-planar (NCHW_FAKE) PillowResize benchmark is tensor-only")
        return None

    N, H, W = shape

    try:
        _, dst_H, dst_W = get_resize_output_shape(shape, resize_type)
    except ValueError as error:
        state.skip(str(error))
        return None

    dtype_size = get_dtype_size(dtype_str)
    src_bytes = N * H * W * dtype_size
    dst_bytes = N * dst_H * dst_W * dtype_size
    if is_fake_planar:
        # reformat(NCHW->NHWC) + resize + reformat(NHWC->NCHW)
        state.add_global_memory_reads(2 * src_bytes + dst_bytes)
        state.add_global_memory_writes(src_bytes + 2 * dst_bytes)
    else:
        state.add_global_memory_reads(src_bytes)
        state.add_global_memory_writes(dst_bytes)

    # Workspace/image format carries the channel count (and planar vs interleaved layout); the tensor
    # path needs it for getWorkspaceRequirements. Matches BenchPillowResize.cpp.
    img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)

    get_stream = create_stream_cache()

    if is_fake_planar:
        # Tensor-only "fake planar": planar->interleaved->resize->interleaved->planar, all timed,
        # as the comparison baseline for the native planar (NCHW) path. img_format is interleaved
        # here (is_planar is False for NCHW_FAKE).
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
            (N, dst_H, dst_W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        dst = create_tensor(
            (N, num_channels, dst_H, dst_W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode=0,
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.pillowresize_into(
                inter_dst, inter_src, img_format, interp, stream=stream
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    # Channels come from the dtype (e.g. uchar3/float3 -> 3); planar (NCHW) puts the channel dimension
    # before the spatial dims, interleaved (NHWC) after.
    def shaped(h, w):
        return (N, num_channels, h, w) if is_planar else (N, h, w, num_channels)

    if input_kind == "Tensor":
        src = create_tensor(
            shaped(H, W), dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = cvcuda.Tensor(shaped(dst_H, dst_W), dtype, layout)

        def run(launch):
            stream = get_stream(launch)
            cvcuda.pillowresize_into(dst, src, img_format, interp, stream=stream)

        return run
    else:  # VarShape
        src = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype=dtype,
            device=device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, dst_H, dst_W, num_channels),
            0,
            img_format,
            dtype=dtype,
            device=device_id,
            fill_mode=0,
        )

        def run(launch):
            stream = get_stream(launch)
            cvcuda.pillowresize_into(dst, src, interp, stream=stream)

        return run


if __name__ == "__main__":
    run_benchmark("pillowresize", pillowresize)
