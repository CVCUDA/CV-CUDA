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

"""CV-CUDA ResizeCropConvertReformat operator benchmark.

Python equivalent of BenchResizeCropConvertReformat.cpp
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math  # noqa: E402

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_num_channels,
    get_dtype_size,
    get_interpolation_type,
    get_format_from_dtype,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def resizecropconvertreformat(state):
    """ResizeCropConvertReformat fused operator benchmark matching C++ BenchResizeCropConvertReformat.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    interp = get_interpolation_type(state.get_string("interpolation"))
    layout = state.get_string("layout")
    device_id = state.get_device()

    N, H, W = shape
    nc = get_num_channels(dtype_str)
    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "ResizeCropConvertReformat benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    if is_fake_planar and input_kind == "VarShape":
        state.skip(
            "Fake-planar (NCHW_FAKE) ResizeCropConvertReformat benchmark is tensor-only"
        )
        return None

    if H <= 1 or W <= 1:
        state.skip("Height and width must be > 1 for resize/crop operations")
        return None

    resize_h, resize_w = H // 2, W // 2
    crop_x, crop_y = 0, 0
    crop_w = max(1, min(512, resize_w - 1))
    crop_h = max(1, min(512, resize_h - 1))

    dst_shape = (
        (N, nc, crop_h, crop_w)
        if is_planar or is_fake_planar
        else (N, crop_h, crop_w, nc)
    )

    scale_x = W / resize_w
    scale_y = H / resize_h
    src_region_w = min(math.ceil((crop_x + crop_w) * scale_x) + 2, W)
    src_region_h = min(math.ceil((crop_y + crop_h) * scale_y) + 2, H)

    dtype_size = get_dtype_size(dtype_str)
    src_bytes = N * H * W * nc * dtype_size
    dst_bytes = N * crop_h * crop_w * nc * dtype_size
    state.add_global_memory_reads(
        N * src_region_h * src_region_w * nc * dtype_size
        + (src_bytes if is_fake_planar else 0)
    )
    state.add_global_memory_writes(dst_bytes + (src_bytes if is_fake_planar else 0))

    get_stream = create_stream_cache()

    if input_kind == "Tensor":
        if is_planar or is_fake_planar:
            src = create_tensor(
                (N, nc, H, W),
                dtype_str,
                device_id,
                layout="NCHW",
                fill_mode="checkerboard",
            )
            dst = create_tensor(
                dst_shape, dtype_str, device_id, layout="NCHW", fill_mode=0
            )
        else:
            src = create_tensor(
                (N, H, W, nc),
                dtype_str,
                device_id,
                layout="NHWC",
                fill_mode="checkerboard",
            )
            dst = create_tensor(
                dst_shape, dtype_str, device_id, layout="NHWC", fill_mode=0
            )

        if is_fake_planar:
            inter_src = create_tensor(
                (N, H, W, nc), dtype_str, device_id, layout="NHWC", fill_mode=0
            )

            def run_fake(launch):
                stream = get_stream(launch)
                cvcuda.reformat_into(inter_src, src, stream=stream)
                cvcuda.resize_crop_convert_reformat_into(
                    dst,
                    inter_src,
                    (resize_w, resize_h),
                    interp,
                    (crop_x, crop_y),
                    manip=cvcuda.ChannelManip.NO_OP,
                    stream=stream,
                )

            return run_fake

        def run(launch):
            stream = get_stream(launch)
            cvcuda.resize_crop_convert_reformat_into(
                dst,
                src,
                (resize_w, resize_h),
                interp,
                (crop_x, crop_y),
                manip=cvcuda.ChannelManip.NO_OP,
                stream=stream,
            )

        return run

    else:
        img_format = get_format_from_dtype(dtype_str, nc, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, nc),
            0,
            img_format,
            dtype=dtype_str,
            device=device_id,
            fill_mode="checkerboard",
        )
        dst = create_tensor(dst_shape, dtype_str, device_id, layout=layout, fill_mode=0)

        def run(launch):
            stream = get_stream(launch)
            cvcuda.resize_crop_convert_reformat_into(
                dst,
                src,
                (resize_w, resize_h),
                interp,
                (crop_x, crop_y),
                manip=cvcuda.ChannelManip.NO_OP,
                stream=stream,
            )

        return run


if __name__ == "__main__":
    run_benchmark("resizecropconvertreformat", resizecropconvertreformat)
