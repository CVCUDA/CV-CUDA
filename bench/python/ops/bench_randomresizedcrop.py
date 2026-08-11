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

"""CV-CUDA RandomResizedCrop operator benchmark - Python equivalent of BenchRandomResizedCrop.cpp"""

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
    get_interpolation_type,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def randomresizedcrop(state):
    """RandomResizedCrop operator benchmark matching C++ BenchRandomResizedCrop.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    nc = get_num_channels(dtype_str)
    interp = get_interpolation_type(state.get_string("interpolation"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    resize_type = state.get_string("resizeType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    device_id = state.get_device()

    N, H, W = shape

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "RandomResizedCrop benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) RandomResizedCrop benchmark is tensor-only")
        return None

    if resize_type == "EXPAND":
        dst_H, dst_W = H * 2, W * 2
    elif resize_type == "CONTRACT":
        dst_H, dst_W = H // 2, W // 2
    else:
        state.skip(f"Invalid resizeType: {resize_type}")
        return None

    min_scale = 0.08
    max_scale = 1.0
    min_ratio = 0.5
    max_ratio = 2.0
    seed = 1234

    dtype_size = get_dtype_size(dtype_str)
    src_bytes = N * H * W * dtype_size
    dst_bytes = N * dst_H * dst_W * dtype_size
    if is_fake_planar:
        # reformat(NCHW->NHWC) + random_resized_crop + reformat(NHWC->NCHW)
        state.add_global_memory_reads(2 * src_bytes + dst_bytes)
        state.add_global_memory_writes(src_bytes + 2 * dst_bytes)
    else:
        state.add_global_memory_reads(src_bytes)
        state.add_global_memory_writes(dst_bytes)

    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, nc, H, W), dtype, device_id, layout="NCHW", fill_mode="checkerboard"
        )
        inter_src = create_tensor(
            (N, H, W, nc), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, dst_H, dst_W, nc), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, nc, dst_H, dst_W), dtype, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.random_resized_crop_into(
                inter_dst,
                inter_src,
                min_scale,
                max_scale,
                min_ratio,
                max_ratio,
                interp,
                seed,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        input_shape = (N, nc, H, W) if is_planar else (N, H, W, nc)
        dst_shape = (N, nc, dst_H, dst_W) if is_planar else (N, dst_H, dst_W, nc)
        src = create_tensor(
            input_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(dst_shape, dtype, device_id, layout=layout, fill_mode=0)
    else:  # VarShape mode
        img_format = get_format_from_dtype(dtype_str, nc, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, nc),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, dst_H, dst_W, nc),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode=0,
        )

    def run(launch):
        stream = get_stream(launch)
        cvcuda.random_resized_crop_into(
            dst,
            src,
            min_scale,
            max_scale,
            min_ratio,
            max_ratio,
            interp,
            seed,
            stream=stream,
        )

    return run


if __name__ == "__main__":
    run_benchmark("randomresizedcrop", randomresizedcrop)
