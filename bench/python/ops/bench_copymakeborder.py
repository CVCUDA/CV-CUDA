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

"""CV-CUDA CopyMakeBorder operator benchmark - Python equivalent of BenchCopyMakeBorder.cpp"""

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
    get_border_type,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def copymakeborder(state):
    """CopyMakeBorder operator benchmark matching C++ BenchCopyMakeBorder.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    border = get_border_type(state.get_string("border"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    layout = state.get_string("layout")
    device_id = state.get_device()

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "CopyMakeBorder benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) CopyMakeBorder benchmark is tensor-only")
        return None

    top = H // 2
    left = W // 2

    dtype_size = get_dtype_size(dtype_str)
    src_bytes = N * H * W * dtype_size
    dst_bytes = N * (top + H) * (left + W) * dtype_size
    if is_fake_planar:
        state.add_global_memory_reads(2 * src_bytes + dst_bytes)
        state.add_global_memory_writes(src_bytes + 2 * dst_bytes)
    else:
        state.add_global_memory_reads(src_bytes)
        state.add_global_memory_writes(dst_bytes)

    border_value = [0.0, 0.0, 0.0, 0.0]

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
            (N, H, W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        inter_dst = create_tensor(
            (N, top + H, left + W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        dst = create_tensor(
            (N, num_channels, top + H, left + W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode=0,
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.copymakeborder_into(
                inter_dst,
                inter_src,
                border_mode=border,
                border_value=border_value,
                top=top,
                left=left,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        src_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        dst_shape = (
            (N, num_channels, top + H, left + W)
            if is_planar
            else (N, top + H, left + W, num_channels)
        )
        src = create_tensor(
            src_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode="checkerboard",
        )
        dst = create_tensor(
            dst_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode=0,
        )
    else:  # ImageBatchVarShape mode
        top_scalar, left_scalar = top, left
        top = create_tensor(
            (N, 1, 1, 1),
            cvcuda.Type.S32,
            device_id,
            layout="NHWC",
            fill_mode=top_scalar,
        )
        left = create_tensor(
            (N, 1, 1, 1),
            cvcuda.Type.S32,
            device_id,
            layout="NHWC",
            fill_mode=left_scalar,
        )
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
            (N, top_scalar + H, left_scalar + W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )

    def run(launch):
        cvcuda.copymakeborder_into(
            dst,
            src,
            border_mode=border,
            border_value=border_value,
            top=top,
            left=left,
            stream=get_stream(launch),
        )

    return run


if __name__ == "__main__":
    run_benchmark("copymakeborder", copymakeborder)
