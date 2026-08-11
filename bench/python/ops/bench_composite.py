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

"""CV-CUDA Composite operator benchmark - Python equivalent of BenchComposite.cpp"""

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


def composite(state):
    """Composite operator benchmark matching C++ BenchComposite.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    input_kind = get_input_kind(state.get_string("inputKind"))
    out_channels = state.get_int64("outChannels")
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    device_id = state.get_device()

    num_channels = get_num_channels(dtype_str)

    if out_channels not in (3, 4):
        state.skip(f"Invalid outChannels: {out_channels}")
        return None
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "Composite benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None
    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) Composite benchmark is tensor-only")
        return None

    dtype_size = get_dtype_size(dtype_str)
    base_dtype_size = get_dtype_size(dtype)
    mask_dtype_size = 1  # uint8 mask
    state.add_global_memory_reads(N * H * W * (dtype_size * 2 + mask_dtype_size))
    state.add_global_memory_writes(N * H * W * out_channels * base_dtype_size)

    if is_fake_planar:
        fg = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        bg = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        mask = create_tensor(
            (N, 1, H, W),
            cvcuda.Type.U8,
            device_id,
            layout="NCHW",
            fill_mode=1,
        )
        inter_fg = create_tensor(
            (N, H, W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        inter_bg = create_tensor(
            (N, H, W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        inter_mask = create_tensor(
            (N, H, W, 1),
            cvcuda.Type.U8,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        inter_dst = create_tensor(
            (N, H, W, out_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        dst = create_tensor(
            (N, out_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode=0,
        )
    elif input_kind == "Tensor":  # Tensor mode
        tensor_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        mask_shape = (N, 1, H, W) if is_planar else (N, H, W, 1)
        dst_shape = (N, out_channels, H, W) if is_planar else (N, H, W, out_channels)
        fg = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode="checkerboard",
        )
        bg = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode="checkerboard",
        )
        mask = create_tensor(
            mask_shape,
            cvcuda.Type.U8,
            device_id,
            layout=layout,
            fill_mode=1,
        )
        dst = create_tensor(
            dst_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode=0,
        )
    else:  # ImageBatchVarShape mode
        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
        dst_format = get_format_from_dtype(dtype_str, out_channels, planar=is_planar)
        mask_format = cvcuda.Format.U8
        fg = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )
        bg = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )
        mask = create_image_batch_varshape(
            (N, H, W), 0, mask_format, cvcuda.Type.U8, device_id, fill_mode=1
        )
        dst = create_image_batch_varshape(
            (N, H, W, out_channels),
            0,
            dst_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )

    get_stream = create_stream_cache()

    def run(launch):
        stream = get_stream(launch)
        if is_fake_planar:
            cvcuda.reformat_into(inter_fg, fg, stream=stream)
            cvcuda.reformat_into(inter_bg, bg, stream=stream)
            cvcuda.reformat_into(inter_mask, mask, stream=stream)
            cvcuda.composite_into(
                inter_dst, inter_fg, inter_bg, inter_mask, stream=stream
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)
        else:
            cvcuda.composite_into(dst, fg, bg, mask, stream=stream)

    return run


if __name__ == "__main__":
    run_benchmark("composite", composite)
