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

"""CV-CUDA PadAndStack operator benchmark - Python equivalent of BenchPadAndStack.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp  # noqa: E402
import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    get_format_from_dtype,
    get_num_channels,
    parse_shape,
    get_dtype,
    get_dtype_size,
    create_stream_cache,
    create_image_batch_varshape,
    run_benchmark,
)


def padandstack(state):
    """PadAndStack operator benchmark matching C++ BenchPadAndStack.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype = get_dtype(state.get_string("InOutDataType"))
    dtype_str = state.get_string("InOutDataType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    layout = state.get_string("layout")
    border_str = state.get_string("border")
    pad = state.get_int64("pad")
    if layout not in ("NHWC", "NCHW"):
        state.skip("PadAndStack benchmark supports only NHWC and NCHW layouts")
        return None
    is_planar = layout == "NCHW"
    num_channels = get_num_channels(dtype_str)
    if is_planar and num_channels == 2:
        state.skip("Planar PadAndStack benchmark does not support 2-channel layouts")
        return None
    if is_planar and num_channels == 1:
        state.skip(
            "Single-channel PadAndStack has no distinct planar image-batch format"
        )
        return None

    N, H, W = shape
    num_batches = N
    src_height = H
    src_width = W

    dst_height = src_height + 2 * pad
    dst_width = src_width + 2 * pad

    border_map = {
        "CONSTANT": cvcuda.Border.CONSTANT,
        "REPLICATE": cvcuda.Border.REPLICATE,
        "REFLECT": cvcuda.Border.REFLECT,
        "WRAP": cvcuda.Border.WRAP,
        "REFLECT101": cvcuda.Border.REFLECT101,
    }
    border_type = border_map.get(border_str, cvcuda.Border.REFLECT101)
    border_value = 0.0

    dtype_size = get_dtype_size(dtype)
    state.add_global_memory_reads(
        num_batches * src_height * src_width * num_channels * dtype_size
        + num_batches * 4 * 2
    )
    state.add_global_memory_writes(
        num_batches * dst_height * dst_width * num_channels * dtype_size
    )

    if input_kind == "Tensor":
        state.skip(
            "PadAndStack only supports ImageBatchVarShape input (inputKind=VarShape)"
        )
        return None

    img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)

    device_id = state.get_device()
    src = create_image_batch_varshape(
        (num_batches, src_height, src_width, num_channels),
        0,
        img_format,
        dtype,
        device_id,
        fill_mode="checkerboard",
    )

    dst_shape = (
        (num_batches, num_channels, dst_height, dst_width)
        if is_planar
        else (num_batches, dst_height, dst_width, num_channels)
    )
    dst = cvcuda.Tensor(dst_shape, dtype, layout)

    top_pad = (dst_height - src_height) // 2
    left_pad = (dst_width - src_width) // 2

    top_data = cp.full((1, 1, num_batches, 1), top_pad, dtype=cp.int32)
    left_data = cp.full((1, 1, num_batches, 1), left_pad, dtype=cp.int32)

    top = cvcuda.as_tensor(top_data, "NHWC")
    left = cvcuda.as_tensor(left_data, "NHWC")

    get_stream = create_stream_cache()

    def run(launch):
        stream = get_stream(launch)
        cvcuda.padandstack_into(
            dst, src, top, left, border_type, border_value, stream=stream
        )

    return run


if __name__ == "__main__":
    run_benchmark("padandstack", padandstack)
