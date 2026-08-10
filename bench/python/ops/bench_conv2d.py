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
    create_image_batch_varshape,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def conv2d(state):
    """Conv2D operator benchmark matching C++ BenchConv2D.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    kernel_size_str = state.get_string("kernelSize")
    border = get_border_type(state.get_string("border"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    if input_kind == "Tensor":  # Tensor mode
        state.skip("Tensor not implemented for this benchmark")
        return None

    if layout not in ("NHWC", "NCHW"):
        state.skip("Conv2D benchmark supports only NHWC and NCHW layouts")
        return None

    is_planar = layout == "NCHW"
    if is_planar and num_channels == 1:
        state.skip("Single-channel Conv2D has no distinct planar image-batch layout")
        return None
    if is_planar and num_channels == 2:
        state.skip("Planar Conv2D benchmark does not support 2-channel layouts")
        return None

    device_id = state.get_device()

    kernel_h, kernel_w = map(int, kernel_size_str.split("x"))

    dtype_size = get_dtype_size(dtype_str)
    state.add_global_memory_reads(N * H * W * dtype_size)
    state.add_global_memory_writes(N * H * W * dtype_size)

    input_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
    kernel_format = cvcuda.Format.F32
    image_shape = (N, H, W) if num_channels == 1 else (N, H, W, num_channels)

    src_batch = create_image_batch_varshape(
        image_shape, 0, input_format, dtype, device_id, fill_mode="checkerboard"
    )
    dst_batch = create_image_batch_varshape(
        image_shape, 0, input_format, dtype, device_id, fill_mode=0
    )
    kernel_batch = create_image_batch_varshape(
        (N, kernel_h, kernel_w),
        0,
        kernel_format,
        cvcuda.Type.F32,
        device_id,
        fill_mode="lcg",
    )
    kernel_anchor_cvcuda = create_tensor(
        (N, 2), cvcuda.Type.S32, device_id, layout="NC", fill_mode=[-1, -1]
    )

    get_stream = create_stream_cache()

    def run(launch):
        cvcuda.conv2d_into(
            dst_batch,
            src_batch,
            kernel_batch,
            kernel_anchor_cvcuda,
            border,
            stream=get_stream(launch),
        )

    return run


if __name__ == "__main__":
    run_benchmark("conv2d", conv2d)
