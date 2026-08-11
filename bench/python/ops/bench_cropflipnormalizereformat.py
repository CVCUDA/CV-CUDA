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


FLAGS_MODES = {
    "normal": 0,
    "stddev": cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
}


def cropflipnormalizereformat(state):
    """CropFlipNormalizeReformat operator benchmark matching C++ BenchCropFlipNormalizeReformat.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    border = get_border_type(state.get_string("border"))
    crop_mode = state.get_string("cropMode")
    flags_mode = state.get_string("flagsMode")
    input_kind = get_input_kind(state.get_string("inputKind"))
    src_layout = state.get_string("srcLayout")
    dst_layout = state.get_string("layout")

    if input_kind == "Tensor":  # Tensor mode
        state.skip("Tensor not implemented for this benchmark")
        return None
    if src_layout not in ("NHWC", "NCHW") or dst_layout not in ("NHWC", "NCHW"):
        state.skip(
            "CropFlipNormalizeReformat benchmark supports only NHWC and NCHW source/output layouts"
        )
        return None
    if crop_mode not in ["full", "padded16"]:
        state.skip(f"Invalid cropMode: {crop_mode}")
        return None
    if flags_mode not in FLAGS_MODES:
        state.skip(f"Invalid flagsMode: {flags_mode}")
        return None

    device_id = state.get_device()

    globalScale = 1.234
    globalShift = 2.345
    epsilon = 12.34
    flags = FLAGS_MODES[flags_mode]
    borderValue = 0.0

    dtype_size = get_dtype_size(dtype_str)
    state.add_global_memory_reads(N * H * W * dtype_size + N * 4 + N * 4 + N * 4 * 4)
    state.add_global_memory_writes(N * H * W * dtype_size)

    src_planar = src_layout == "NCHW"
    dst_planar = dst_layout == "NCHW"
    input_format = get_format_from_dtype(dtype_str, num_channels, planar=src_planar)

    src_batch = create_image_batch_varshape(
        (N, H, W, num_channels),
        0,
        input_format,
        dtype,
        device_id,
        fill_mode="checkerboard",
    )
    dst_shape = (N, num_channels, H, W) if dst_planar else (N, H, W, num_channels)
    dst = create_tensor(dst_shape, dtype, device_id, layout=dst_layout, fill_mode=0)

    crop_x = -16 if crop_mode == "padded16" else 0
    crop_y = -16 if crop_mode == "padded16" else 0
    rect = create_tensor(
        (N, 1, 1, 4),
        cvcuda.Type.S32,
        device_id,
        layout="NHWC",
        fill_mode=[crop_x, crop_y, W, H],
    )
    flip_code = create_tensor(
        (N,), cvcuda.Type.S32, device_id, layout="N", fill_mode=-1
    )

    # base and scale: deterministic LCG over float [-1, +1], bit-identical to
    # the C++ RandomValues<float>() GPU fast path (see BenchCropFlipNormalizeReformat.cpp).
    base = create_tensor(
        (N, 1, 1, 1), cvcuda.Type.F32, device_id, layout="NHWC", fill_mode="lcg"
    )
    scale = create_tensor(
        (N, 1, 1, 1), cvcuda.Type.F32, device_id, layout="NHWC", fill_mode="lcg"
    )

    get_stream = create_stream_cache()

    def run(launch):
        cvcuda.crop_flip_normalize_reformat_into(
            dst,
            src_batch,
            rect,
            flip_code,
            base,
            scale,
            globalscale=globalScale,
            globalshift=globalShift,
            epsilon=epsilon,
            flags=flags,
            border=border,
            bvalue=borderValue,
            stream=get_stream(launch),
        )

    return run


if __name__ == "__main__":
    run_benchmark("cropflipnormalizereformat", cropflipnormalizereformat)
