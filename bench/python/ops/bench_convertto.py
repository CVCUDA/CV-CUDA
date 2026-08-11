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

"""CV-CUDA ConvertTo operator benchmark - Python equivalent of BenchConvertTo.cpp"""

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

SCALE_MODES = {
    "affine": (0.123, 0.456),
    "identity": (1.0, 0.0),
}


def convertto(state):
    """ConvertTo operator benchmark matching C++ BenchConvertTo.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    out_dtype_str = state.get_string("outDataType")
    out_dtype = get_dtype(out_dtype_str)
    scale_mode = state.get_string("scaleMode")
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "ConvertTo benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"

    # ConvertTo is tensor-only; the native and fake planar paths are too.
    if input_kind == "VarShape":  # ImageBatchVarShape mode
        state.skip("ImageBatchVarShape not implemented for this benchmark")
        return None
    if scale_mode not in SCALE_MODES:
        state.skip(f"Invalid scaleMode: {scale_mode}")
        return None

    scale, offset = SCALE_MODES[scale_mode]

    device_id = state.get_device()

    dtype_size = get_dtype_size(dtype_str)
    out_dtype_size = get_dtype_size(out_dtype) * num_channels
    src_bytes = N * H * W * dtype_size
    dst_bytes = N * H * W * out_dtype_size
    if is_fake_planar:
        # reformat(NCHW->NHWC) + convert + reformat(NHWC->NCHW)
        state.add_global_memory_reads(2 * src_bytes + dst_bytes)
        state.add_global_memory_writes(src_bytes + 2 * dst_bytes)
    else:
        state.add_global_memory_reads(src_bytes)
        state.add_global_memory_writes(dst_bytes)

    get_stream = create_stream_cache()

    if is_fake_planar:
        # Tensor-only "fake planar": planar->interleaved->convert->interleaved->planar,
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
            (N, H, W, num_channels), out_dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, num_channels, H, W), out_dtype, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.convertto_into(
                inter_dst, inter_src, scale=scale, offset=offset, stream=stream
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    src_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
    dst_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
    src = create_tensor(
        src_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
    )
    dst = create_tensor(dst_shape, out_dtype, device_id, layout=layout, fill_mode=0)

    def run(launch):
        cvcuda.convertto_into(
            dst, src, scale=scale, offset=offset, stream=get_stream(launch)
        )

    return run


if __name__ == "__main__":
    run_benchmark("convertto", convertto)
