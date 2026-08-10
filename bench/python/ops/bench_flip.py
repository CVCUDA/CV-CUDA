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

"""CV-CUDA Flip operator benchmark - Python equivalent of BenchFlip.cpp"""

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


def flip(state):
    """Flip operator benchmark matching C++ BenchFlip.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    flip_type = state.get_string("flipType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    device_id = state.get_device()

    flip_code_map = {"HORIZONTAL": 0, "VERTICAL": 1, "BOTH": -1}
    flip_code = flip_code_map[flip_type]

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("Flip benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) flip benchmark is tensor-only")
        return None

    # Use the dtype string so vector types (uchar3/float4/...) count their
    # channels, matching C++ sizeof(T) in BenchFlip.cpp. Flip preserves size.
    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    if is_fake_planar:
        # reformat(NCHW->NHWC) + flip + reformat(NHWC->NCHW)
        state.add_global_memory_reads(3 * bytes_)
        state.add_global_memory_writes(3 * bytes_)
    else:
        state.add_global_memory_reads(bytes_)
        state.add_global_memory_writes(bytes_)

    get_stream = create_stream_cache()

    if is_fake_planar:
        # Tensor-only "fake planar": planar->interleaved->flip->interleaved->planar,
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
            (N, H, W, num_channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, num_channels, H, W), dtype, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.flip_into(inter_dst, inter_src, flipCode=flip_code, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        src = create_tensor(
            shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(shape, dtype, device_id, layout=layout, fill_mode=0)
    else:  # ImageBatchVarShape mode
        flip_code = create_tensor(
            (N,), cvcuda.Type.S32, device_id, layout="N", fill_mode=flip_code
        )
        # Select the multi-channel image format from the dtype string so the
        # varshape benchmark moves the same bytes as C++ FillImageBatch<T>.
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
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode=0,
        )

    def run(launch):
        cvcuda.flip_into(dst, src, flipCode=flip_code, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("flip", flip)
