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

"""CV-CUDA Remap operator benchmark - Python equivalent of BenchRemap.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402
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
    _lcg_fill,
    _LCG_SEED,
)


def remap(state):
    """Remap operator benchmark matching C++ BenchRemap.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    nc = get_num_channels(dtype_str)
    map_type = state.get_string("mapType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    device_id = state.get_device()

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("Remap benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None
    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) remap benchmark is tensor-only")
        return None

    N, H, W = shape

    if map_type == "DENSE":
        src_interp = cvcuda.Interp.NEAREST
        map_interp = cvcuda.Interp.NEAREST
        border_type = cvcuda.Border.CONSTANT
        map_value_type = cvcuda.Remap.ABSOLUTE_NORMALIZED
        map_shape = (N, H, W, 1)
    elif map_type == "RELATIVE":
        src_interp = cvcuda.Interp.CUBIC
        map_interp = cvcuda.Interp.CUBIC
        border_type = cvcuda.Border.REFLECT101
        map_value_type = cvcuda.Remap.RELATIVE_NORMALIZED
        map_shape = (N, 4, 4, 1)
    else:
        state.skip(f"Invalid mapType = {map_type}")
        return None

    align_corners = True
    border_value = np.array([0, 0, 0, 0], dtype=np.float32)

    dtype_size = get_dtype_size(dtype_str)
    state.add_global_memory_reads(
        N * H * W * dtype_size + map_shape[0] * map_shape[1] * map_shape[2] * 2 * 4
    )
    state.add_global_memory_writes(N * H * W * dtype_size)

    # map: deterministic LCG over float [-1, +1] (bit-identical to C++
    # RandomValues<float2>() which routes through the GPU LCG fast path).
    with cp.cuda.Device(device_id):
        map_data = cp.empty(
            (map_shape[0], map_shape[1], map_shape[2], 2), dtype=cp.float32
        )
        _lcg_fill(map_data, _LCG_SEED)
    map_tensor = cvcuda.as_tensor(map_data, "NHWC")

    get_stream = create_stream_cache()

    def do_remap(dst, src, stream):
        cvcuda.remap_into(
            dst,
            src,
            map_tensor,
            src_interp=src_interp,
            map_interp=map_interp,
            map_type=map_value_type,
            align_corners=align_corners,
            border=border_type,
            border_value=border_value,
            stream=stream,
        )

    if is_fake_planar:
        # Tensor-only: planar->interleaved->remap->interleaved->planar, all timed.
        src = create_tensor(
            (N, nc, H, W), dtype, device_id, layout="NCHW", fill_mode="checkerboard"
        )
        inter_src = create_tensor(
            (N, H, W, nc), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, nc), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor((N, nc, H, W), dtype, device_id, layout="NCHW", fill_mode=0)

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            do_remap(inter_dst, inter_src, stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":
        src_shape = (N, nc, H, W) if is_planar else (N, H, W, nc)
        src = create_tensor(
            src_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(src_shape, dtype, device_id, layout=layout, fill_mode=0)
    else:
        img_format = get_format_from_dtype(dtype_str, nc, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, nc),
            0,
            img_format,
            dtype=dtype,
            device=device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, H, W, nc), 0, img_format, dtype=dtype, device=device_id, fill_mode=0
        )

    def run(launch):
        do_remap(dst, src, get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("remap", remap)
