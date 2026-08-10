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

"""CV-CUDA CLAHE operator benchmark - Python equivalent of BenchCLAHE.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype_size,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    get_format_from_dtype,
    run_benchmark,
)


def clahe(state):
    """CLAHE operator benchmark matching C++ BenchCLAHE.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    tiles_x = state.get_int64("tilesX")
    tiles_y = state.get_int64("tilesY")
    clip10 = state.get_int64("clip10")
    device_id = state.get_device()

    clip_limit = clip10 / 10.0
    tile_grid_size = (int(tiles_x), int(tiles_y))

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("CLAHE benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) CLAHE benchmark is tensor-only")
        return None

    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    state.add_global_memory_reads(3 * bytes_ if is_fake_planar else bytes_)
    state.add_global_memory_writes(3 * bytes_ if is_fake_planar else bytes_)

    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, 1, H, W),
            cvcuda.Type.U8,
            device_id,
            layout="NCHW",
            fill_mode="lcg",
        )
        inter_src = create_tensor(
            (N, H, W, 1),
            cvcuda.Type.U8,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        inter_dst = create_tensor(
            (N, H, W, 1),
            cvcuda.Type.U8,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        dst = create_tensor(
            (N, 1, H, W),
            cvcuda.Type.U8,
            device_id,
            layout="NCHW",
            fill_mode=0,
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.clahe_into(
                dst=inter_dst,
                src=inter_src,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        shape = (N, 1, H, W) if is_planar else (N, H, W, 1)
        src = create_tensor(
            shape,
            cvcuda.Type.U8,
            device_id,
            layout=layout,
            fill_mode="lcg",
        )
        dst = create_tensor(
            shape,
            cvcuda.Type.U8,
            device_id,
            layout=layout,
            fill_mode=0,
        )

        def run(launch):
            cvcuda.clahe_into(
                dst=dst,
                src=src,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                stream=get_stream(launch),
            )

    else:  # ImageBatchVarShape mode
        img_format = get_format_from_dtype(dtype_str)
        src = create_image_batch_varshape(
            (N, H, W, 1),
            0,
            img_format,
            dtype_str,
            device_id,
            fill_mode="lcg",
        )
        dst = create_image_batch_varshape(
            (N, H, W, 1),
            0,
            img_format,
            dtype_str,
            device_id,
            fill_mode=0,
        )

        def run(launch):
            cvcuda.clahe_into(
                dst=dst,
                src=src,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                stream=get_stream(launch),
            )

    return run


if __name__ == "__main__":
    run_benchmark("clahe", clahe)
