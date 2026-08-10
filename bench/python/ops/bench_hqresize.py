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

"""CV-CUDA HQResize operator benchmark - Python equivalent of BenchHQResize.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_resize_output_shape,
    get_format_from_dtype,
    get_interpolation_type,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def hqresize(state):
    """HQResize operator benchmark matching C++ BenchHQResize.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    resize_type = state.get_string("resizeType")
    interp = get_interpolation_type(state.get_string("interpolation"))
    antialias = bool(state.get_int64("antialias"))
    input_kind = state.get_string("inputKind")
    device_id = state.get_device()

    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("HQResize benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None
    if input_kind not in ("Tensor", "VarShape", "TensorBatch"):
        state.skip(
            "HQResize benchmark supports only Tensor, VarShape, and TensorBatch input kinds"
        )
        return None
    planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    # Channel count is an explicit axis (default 1): legacy single-channel NHWC profiles omit it, while
    # the planar-comparison profiles (NHWC/NCHW/NCHW_FAKE) set numChannels=3 to compare at parity.
    try:
        channels = int(state.get_int64("numChannels"))
    except Exception:
        channels = 1

    # NCHW_FAKE ("fake planar") is a tensor-only comparison path matching BenchHQResize.cpp.
    if is_fake_planar and input_kind != "Tensor":
        state.skip("Fake-planar (NCHW_FAKE) HQResize benchmark is tensor-only")
        return None

    try:
        _, dst_h, dst_w = get_resize_output_shape((N, H, W), resize_type)
    except ValueError as error:
        state.skip(str(error))
        return None
    if dst_h >= H and dst_w >= W and (dst_h > H or dst_w > W) and antialias:
        state.skip("Antialias is no-op for expanding")
        return None

    dtype_size = get_dtype_size(dtype)

    get_stream = create_stream_cache()

    if is_fake_planar:  # tensor-only: NCHW -> NHWC -> hqresize -> NHWC -> NCHW
        src_bytes = N * H * W * channels * dtype_size
        dst_bytes = N * dst_h * dst_w * channels * dtype_size
        state.add_global_memory_reads(2 * src_bytes + dst_bytes)
        state.add_global_memory_writes(src_bytes + 2 * dst_bytes)

        src = create_tensor(
            (N, channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        inter_src = create_tensor(
            (N, H, W, channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, dst_h, dst_w, channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, channels, dst_h, dst_w), dtype, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.hq_resize_into(
                inter_dst,
                inter_src,
                min_interpolation=interp,
                mag_interpolation=interp,
                antialias=antialias,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":
        state.add_global_memory_reads(N * H * W * channels * dtype_size)
        state.add_global_memory_writes(N * dst_h * dst_w * channels * dtype_size)

        src_shape = (N, channels, H, W) if planar else (N, H, W, channels)
        dst_shape = (
            (N, channels, dst_h, dst_w) if planar else (N, dst_h, dst_w, channels)
        )
        src = create_tensor(
            src_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(dst_shape, dtype, device_id, layout=layout, fill_mode=0)

        def run(launch):
            cvcuda.hq_resize_into(
                dst,
                src,
                min_interpolation=interp,
                mag_interpolation=interp,
                antialias=antialias,
                stream=get_stream(launch),
            )

        return run
    elif input_kind == "VarShape":
        if channels != 3:
            state.skip(
                "HQResize ImageBatchVarShape benchmark currently requires three channels"
            )
            return None

        state.add_global_memory_reads(N * H * W * channels * dtype_size)
        state.add_global_memory_writes(N * dst_h * dst_w * channels * dtype_size)

        img_format = get_format_from_dtype(dtype_str, channels, planar=planar)
        src_batch = create_image_batch_varshape(
            (N, H, W, channels),
            0,
            img_format,
            dtype=dtype,
            device=device_id,
            fill_mode="checkerboard",
        )
        dst_batch = create_image_batch_varshape(
            (N, dst_h, dst_w, channels),
            0,
            img_format,
            dtype=dtype,
            device=device_id,
            fill_mode=0,
        )

        def run_varshape(launch):
            cvcuda.hq_resize_into(
                dst_batch,
                src_batch,
                min_interpolation=interp,
                mag_interpolation=interp,
                antialias=antialias,
                stream=get_stream(launch),
            )

        return run_varshape
    else:  # TensorBatch mode
        state.add_global_memory_reads(N * H * W * channels * dtype_size)
        state.add_global_memory_writes(N * dst_h * dst_w * channels * dtype_size)

        src_shape = (channels, H, W) if planar else (H, W, channels)
        dst_shape = (channels, dst_h, dst_w) if planar else (dst_h, dst_w, channels)
        batch_layout = "CHW" if planar else "HWC"
        src_batch = cvcuda.TensorBatch(N)
        dst_batch = cvcuda.TensorBatch(N)
        for _ in range(N):
            src_batch.pushback(
                create_tensor(
                    src_shape,
                    dtype,
                    device_id,
                    layout=batch_layout,
                    fill_mode="checkerboard",
                )
            )
            dst_batch.pushback(
                create_tensor(
                    dst_shape, dtype, device_id, layout=batch_layout, fill_mode=0
                )
            )

        def run(launch):
            cvcuda.hq_resize_into(
                dst_batch,
                src_batch,
                min_interpolation=interp,
                mag_interpolation=interp,
                antialias=antialias,
                stream=get_stream(launch),
            )

        return run


if __name__ == "__main__":
    run_benchmark("hqresize", hqresize)
