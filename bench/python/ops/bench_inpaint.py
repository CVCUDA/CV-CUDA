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

"""CV-CUDA Inpaint operator benchmark - Python equivalent of BenchInpaint.cpp"""

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

try:
    import cupy as cp
except ImportError:
    import numpy as cp


def inpaint(state):
    """Inpaint operator benchmark matching C++ BenchInpaint.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    inpaint_radius = state.get_float64("inpaintRadius")
    device_id = state.get_device()

    N, H, W = shape

    num_channels = get_num_channels(dtype_str)

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("Inpaint benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) Inpaint benchmark is tensor-only")
        return None

    dtype_size = get_dtype_size(dtype_str)
    image_bytes = N * H * W * dtype_size
    mask_bytes = N * H * W
    if is_fake_planar:
        state.add_global_memory_reads(3 * image_bytes + mask_bytes)
        state.add_global_memory_writes(3 * image_bytes)
    else:
        state.add_global_memory_reads(image_bytes + mask_bytes)
        state.add_global_memory_writes(image_bytes)

    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="lcg",
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

        with cp.cuda.Device(device_id):
            mask_data = cp.zeros((N, H, W, 1), dtype=cp.uint8)
            mask_data[:, 1::2, :, :] = 1
        mask = cvcuda.as_tensor(mask_data, "NHWC")

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.inpaint_into(
                inter_dst, inter_src, mask, inpaint_radius, stream=stream
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":
        input_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        mask_shape = (N, H, W, 1)

        src = create_tensor(
            input_shape, dtype, device_id, layout=layout, fill_mode="lcg"
        )
        dst = create_tensor(input_shape, dtype, device_id, layout=layout, fill_mode=0)

        with cp.cuda.Device(device_id):
            mask_data = cp.zeros(mask_shape, dtype=cp.uint8)
            mask_data[:, 1::2, :, :] = 1
        mask = cvcuda.as_tensor(mask_data, "NHWC")

        def run(launch):
            cvcuda.inpaint_into(
                dst, src, mask, inpaint_radius, stream=get_stream(launch)
            )

        return run
    else:  # ImageBatchVarShape mode
        if is_planar and dtype_str == "uchar4":
            state.skip(
                "uchar4 planar var-shape Inpaint benchmark is unsupported by the Python image API"
            )
            return None

        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)

        if num_channels == 1:
            img_shape = (N, H, W)
        else:
            img_shape = (N, H, W, num_channels)

        src = create_image_batch_varshape(
            img_shape, 0, img_format, dtype, device_id, fill_mode="lcg"
        )
        dst = create_image_batch_varshape(
            img_shape, 0, img_format, dtype, device_id, fill_mode=0
        )

        with cp.cuda.Device(device_id):
            mask = cvcuda.ImageBatchVarShape(N)
            for i in range(N):
                # Uniform image sizes (the var-shape benches run with zero size variation).
                img_h, img_w = H, W

                mask_data = cp.zeros((img_h, img_w), dtype=cp.uint8)
                mask_data[:, 1::2] = 1

                img = cvcuda.as_image(mask_data, cvcuda.Format.U8)
                mask.pushback(img)

        def run(launch):
            cvcuda.inpaint_into(
                dst, src, mask, inpaint_radius, stream=get_stream(launch)
            )

        return run


if __name__ == "__main__":
    run_benchmark("inpaint", inpaint)
