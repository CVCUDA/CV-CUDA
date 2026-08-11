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

"""CV-CUDA MedianBlur operator benchmark - Python equivalent of BenchMedianBlur.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp  # noqa: E402
import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_num_channels,
    get_dtype_size,
    get_format_from_dtype,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def medianblur(state):
    """MedianBlur operator benchmark matching C++ BenchMedianBlur.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    ksize_str = state.get_string("kernelSize")
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    device_id = state.get_device()

    ksize = parse_shape(ksize_str)

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "MedianBlur benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) MedianBlur benchmark is tensor-only")
        return None

    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    if is_fake_planar:
        state.add_global_memory_reads(3 * bytes_)
        state.add_global_memory_writes(3 * bytes_)
    else:
        state.add_global_memory_reads(bytes_)
        state.add_global_memory_writes(bytes_)

    get_stream = create_stream_cache()

    cupy_dtype_map = {
        cvcuda.Type.U8: cp.uint8,
        cvcuda.Type.U16: cp.uint16,
        cvcuda.Type.F32: cp.float32,
    }
    cupy_dtype = cupy_dtype_map.get(dtype, cp.float32)
    is_float = dtype == cvcuda.Type.F32

    if is_fake_planar:
        with cp.cuda.Device(device_id):
            gradient = (255 - cp.arange(W, dtype=cp.int32) % 256).astype(cupy_dtype)
            if is_float:
                gradient = gradient / 255.0
            src_data = cp.broadcast_to(
                gradient.reshape(1, 1, 1, W), (N, num_channels, H, W)
            ).copy()
            src = cvcuda.as_tensor(src_data, "NCHW")

            inter_src_data = cp.zeros((N, H, W, num_channels), dtype=cupy_dtype)
            inter_src = cvcuda.as_tensor(inter_src_data, "NHWC")
            inter_dst_data = cp.zeros((N, H, W, num_channels), dtype=cupy_dtype)
            inter_dst = cvcuda.as_tensor(inter_dst_data, "NHWC")

            dst_data = cp.zeros((N, num_channels, H, W), dtype=cupy_dtype)
            dst = cvcuda.as_tensor(dst_data, "NCHW")

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.median_blur_into(inter_dst, inter_src, ksize, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        with cp.cuda.Device(device_id):
            gradient = (255 - cp.arange(W, dtype=cp.int32) % 256).astype(cupy_dtype)
            if is_float:
                gradient = gradient / 255.0
            if is_planar:
                src_data = cp.broadcast_to(
                    gradient.reshape(1, 1, 1, W), (N, num_channels, H, W)
                ).copy()
                src = cvcuda.as_tensor(src_data, "NCHW")

                dst_data = cp.zeros((N, num_channels, H, W), dtype=cupy_dtype)
                dst = cvcuda.as_tensor(dst_data, "NCHW")
            else:
                src_data = cp.broadcast_to(
                    gradient.reshape(1, 1, W, 1), (N, H, W, num_channels)
                ).copy()
                src = cvcuda.as_tensor(src_data, "NHWC")

                dst_data = cp.zeros((N, H, W, num_channels), dtype=cupy_dtype)
                dst = cvcuda.as_tensor(dst_data, "NHWC")

        def run(launch):
            cvcuda.median_blur_into(dst, src, ksize, stream=get_stream(launch))

        return run

    else:  # ImageBatchVarShape mode
        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="gradient_h",
        )
        dst = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode=0,
        )

        with cp.cuda.Device(device_id):
            ksize_data = cp.tile(cp.array(ksize, dtype=cp.int32), (N, 1))
            ksize_tensor = cvcuda.as_tensor(ksize_data, "NC")

        def run(launch):
            cvcuda.median_blur_into(dst, src, ksize_tensor, stream=get_stream(launch))

        return run


if __name__ == "__main__":
    run_benchmark("medianblur", medianblur)
