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

"""CV-CUDA Gaussian operator benchmark - Python equivalent of BenchGaussian.cpp"""

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
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def gaussian(state):
    """Gaussian operator benchmark matching C++ BenchGaussian.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    sigma = state.get_float64("sigma")
    border = get_border_type(state.get_string("border"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    device_id = state.get_device()
    nc = get_num_channels(dtype_str)

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("Gaussian benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) Gaussian benchmark is tensor-only")
        return None

    mult = 3 if get_dtype(dtype_str) == cvcuda.Type.U8 else 4
    kernel_size = round(sigma * mult * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    if is_fake_planar:
        state.add_global_memory_reads(3 * bytes_)
        state.add_global_memory_writes(3 * bytes_)
    else:
        state.add_global_memory_reads(bytes_)
        state.add_global_memory_writes(bytes_)

    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, nc, H, W), dtype_str, device_id, layout="NCHW", fill_mode="checkerboard"
        )
        inter_src = create_tensor(
            (N, H, W, nc), dtype_str, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, nc), dtype_str, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, nc, H, W), dtype_str, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.gaussian_into(
                inter_dst,
                inter_src,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma),
                border=border,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        tensor_shape = (N, nc, H, W) if is_planar else (N, H, W, nc)
        src = create_tensor(
            tensor_shape, dtype_str, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(
            tensor_shape, dtype_str, device_id, layout=layout, fill_mode=0
        )

        def run(launch):
            cvcuda.gaussian_into(
                dst,
                src,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma),
                border=border,
                stream=get_stream(launch),
            )

        return run
    else:  # ImageBatchVarShape mode
        if is_planar and get_dtype(dtype_str) == cvcuda.Type.U8 and nc == 4:
            state.skip(
                "Gaussian RGBA8p planar var-shape is unsupported by the Python image API"
            )
            return None

        img_format = get_format_from_dtype(dtype_str, nc, planar=is_planar)

        src = create_image_batch_varshape(
            (N, H, W, nc),
            0,
            img_format,
            dtype_str,
            device=device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, H, W, nc), 0, img_format, dtype_str, device=device_id, fill_mode=0
        )
        kernel_size_tensor = create_tensor(
            (N, 2),
            cvcuda.Type.S32,
            device_id,
            layout="NC",
            fill_mode=(kernel_size, kernel_size),
        )
        sigma_tensor = create_tensor(
            (N, 2), cvcuda.Type.F64, device_id, layout="NC", fill_mode=(sigma, sigma)
        )

        def run(launch):
            cvcuda.gaussian_into(
                src=src,
                dst=dst,
                max_kernel_size=(kernel_size, kernel_size),
                kernel_size=kernel_size_tensor,
                sigma=sigma_tensor,
                border=border,
                stream=get_stream(launch),
            )

        return run


if __name__ == "__main__":
    run_benchmark("gaussian", gaussian)
