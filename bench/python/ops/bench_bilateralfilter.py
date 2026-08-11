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

"""CV-CUDA BilateralFilter operator benchmark - Python equivalent of BenchBilateralFilter.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_num_channels,
    get_dtype_size,
    get_format_from_dtype,
    get_border_type,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def bilateralfilter(state):
    """BilateralFilter operator benchmark matching C++ BenchBilateralFilter.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    border = get_border_type(state.get_string("border"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except KeyError:
        layout = "NHWC"
    device_id = state.get_device()

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "BilateralFilter benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None
    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) BilateralFilter benchmark is tensor-only")
        return None

    # Bilateral filter parameters (match C++ lines 32-34)
    diameter_scalar = int(state.get_int64("diameter"))  # -1 from config
    sigma_space_scalar = float(state.get_float64("sigmaSpace"))  # 1.2 from config
    sigma_color_scalar = -1.0  # hardcoded like C++

    # Report memory (match C++)
    dtype_size = get_dtype_size(dtype_str)
    state.add_global_memory_reads(N * H * W * dtype_size)
    state.add_global_memory_writes(N * H * W * dtype_size)

    get_stream = create_stream_cache()

    def do_bilateralfilter(dst, src, diameter, sigma_color, sigma_space, stream):
        cvcuda.bilateral_filter_into(
            dst,
            src,
            diameter,
            sigma_color,
            sigma_space,
            border=border,
            stream=stream,
        )

    if is_fake_planar:
        src = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="lcg",
        )
        inter_src = create_tensor(
            (N, H, W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        inter_dst = create_tensor(
            (N, H, W, num_channels),
            dtype,
            device_id,
            layout="NHWC",
            fill_mode=0,
        )
        dst = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode=0,
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            do_bilateralfilter(
                inter_dst,
                inter_src,
                diameter_scalar,
                sigma_color_scalar,
                sigma_space_scalar,
                stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        diameter = diameter_scalar
        sigma_space = sigma_space_scalar
        sigma_color = sigma_color_scalar
        tensor_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        src = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode="lcg",
        )
        dst = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode=0,
        )
    else:  # ImageBatchVarShape mode
        diameter = create_tensor(
            (N,), cvcuda.Type.S32, device_id, layout="N", fill_mode=diameter_scalar
        )
        sigma_space = create_tensor(
            (N,), cvcuda.Type.F32, device_id, layout="N", fill_mode=sigma_space_scalar
        )
        sigma_color = create_tensor(
            (N,), cvcuda.Type.F32, device_id, layout="N", fill_mode=sigma_color_scalar
        )
        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="lcg",
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
        do_bilateralfilter(
            dst,
            src,
            diameter,
            sigma_color,
            sigma_space,
            get_stream(launch),
        )

    return run


if __name__ == "__main__":
    run_benchmark("bilateralfilter", bilateralfilter)
