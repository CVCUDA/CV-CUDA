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

"""CV-CUDA GammaContrast operator benchmark - Python equivalent of BenchGammaContrast.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    parse_shape,
    get_dtype,
    get_num_channels,
    get_dtype_size,
    get_format_from_dtype,
    create_image_batch_varshape,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def gammacontrast(state):
    """GammaContrast operator benchmark matching C++ BenchGammaContrast.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    input_kind = state.get_string("inputKind")
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"

    # ScalarGamma benches the host-scalar gamma/gain overload: same dense-tensor input as
    # Tensor, but gamma/gain are kernel launch arguments -- no device gamma tensor is staged.
    if input_kind not in ("Tensor", "VarShape", "ScalarGamma"):
        state.skip(
            "GammaContrast benchmark supports only Tensor, VarShape, and ScalarGamma"
            " input kinds"
        )
        return None
    is_scalar_gamma = input_kind == "ScalarGamma"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "GammaContrast benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind != "Tensor":
        state.skip("Fake-planar (NCHW_FAKE) GammaContrast benchmark is tensor-only")
        return None
    if is_planar and input_kind == "VarShape" and dtype_str == "uchar4":
        state.skip("RGBA8p varshape is unsupported by the Python image API")
        return None

    device_id = state.get_device()

    num_channels = get_num_channels(dtype_str)
    dtype_size = get_dtype_size(dtype_str)

    image_bytes = N * H * W * dtype_size
    gamma_bytes = 0 if is_scalar_gamma else N * num_channels * 4
    if is_fake_planar:
        state.add_global_memory_reads(3 * image_bytes + gamma_bytes)
        state.add_global_memory_writes(3 * image_bytes)
    else:
        state.add_global_memory_reads(image_bytes + gamma_bytes)
        state.add_global_memory_writes(image_bytes)

    # ScalarGamma passes the same 0.75 by value (gain 1.0) instead of staging this tensor.
    if not is_scalar_gamma:
        gamma = create_tensor(
            (N * num_channels,), cvcuda.Type.F32, device_id, layout="N", fill_mode=0.75
        )

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

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.gamma_contrast_into(inter_dst, inter_src, gamma, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind in ("Tensor", "ScalarGamma"):  # dense-tensor modes
        tensor_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        src = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode="lcg",
        )
        dst = create_tensor(tensor_shape, dtype, device_id, layout=layout, fill_mode=0)
    else:  # ImageBatchVarShape mode
        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype_str,
            device=device_id,
            fill_mode="lcg",
        )
        dst = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype_str,
            device=device_id,
            fill_mode=0,
        )

    if is_scalar_gamma:

        def run(launch):
            cvcuda.gamma_contrast_into(dst, src, 0.75, 1.0, stream=get_stream(launch))

    else:

        def run(launch):
            cvcuda.gamma_contrast_into(dst, src, gamma, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("gammacontrast", gammacontrast)
