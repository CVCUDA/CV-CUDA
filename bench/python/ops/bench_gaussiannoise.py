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

"""CV-CUDA GaussianNoise operator benchmark - Python equivalent of BenchGaussianNoise.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype_size,
    get_num_channels,
    get_format_from_dtype,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def gaussiannoise(state):
    """GaussianNoise operator benchmark matching C++ BenchGaussianNoise.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    device_id = state.get_device()

    num_channels = get_num_channels(dtype_str)
    per_channel = num_channels > 1
    seed = 12345

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "GaussianNoise benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) GaussianNoise benchmark is tensor-only")
        return None

    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    state.add_global_memory_reads((3 if is_fake_planar else 1) * bytes_)
    state.add_global_memory_writes((3 if is_fake_planar else 1) * bytes_)

    mu = create_tensor((N,), "float32", device_id, layout="N", fill_mode=0.5)
    sigma = create_tensor((N,), "float32", device_id, layout="N", fill_mode=0.075)
    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, num_channels, H, W),
            dtype_str,
            device_id,
            layout="NCHW",
            fill_mode="lcg",
        )
        inter_src = create_tensor(
            (N, H, W, num_channels), dtype_str, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, num_channels), dtype_str, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, num_channels, H, W), dtype_str, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.gaussiannoise_into(
                src=inter_src,
                dst=inter_dst,
                mu=mu,
                sigma=sigma,
                per_channel=per_channel,
                seed=seed,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        input_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        src = create_tensor(
            input_shape,
            dtype_str,
            device_id,
            layout="NCHW" if is_planar else "NHWC",
            fill_mode="lcg",
        )
        dst = create_tensor(
            input_shape,
            dtype_str,
            device_id,
            layout="NCHW" if is_planar else "NHWC",
            fill_mode=0,
        )
    else:  # ImageBatchVarShape mode
        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype_str,
            device_id,
            fill_mode="lcg",
        )
        dst = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype_str,
            device_id,
            fill_mode=0,
        )

    def run(launch):
        cvcuda.gaussiannoise_into(
            src=src,
            dst=dst,
            mu=mu,
            sigma=sigma,
            per_channel=per_channel,
            seed=seed,
            stream=get_stream(launch),
        )

    return run


if __name__ == "__main__":
    run_benchmark("gaussiannoise", gaussiannoise)
