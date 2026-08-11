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

"""CV-CUDA Normalize operator benchmark - Python equivalent of BenchNormalize.cpp"""

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


def normalize(state):
    """Normalize operator benchmark matching C++ BenchNormalize.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    input_kind_name = state.get_string("inputKind")
    scalar_params = input_kind_name == "TensorScalar"
    input_kind = "Tensor" if scalar_params else get_input_kind(input_kind_name)
    device_id = state.get_device()

    N, H, W = shape
    dtype_size = get_dtype_size(dtype_str)

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "Normalize benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) normalize benchmark is tensor-only")
        return None

    param_samples = 1 if is_planar and input_kind == "VarShape" else N
    param_bytes = 0 if scalar_params else param_samples * num_channels * 4 * 2
    bytes_ = N * H * W * dtype_size
    if is_fake_planar:
        # reformat(NCHW->NHWC) + normalize + reformat(NHWC->NCHW); normalize preserves size.
        state.add_global_memory_reads(3 * bytes_ + param_bytes)
        state.add_global_memory_writes(3 * bytes_)
    else:
        state.add_global_memory_reads(bytes_ + param_bytes)
        state.add_global_memory_writes(bytes_)

    get_stream = create_stream_cache()

    global_scale = 1.234
    global_shift = 2.345
    epsilon = 12.34
    flags = cvcuda.NormalizeFlags.SCALE_IS_STDDEV

    if scalar_params:
        if is_fake_planar:
            state.skip(
                "Scalar-parameter normalize benchmark supports only native NHWC and NCHW tensors"
            )
            return None

        input_shape = (
            (N, num_channels, H, W) if layout == "NCHW" else (N, H, W, num_channels)
        )
        src = create_tensor(
            input_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(input_shape, dtype, device_id, layout=layout, fill_mode=0)
        base = [0.25, 0.5, 0.75, 1.0][:num_channels]
        scale = [1.25, 1.5, 1.75, 2.0][:num_channels]

        def run_scalar(launch):
            cvcuda.normalize_into(
                dst,
                src,
                base,
                scale,
                flags,
                globalscale=global_scale,
                globalshift=global_shift,
                epsilon=epsilon,
                stream=get_stream(launch),
            )

        return run_scalar

    # Fake-planar runs the interleaved kernel, so its base/scale are interleaved (NHWC).
    base_shape = (
        (param_samples, num_channels, 1, 1)
        if is_planar
        else (param_samples, 1, 1, num_channels)
    )
    param_layout = "NCHW" if is_planar else "NHWC"

    # base, scale: deterministic LCG over float [-1, +1], bit-identical to
    # the C++ RandomValues<float>() GPU fast path (see BenchNormalize.cpp).
    base = create_tensor(
        base_shape,
        cvcuda.Type.F32,
        device_id,
        layout=param_layout,
        fill_mode="lcg",
    )
    scale = create_tensor(
        base_shape,
        cvcuda.Type.F32,
        device_id,
        layout=param_layout,
        fill_mode="lcg",
    )

    if is_fake_planar:
        # Tensor-only "fake planar": planar->interleaved->normalize->interleaved->planar,
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
            cvcuda.normalize_into(
                inter_dst,
                inter_src,
                base,
                scale,
                flags,
                globalscale=global_scale,
                globalshift=global_shift,
                epsilon=epsilon,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        input_shape = (
            (N, num_channels, H, W) if layout == "NCHW" else (N, H, W, num_channels)
        )
        src = create_tensor(
            input_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(input_shape, dtype, device_id, layout=layout, fill_mode=0)

        def run(launch):
            cvcuda.normalize_into(
                dst,
                src,
                base,
                scale,
                flags,
                globalscale=global_scale,
                globalshift=global_shift,
                epsilon=epsilon,
                stream=get_stream(launch),
            )

        return run

    else:  # ImageBatchVarShape mode
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
            cvcuda.normalize_into(
                dst,
                src,
                base,
                scale,
                flags,
                globalscale=global_scale,
                globalshift=global_shift,
                epsilon=epsilon,
                stream=get_stream(launch),
            )

        return run


if __name__ == "__main__":
    run_benchmark("normalize", normalize)
