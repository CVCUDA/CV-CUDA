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

"""CV-CUDA AdaptiveThreshold operator benchmark - Python equivalent of BenchAdaptiveThreshold.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_dtype_size,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    get_format_from_dtype,
    run_benchmark,
)


def adaptivethreshold(state):
    """AdaptiveThreshold operator benchmark matching C++ BenchAdaptiveThreshold.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype = get_dtype(state.get_string("InOutDataType"))
    device_id = state.get_device()
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "AdaptiveThreshold benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None
    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if (is_planar or is_fake_planar) and input_kind == "VarShape":
        state.skip("Planar AdaptiveThreshold benchmark rows are tensor-only")
        return None
    # Adaptive threshold parameters (match C++ values)
    maxValue = 123.0
    adaptiveMethod = cvcuda.AdaptiveThresholdType.GAUSSIAN_C
    thresholdType = cvcuda.ThresholdType.BINARY
    blockSize = int(state.get_int64("blockSize"))  # Get from config like C++
    c = -2.3

    # Report memory (match C++)
    dtype_size = get_dtype_size(dtype)
    bytes_ = N * H * W * dtype_size
    if is_fake_planar:
        state.add_global_memory_reads(3 * bytes_)
        state.add_global_memory_writes(3 * bytes_)
    else:
        state.add_global_memory_reads(bytes_)
        state.add_global_memory_writes(bytes_)

    get_stream = create_stream_cache()

    def do_adaptivethreshold(dst, src, stream):
        cvcuda.adaptivethreshold_into(
            dst,
            src,
            maxValue,
            adaptiveMethod,
            thresholdType,
            blockSize,
            c,
            stream=stream,
        )

    if is_fake_planar:
        src = create_tensor(
            (N, 1, H, W), dtype, device_id, layout="NCHW", fill_mode="checkerboard"
        )
        inter_src = create_tensor(
            (N, H, W, 1), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, 1), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor((N, 1, H, W), dtype, device_id, layout="NCHW", fill_mode=0)

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            do_adaptivethreshold(inter_dst, inter_src, stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode (matches C++ line 47)
        tensor_shape = (N, 1, H, W) if is_planar else (N, H, W, 1)
        src = create_tensor(
            tensor_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(tensor_shape, dtype, device_id, layout=layout, fill_mode=0)

        def run(launch):
            do_adaptivethreshold(dst, src, get_stream(launch))

        return run

    else:  # ImageBatchVarShape mode (matches C++ line 60)
        # Determine image format (single channel grayscale)
        dtype_str = state.get_string("InOutDataType")
        img_format = get_format_from_dtype(dtype_str, 1)

        # Create src and dst batches independently using utility
        src = create_image_batch_varshape(
            (N, H, W, 1),
            0,
            img_format,
            dtype_str,
            device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, H, W, 1), 0, img_format, dtype_str, device_id, fill_mode=0
        )

        # Create parameter tensors (match C++ lines 69-75)
        maxValue_tensor = create_tensor(
            (N,), cvcuda.Type.F64, device_id, layout="N", fill_mode=maxValue
        )
        blockSize_tensor = create_tensor(
            (N,), cvcuda.Type.S32, device_id, layout="N", fill_mode=blockSize
        )
        c_tensor = create_tensor(
            (N,), cvcuda.Type.F64, device_id, layout="N", fill_mode=c
        )

        def run(launch):
            cvcuda.adaptivethreshold_into(
                dst,
                src,
                maxValue_tensor,
                adaptiveMethod,
                thresholdType,
                blockSize,  # max_block_size (int)
                blockSize_tensor,  # block_size tensor
                c_tensor,
                stream=get_stream(launch),
            )

        return run


if __name__ == "__main__":
    run_benchmark("adaptivethreshold", adaptivethreshold)
