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

"""CV-CUDA Reformat operator benchmark - Python equivalent of BenchReformat.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_input_kind,
    get_num_channels,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def reformat(state):
    """Reformat operator benchmark matching C++ BenchReformat.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    nc = get_num_channels(dtype_str)
    input_kind = get_input_kind(state.get_string("inputKind"))
    row_alignment = int(state.get_int64("rowAlignment"))
    device_id = state.get_device()

    if row_alignment < 0:
        raise ValueError("rowAlignment must be non-negative")

    N, H, W = shape

    dtype_size = get_dtype_size(dtype_str)
    state.add_global_memory_reads(N * H * W * dtype_size)
    state.add_global_memory_writes(N * H * W * dtype_size)

    if input_kind != "Tensor":
        state.skip("ImageBatchVarShape not implemented for this operator")
        return None

    # Reformat converts planar NCHW -> interleaved NHWC; use the real channel
    # count so both layouts move the same bytes as the C++ benchmark.
    if row_alignment > 0:
        src = cvcuda.Tensor((N, nc, H, W), dtype, layout="NCHW", rowalign=row_alignment)
        dst = cvcuda.Tensor((N, H, W, nc), dtype, layout="NHWC", rowalign=row_alignment)
    else:
        src = create_tensor(
            (N, nc, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        dst = create_tensor((N, H, W, nc), dtype, device_id, layout="NHWC", fill_mode=0)

    get_stream = create_stream_cache()

    def run(launch):
        stream = get_stream(launch)
        cvcuda.reformat_into(dst, src, stream=stream)

    return run


if __name__ == "__main__":
    run_benchmark("reformat", reformat)
