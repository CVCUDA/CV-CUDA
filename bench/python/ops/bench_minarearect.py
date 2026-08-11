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

"""MinAreaRect operator benchmark matching C++ BenchMinAreaRect.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp  # noqa: E402
import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_dtype_size,
    create_stream_cache,
    run_benchmark,
    _lcg_fill,
    _LCG_SEED,
)


def minarearect(state):
    """MinAreaRect operator benchmark matching C++ BenchMinAreaRect.cpp"""

    shape_str = state.get_string("shape")
    input_kind = get_input_kind(state.get_string("inputKind"))
    dtype_str = state.get_string("InOutDataType")
    num_points_axis = state.get_int64("numPoints")
    device_id = state.get_device()

    shape = parse_shape(shape_str)
    N = shape[0]
    max_points = shape[1]

    if num_points_axis < 0 or num_points_axis > max_points:
        raise ValueError("numPoints must be 0 or in [1, max_points]")

    dtype = get_dtype(dtype_str)
    cupy_dtype_map = {
        cvcuda.Type.U16: cp.uint16,
        cvcuda.Type.S16: cp.int16,
        cvcuda.Type.S32: cp.int32,
    }
    cupy_dtype = cupy_dtype_map.get(dtype)
    if cupy_dtype is None:
        raise ValueError(f"Unsupported MinAreaRect dtype: {dtype_str}")

    dtype_size = get_dtype_size(dtype_str)
    state.add_global_memory_reads(N * max_points * dtype_size)
    state.add_global_memory_writes(N * 8 * 4 + N * 4)

    get_stream = create_stream_cache()

    if input_kind == "Tensor":
        with cp.cuda.Device(device_id):
            # src: deterministic LCG over the full dtype range, matching the
            # C++ LcgValues<T>() GPU fast path.
            src_data = cp.empty((N, max_points, 2), dtype=cupy_dtype)
            _lcg_fill(src_data, _LCG_SEED)
            src = cvcuda.as_tensor(src_data, "NWC")

            dst_data = cp.zeros((N, 8), dtype=cp.float32)
            dst = cvcuda.as_tensor(dst_data, "NW")

            # numPoints=0 preserves the original deterministic [10, 100] cycle.
            if num_points_axis > 0:
                points_data = cp.full((1, N), num_points_axis, dtype=cp.int32)
            else:
                points_data = (
                    cp.arange(N, dtype=cp.int32).reshape(1, N) % 91 + 10
                ).astype(cp.int32)
            num_points = cvcuda.as_tensor(points_data, "NW")

        def run(launch):
            stream = get_stream(launch)
            cvcuda.minarearect_into(dst, src, num_points, N, stream=stream)

        return run

    else:
        state.skip("ImageBatchVarShape not implemented for this operator")
        return None


if __name__ == "__main__":
    run_benchmark("minarearect", minarearect)
