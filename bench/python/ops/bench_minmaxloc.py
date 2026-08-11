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

"""CV-CUDA MinMaxLoc operator benchmark - Python equivalent of BenchMinMaxLoc.cpp"""

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
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def minmaxloc(state):
    """MinMaxLoc operator benchmark matching C++ BenchMinMaxLoc.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype = get_dtype(state.get_string("InOutDataType"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    max_locs = state.get_int64("maxLocations")
    run_choice = state.get_string("runChoice")
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    device_id = state.get_device()

    N, H, W = shape
    dtype_size = get_dtype_size(dtype)

    if run_choice not in ("MIN", "MAX", "MIN_MAX"):
        raise ValueError("runChoice must be MIN, MAX, or MIN_MAX")
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "MinMaxLoc benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None
    if input_kind != "Tensor" and layout != "NHWC":
        state.skip("Planar MinMaxLoc benchmark is tensor-only")
        return None

    run_min = run_choice in ("MIN", "MIN_MAX")
    run_max = run_choice in ("MAX", "MIN_MAX")
    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"

    input_bytes = N * H * W * dtype_size
    state.add_global_memory_reads((3 if is_fake_planar else 2) * input_bytes)
    state.add_global_memory_writes(
        (int(run_min) + int(run_max)) * N * (4 + max_locs * 8 + 4)
        + (input_bytes if is_fake_planar else 0)
    )

    get_stream = create_stream_cache()

    if dtype in (cvcuda.Type.S8, cvcuda.Type.S16, cvcuda.Type.S32):
        val_dtype = cvcuda.Type.S32
    elif dtype in (cvcuda.Type.U8, cvcuda.Type.U16, cvcuda.Type.U32):
        val_dtype = cvcuda.Type.U32
    else:
        val_dtype = dtype

    val_dtype_map = {
        cvcuda.Type.U32: cp.uint32,
        cvcuda.Type.S32: cp.int32,
        cvcuda.Type.F32: cp.float32,
        cvcuda.Type.F64: cp.float64,
    }

    with cp.cuda.Device(device_id):
        min_val_data = cp.zeros((N, 1), dtype=val_dtype_map[val_dtype])
        min_val = cvcuda.as_tensor(min_val_data, "NC")

        min_loc_data = cp.zeros((N, max_locs, 2), dtype=cp.int32)
        min_loc = cvcuda.as_tensor(min_loc_data, "NMC")

        num_min_data = cp.zeros((N, 1), dtype=cp.int32)
        num_min = cvcuda.as_tensor(num_min_data, "NC")

        max_val_data = cp.zeros((N, 1), dtype=val_dtype_map[val_dtype])
        max_val = cvcuda.as_tensor(max_val_data, "NC")

        max_loc_data = cp.zeros((N, max_locs, 2), dtype=cp.int32)
        max_loc = cvcuda.as_tensor(max_loc_data, "NMC")

        num_max_data = cp.zeros((N, 1), dtype=cp.int32)
        num_max = cvcuda.as_tensor(num_max_data, "NC")

    def run_op(stream, src):
        if run_choice == "MIN":
            cvcuda.min_loc_into(min_val, min_loc, num_min, src, stream=stream)
        elif run_choice == "MAX":
            cvcuda.max_loc_into(max_val, max_loc, num_max, src, stream=stream)
        else:
            cvcuda.min_max_loc_into(
                min_val,
                min_loc,
                num_min,
                max_val,
                max_loc,
                num_max,
                src,
                stream=stream,
            )

    if input_kind == "Tensor":  # Tensor mode
        tensor_shape = (N, 1, H, W) if (is_planar or is_fake_planar) else (N, H, W, 1)
        src = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout="NCHW" if (is_planar or is_fake_planar) else "NHWC",
            fill_mode="lcg",
        )

        if is_fake_planar:
            inter_src = create_tensor(
                (N, H, W, 1), dtype, device_id, layout="NHWC", fill_mode=0
            )

            def run_fake(launch):
                stream = get_stream(launch)
                cvcuda.reformat_into(inter_src, src, stream=stream)
                run_op(stream, inter_src)

            return run_fake

        def run(launch):
            run_op(get_stream(launch), src)

        return run

    else:  # ImageBatchVarShape mode
        if dtype == cvcuda.Type.U8:
            img_format = cvcuda.Format.U8
        elif dtype == cvcuda.Type.U16:
            img_format = cvcuda.Format.U16
        elif dtype == cvcuda.Type.U32:
            img_format = cvcuda.Format.U32
        elif dtype == cvcuda.Type.S16:
            img_format = cvcuda.Format.S16
        elif dtype == cvcuda.Type.S32:
            img_format = cvcuda.Format.S32
        else:
            img_format = cvcuda.Format.F32

        src = create_image_batch_varshape(
            (N, H, W), 0, img_format, dtype, device_id, fill_mode="lcg"
        )

        def run(launch):
            run_op(get_stream(launch), src)

        return run


if __name__ == "__main__":
    run_benchmark("minmaxloc", minmaxloc)
