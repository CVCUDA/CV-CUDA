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

"""CV-CUDA WarpAffine operator benchmark - Python equivalent of BenchWarpAffine.cpp"""

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
    get_num_channels,
    get_format_from_dtype,
    get_border_type,
    get_interpolation_type,
    create_tensor,
    create_stream_cache,
    create_image_batch_varshape,
    run_benchmark,
)


def warpaffine(state):
    """WarpAffine operator benchmark matching C++ BenchWarpAffine.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    border = get_border_type(state.get_string("border"))
    interp = get_interpolation_type(state.get_string("interpolation"))
    inverse_map = state.get_string("inverseMap") == "Y"
    input_kind = get_input_kind(state.get_string("inputKind"))
    device_id = state.get_device()

    N, H, W = shape
    nc = get_num_channels(dtype_str)

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "WarpAffine benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) WarpAffine benchmark is tensor-only")
        return None

    flags = interp
    if inverse_map:
        flags = flags | cvcuda.Interp.WARP_INVERSE_MAP

    # WarpAffine preserves size; the 6-float matrix is also read each launch.
    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    if is_fake_planar:
        state.add_global_memory_reads(3 * bytes_ + 6 * 4)
        state.add_global_memory_writes(3 * bytes_)
    else:
        state.add_global_memory_reads(bytes_ + 6 * 4)
        state.add_global_memory_writes(bytes_)

    get_stream = create_stream_cache()

    if is_fake_planar:
        # Tensor-only "fake planar": planar->interleaved->warp->interleaved->planar, all timed, as the
        # comparison baseline for the native planar (NCHW) path.
        src = create_tensor(
            (N, nc, H, W), dtype, device_id, layout="NCHW", fill_mode="checkerboard"
        )
        inter_src = create_tensor(
            (N, H, W, nc), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, nc), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor((N, nc, H, W), dtype, device_id, layout="NCHW", fill_mode=0)

        xform = [[2.0, 2.0, 0.0], [3.0, 1.0, 0.0]]

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.warp_affine_into(
                inter_dst,
                inter_src,
                xform,
                flags=flags,
                border_mode=border,
                border_value=(0, 0, 0, 0),
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":
        tshape = (N, nc, H, W) if is_planar else (N, H, W, nc)
        src = create_tensor(
            tshape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(tshape, dtype, device_id, layout=layout, fill_mode=0)

        xform = [[2.0, 2.0, 0.0], [3.0, 1.0, 0.0]]

        def run(launch):
            stream = get_stream(launch)
            cvcuda.warp_affine_into(
                dst,
                src,
                xform,
                flags=flags,
                border_mode=border,
                border_value=(0, 0, 0, 0),
                stream=stream,
            )

        return run

    else:
        img_format = get_format_from_dtype(dtype_str, nc, planar=is_planar)

        src = create_image_batch_varshape(
            (N, H, W, nc),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, H, W, nc), 0, img_format, dtype, device_id, fill_mode=0
        )

        xform_data = cp.array([[2.0, 2.0, 0.0, 3.0, 1.0, 0.0]] * N, dtype=cp.float32)
        xform_tensor = cvcuda.as_tensor(xform_data, "NW")

        def run(launch):
            stream = get_stream(launch)
            cvcuda.warp_affine_into(
                dst,
                src,
                xform_tensor,
                flags=flags,
                border_mode=border,
                border_value=(0, 0, 0, 0),
                stream=stream,
            )

        return run


if __name__ == "__main__":
    run_benchmark("warpaffine", warpaffine)
