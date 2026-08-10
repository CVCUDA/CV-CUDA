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

"""CV-CUDA Rotate operator benchmark - Python equivalent of BenchRotate.cpp"""

import math
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
    get_interpolation_type,
    get_format_from_dtype,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def _rotate_in_bounds_fraction(angle_deg, x_shift, y_shift, w, h):
    """Fraction of dst pixels in [0,W]x[0,H] whose inverse rotation lands inside source.

    The rotate kernel skips both read and write outside this region (rotate.cu:61),
    so the byte model is scaled by this fraction to track HBM traffic.
    """
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)

    # Kernel mapping is src = R(+θ)·(dst - shift), so the in-bounds dst region is
    # the image of [0,W]x[0,H] under dst = R(-θ)·src + shift.
    def m(x, y):
        return (c * x + s * y + x_shift, -s * x + c * y + y_shift)

    poly = [m(0.0, 0.0), m(w, 0.0), m(w, h), m(0.0, h)]

    def clip(in_pts, axis, bound, keep_above):
        if not in_pts:
            return []
        out = []
        for i, cur in enumerate(in_pts):
            prev = in_pts[i - 1]
            ci = (cur[axis] >= bound) if keep_above else (cur[axis] <= bound)
            pi = (prev[axis] >= bound) if keep_above else (prev[axis] <= bound)
            if ci:
                if not pi:
                    t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                    out.append(
                        (
                            prev[0] + t * (cur[0] - prev[0]),
                            prev[1] + t * (cur[1] - prev[1]),
                        )
                    )
                out.append(cur)
            elif pi:
                t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                out.append(
                    (prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1]))
                )
        return out

    poly = clip(poly, 0, 0.0, True)
    poly = clip(poly, 0, w, False)
    poly = clip(poly, 1, 0.0, True)
    poly = clip(poly, 1, h, False)
    if len(poly) < 3:
        return 0.0
    area = 0.0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) * 0.5 / (w * h)


def rotate(state):
    """Rotate operator benchmark matching C++ BenchRotate.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    _ = get_dtype(dtype_str)
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    input_kind = get_input_kind(state.get_string("inputKind"))
    interp = get_interpolation_type(state.get_string("interpolation"))
    device_id = state.get_device()

    N, H, W = shape
    nc = get_num_channels(dtype_str)

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("Rotate benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) rotate benchmark is tensor-only")
        return None

    # Rotation around image center: keeps the bulk of dst pixels inside source so
    # the kernel actually exercises read+interp+write rather than the out-of-bounds
    # early-exit guard at rotate.cu:61.
    angle_deg = 30.0
    th = math.radians(angle_deg)
    cx, cy = W * 0.5, H * 0.5
    x_shift = cx * (1.0 - math.cos(th)) - cy * math.sin(th)
    y_shift = cy * (1.0 - math.cos(th)) + cx * math.sin(th)
    shift = (x_shift, y_shift)

    in_bounds_frac = _rotate_in_bounds_fraction(angle_deg, x_shift, y_shift, W, H)
    bytes_full = N * H * W * get_dtype_size(dtype_str)
    # Native rotate only touches the in-bounds region; the two reformats in the fake-planar path move
    # the full tensor each way (reformat is full-coverage), so add 2*bytes_full on top of the rotate.
    bytes_io = int(
        (2 * bytes_full if is_fake_planar else 0) + in_bounds_frac * bytes_full
    )
    state.add_global_memory_reads(bytes_io)
    state.add_global_memory_writes(bytes_io)

    get_stream = create_stream_cache()

    if is_fake_planar:
        # Tensor-only "fake planar": planar->interleaved->rotate->interleaved->planar,
        # all timed, as the comparison baseline for the native planar (NCHW) path.
        src = create_tensor(
            (N, nc, H, W), dtype_str, device_id, layout="NCHW", fill_mode="lcg"
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
            cvcuda.rotate_into(
                inter_dst, inter_src, angle_deg, shift, interp, stream=stream
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":
        tshape = (N, nc, H, W) if is_planar else (N, H, W, nc)
        src = create_tensor(
            tshape, dtype_str, device_id, layout=layout, fill_mode="lcg"
        )
        dst = create_tensor(tshape, dtype_str, device_id, layout=layout, fill_mode=0)

        def run(launch):
            stream = get_stream(launch)
            cvcuda.rotate_into(dst, src, angle_deg, shift, interp, stream=stream)

        return run

    else:
        img_format = get_format_from_dtype(dtype_str, nc, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, nc),
            0,
            img_format,
            dtype=dtype_str,
            device=device_id,
            fill_mode="lcg",
        )
        dst = create_image_batch_varshape(
            (N, H, W, nc),
            0,
            img_format,
            dtype=dtype_str,
            device=device_id,
            fill_mode=0,
        )

        with cp.cuda.Device(device_id):
            angle_data = cp.full((N,), angle_deg, dtype=cp.float64)
            shift_data = cp.zeros((N, 2), dtype=cp.float64)
            shift_data[:, 0] = shift[0]
            shift_data[:, 1] = shift[1]
        angle_tensor = cvcuda.as_tensor(angle_data, "N")
        shift_tensor = cvcuda.as_tensor(shift_data, "NW")

        def run(launch):
            stream = get_stream(launch)
            cvcuda.rotate_into(
                dst, src, angle_tensor, shift_tensor, interp, stream=stream
            )

        return run


if __name__ == "__main__":
    run_benchmark("rotate", rotate)
