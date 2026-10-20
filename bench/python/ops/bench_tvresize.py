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

"""TvResize representative timings; frozen diagnostic inventory is not auto-registered."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp  # noqa: E402
import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_num_channels,
    get_format_from_dtype,
    create_stream_cache,
    run_benchmark,
)


def _sizes(count, height, width, batch_case, output=False):
    if count < 0:
        raise ValueError("Invalid TvResize batchCase/count")
    if batch_case == "uniform":
        return [(height, width)] * count
    if (
        count <= 0
        or count % 3 != 0
        or batch_case not in ("ordinary", "mixed-singleton")
    ):
        raise ValueError("Invalid TvResize batchCase/count")
    if output:
        return [(720, 1280), (1080, 1920), (4, 3)] * (count // 3)
    return [(1080, 1920), (2160, 3840), (7 if batch_case == "ordinary" else 1, 7)] * (
        count // 3
    )


def _ramp(height, width, channels, sample, dtype):
    values = (
        cp.arange(height * width * channels, dtype=cp.int64) * 17 + sample * 13
    ) % 251
    if dtype == cp.float32:
        values = values.astype(cp.float32) / cp.float32(251)
    return values.astype(dtype).reshape(height, width, channels)


def tvresize(state):
    count, height, width = parse_shape(state.get_string("shape"))
    out_h, out_w = parse_shape(state.get_string("outputShape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    channels = int(state.get_int64("numChannels"))
    layout = state.get_string("layout")
    kind = state.get_string("inputKind")
    batch_case = state.get_string("batchCase")
    mode = state.get_string("interpolation")
    interp = getattr(cvcuda.Interp, mode)
    antialias = bool(state.get_int64("antialias"))
    fake = layout == "NCHW_FAKE"
    planar = layout in ("NCHW", "NCHW_FAKE")
    if (
        (layout != "NHWC" and not planar)
        or kind not in ("Tensor", "VarShape")
        or (
            kind == "VarShape"
            and (layout != "NHWC" or channels != get_num_channels(dtype_str))
        )
        or (kind == "Tensor" and batch_case != "uniform")
        or (fake and channels > 4)
    ):
        state.skip("Unsupported TvResize benchmark axis combination")
        return None

    src_sizes = _sizes(count, height, width, batch_case)
    dst_sizes = _sizes(count, out_h, out_w, batch_case, output=True)
    lane_bytes = get_dtype_size(dtype)
    src_bytes = sum(h * w for h, w in src_sizes) * channels * lane_bytes
    dst_bytes = sum(h * w for h, w in dst_sizes) * channels * lane_bytes
    state.add_global_memory_reads(2 * src_bytes + dst_bytes if fake else src_bytes)
    state.add_global_memory_writes(src_bytes + 2 * dst_bytes if fake else dst_bytes)
    get_stream = create_stream_cache()

    with cp.cuda.Device(state.get_device()):
        if kind == "VarShape":
            src = cvcuda.ImageBatchVarShape(count)
            dst = cvcuda.ImageBatchVarShape(count)
            fmt = get_format_from_dtype(dtype_str, channels)
            for n, ((h, w), (dh, dw)) in enumerate(zip(src_sizes, dst_sizes)):
                image = cvcuda.Image((w, h), fmt)
                view = cp.asarray(image.cuda())
                view[...] = _ramp(h, w, channels, n, view.dtype).reshape(view.shape)
                src.pushback(image)
                dst.pushback(cvcuda.Image((dw, dh), fmt))
        else:
            actual_layout = "NCHW" if planar else "NHWC"
            src_shape = (
                (count, channels, height, width)
                if planar
                else (count, height, width, channels)
            )
            dst_shape = (
                (count, channels, out_h, out_w)
                if planar
                else (count, out_h, out_w, channels)
            )
            src = cvcuda.Tensor(src_shape, dtype, actual_layout)
            dst = cvcuda.Tensor(dst_shape, dtype, actual_layout)
            view = cp.asarray(src.cuda())
            for n in range(count):
                values = _ramp(height, width, channels, n, view.dtype)
                view[n] = values.transpose(2, 0, 1) if planar else values
            if fake:
                inter_src = cvcuda.Tensor(
                    (count, height, width, channels), dtype, "NHWC"
                )
                inter_dst = cvcuda.Tensor(
                    (count, out_h, out_w, channels), dtype, "NHWC"
                )
        # Inputs must be ready before nvbench measures its own stream.
        cp.cuda.get_current_stream().synchronize()

    if fake:

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.tvresize_into(
                inter_dst,
                inter_src,
                interp=interp,
                antialias=antialias,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    def run(launch):
        cvcuda.tvresize_into(
            dst, src, interp=interp, antialias=antialias, stream=get_stream(launch)
        )

    return run


if __name__ == "__main__":
    run_benchmark("tvresize", tvresize)
