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

"""CV-CUDA CvtColor operator benchmark - Python equivalent of BenchCvtColor.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_format_from_dtype,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def _get_layout(state):
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("CvtColor benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None
    return layout


def _planar_format(img_format):
    if img_format == cvcuda.Format.RGB8:
        return cvcuda.Format.RGB8p
    if img_format == cvcuda.Format.BGR8:
        return cvcuda.Format.BGR8p
    if img_format == cvcuda.Format.RGBA8:
        return cvcuda.Format.RGBA8p
    if img_format == cvcuda.Format.LAB8:
        return cvcuda.Format.LAB8p
    raise ValueError(
        "Planar CvtColor var-shape benchmark supports only RGB/BGR/RGBA/Lab formats"
    )


def _format_for_dtype(img_format, dtype_str, channels, planar):
    dtype = get_dtype(dtype_str)
    if img_format == cvcuda.Format.BGR8:
        if dtype == cvcuda.Type.U8:
            return cvcuda.Format.BGR8p if planar else cvcuda.Format.BGR8
        if dtype == cvcuda.Type.F16:
            return cvcuda.Format.BGRf16p if planar else cvcuda.Format.BGRf16
        if dtype == cvcuda.Type.F32:
            return cvcuda.Format.BGRf32p if planar else cvcuda.Format.BGRf32
    if img_format == cvcuda.Format.LAB8:
        if dtype == cvcuda.Type.U8:
            return cvcuda.Format.LAB8p if planar else cvcuda.Format.LAB8
        if dtype == cvcuda.Type.F16:
            return cvcuda.Format.LABf16p if planar else cvcuda.Format.LABf16
        if dtype == cvcuda.Type.F32:
            return cvcuda.Format.LABf32p if planar else cvcuda.Format.LABf32
    return get_format_from_dtype(dtype_str, channels, planar=planar)


def cvtcolor(state):
    """CvtColor operator benchmark matching C++ BenchCvtColor.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    code_str = state.get_string("code")
    input_kind = get_input_kind(state.get_string("inputKind"))
    layout = _get_layout(state)
    if layout is None:
        return None
    device_id = state.get_device()

    code_map = {
        "RGB2BGR": (
            cvcuda.ColorConversion.RGB2BGR,
            cvcuda.Format.RGB8,
            cvcuda.Format.BGR8,
            3,
            3,
        ),
        "RGB2RGBA": (
            cvcuda.ColorConversion.RGB2RGBA,
            cvcuda.Format.RGB8,
            cvcuda.Format.RGBA8,
            3,
            4,
        ),
        "RGBA2RGB": (
            cvcuda.ColorConversion.RGBA2RGB,
            cvcuda.Format.RGBA8,
            cvcuda.Format.RGB8,
            4,
            3,
        ),
        "RGB2GRAY": (
            cvcuda.ColorConversion.RGB2GRAY,
            cvcuda.Format.RGB8,
            cvcuda.Format.Y8,
            3,
            1,
        ),
        "GRAY2RGB": (
            cvcuda.ColorConversion.GRAY2RGB,
            cvcuda.Format.Y8,
            cvcuda.Format.RGB8,
            1,
            3,
        ),
        "RGB2HSV": (
            cvcuda.ColorConversion.RGB2HSV,
            cvcuda.Format.RGB8,
            cvcuda.Format.HSV8,
            3,
            3,
        ),
        "HSV2RGB": (
            cvcuda.ColorConversion.HSV2RGB,
            cvcuda.Format.HSV8,
            cvcuda.Format.RGB8,
            3,
            3,
        ),
        "BGR2Lab": (
            cvcuda.ColorConversion.BGR2Lab,
            cvcuda.Format.BGR8,
            cvcuda.Format.LAB8,
            3,
            3,
        ),
        "RGB2Lab": (
            cvcuda.ColorConversion.RGB2Lab,
            cvcuda.Format.RGB8,
            cvcuda.Format.LAB8,
            3,
            3,
        ),
        "Lab2BGR": (
            cvcuda.ColorConversion.Lab2BGR,
            cvcuda.Format.LAB8,
            cvcuda.Format.BGR8,
            3,
            3,
        ),
        "Lab2RGB": (
            cvcuda.ColorConversion.Lab2RGB,
            cvcuda.Format.LAB8,
            cvcuda.Format.RGB8,
            3,
            3,
        ),
        "LBGR2Lab": (
            cvcuda.ColorConversion.LBGR2Lab,
            cvcuda.Format.BGR8,
            cvcuda.Format.LAB8,
            3,
            3,
        ),
        "LRGB2Lab": (
            cvcuda.ColorConversion.LRGB2Lab,
            cvcuda.Format.RGB8,
            cvcuda.Format.LAB8,
            3,
            3,
        ),
        "Lab2LBGR": (
            cvcuda.ColorConversion.Lab2LBGR,
            cvcuda.Format.LAB8,
            cvcuda.Format.BGR8,
            3,
            3,
        ),
        "Lab2LRGB": (
            cvcuda.ColorConversion.Lab2LRGB,
            cvcuda.Format.LAB8,
            cvcuda.Format.RGB8,
            3,
            3,
        ),
        "RGB2YUV": (
            cvcuda.ColorConversion.RGB2YUV,
            cvcuda.Format.RGB8,
            cvcuda.Format.YUV8p,
            3,
            3,
        ),
        "YUV2RGB": (
            cvcuda.ColorConversion.YUV2RGB,
            cvcuda.Format.YUV8p,
            cvcuda.Format.RGB8,
            3,
            3,
        ),
        "RGB2YUV_NV12": (
            cvcuda.ColorConversion.RGB2YUV_NV12,
            cvcuda.Format.RGB8,
            cvcuda.Format.NV12,
            3,
            1.5,
        ),
        "YUV2RGB_NV12": (
            cvcuda.ColorConversion.YUV2RGB_NV12,
            cvcuda.Format.NV12,
            cvcuda.Format.RGB8,
            1.5,
            3,
        ),
    }

    code, in_format, out_format, in_bpp, out_bpp = code_map[code_str]

    if (
        in_format == cvcuda.Format.NV12 or out_format == cvcuda.Format.NV12
    ) and dtype != cvcuda.Type.U8:
        state.skip("NV12 CvtColor benchmarks support only U8")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) CvtColor benchmark is tensor-only")
        return None
    if (is_planar or is_fake_planar) and (
        in_bpp != int(in_bpp) or out_bpp != int(out_bpp)
    ):
        state.skip("Skipping subsampled YUV CvtColor formats for planar benchmarks")
        return None

    dtype_size = get_dtype_size(dtype)
    src_bytes = int(N * H * W * in_bpp * dtype_size)
    dst_bytes = int(N * H * W * out_bpp * dtype_size)
    if is_fake_planar:
        state.add_global_memory_reads(2 * src_bytes + dst_bytes)
        state.add_global_memory_writes(src_bytes + 2 * dst_bytes)
    else:
        state.add_global_memory_reads(src_bytes)
        state.add_global_memory_writes(dst_bytes)

    if input_kind == "Tensor":  # Tensor mode
        if is_fake_planar:
            in_ch = int(in_bpp)
            out_ch = int(out_bpp)
            src = create_tensor(
                (N, in_ch, H, W),
                dtype,
                device_id,
                layout="NCHW",
                fill_mode="checkerboard",
            )
            inter_src = create_tensor(
                (N, H, W, in_ch),
                dtype,
                device_id,
                layout="NHWC",
                fill_mode=0,
            )
            inter_dst = create_tensor(
                (N, H, W, out_ch),
                dtype,
                device_id,
                layout="NHWC",
                fill_mode=0,
            )
            dst = create_tensor(
                (N, out_ch, H, W),
                dtype,
                device_id,
                layout="NCHW",
                fill_mode=0,
            )
        elif is_planar:
            in_ch = int(in_bpp)
            out_ch = int(out_bpp)
            src = create_tensor(
                (N, in_ch, H, W),
                dtype,
                device_id,
                layout="NCHW",
                fill_mode="checkerboard",
            )
            dst = create_tensor(
                (N, out_ch, H, W), dtype, device_id, layout="NCHW", fill_mode=0
            )
        elif in_format == cvcuda.Format.NV12:
            height420 = (H * 3) // 2
            src = create_tensor(
                (N, height420, W, 1),
                dtype,
                device_id,
                layout="NHWC",
                fill_mode="checkerboard",
            )
        else:
            in_ch = int(in_bpp)
            src = create_tensor(
                (N, H, W, in_ch),
                dtype,
                device_id,
                layout="NHWC",
                fill_mode="checkerboard",
            )

        if not (is_fake_planar or is_planar):
            if out_format == cvcuda.Format.NV12:
                height420 = (H * 3) // 2
                dst = create_tensor(
                    (N, height420, W, 1),
                    dtype,
                    device_id,
                    layout="NHWC",
                    fill_mode=0,
                )
            else:
                out_ch = int(out_bpp)
                dst = create_tensor(
                    (N, H, W, out_ch),
                    dtype,
                    device_id,
                    layout="NHWC",
                    fill_mode=0,
                )
    else:  # ImageBatchVarShape mode
        if (
            in_format == cvcuda.Format.NV12
            or out_format == cvcuda.Format.NV12
            or in_format == cvcuda.Format.YUV8p
            or out_format == cvcuda.Format.YUV8p
        ):
            state.skip(
                "Skipping formats that have subsampled planes for the varshape benchmark"
            )
            return None

        in_ch = int(in_bpp)
        out_ch = int(out_bpp)
        if dtype != cvcuda.Type.U8:
            try:
                in_format = _format_for_dtype(
                    in_format, dtype_str, in_ch, planar=is_planar
                )
                out_format = _format_for_dtype(
                    out_format, dtype_str, out_ch, planar=is_planar
                )
            except ValueError as exc:
                state.skip(str(exc))
                return None
        elif is_planar:
            try:
                in_format = _planar_format(in_format)
                out_format = _planar_format(out_format)
            except ValueError as exc:
                state.skip(str(exc))
                return None

        src = create_image_batch_varshape(
            (N, H, W, in_ch),
            0,
            in_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, H, W, out_ch),
            0,
            out_format,
            dtype,
            device_id,
            fill_mode=0,
        )

    get_stream = create_stream_cache()

    if is_fake_planar:

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.cvtcolor_into(inter_dst, inter_src, code, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    def run(launch):
        cvcuda.cvtcolor_into(dst, src, code, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("cvtcolor", cvtcolor)
