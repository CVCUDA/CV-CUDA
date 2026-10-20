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

"""CV-CUDA HQResize operator benchmark - Python equivalent of BenchHQResize.cpp"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    parse_shape,
    get_positive_square_or_shape,
    get_dtype,
    get_dtype_size,
    get_resize_output_shape,
    get_format_from_dtype,
    get_interpolation_type,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def _vlm_center_crop_roi(input_h, input_w, output_h, output_w):
    """Return the resize-to-cover source ROI shared with the C++ benchmark."""
    resize_scale = max(output_h / input_h, output_w / input_w)
    scaled_h = input_h * resize_scale + 0.5
    scaled_w = input_w * resize_scale + 0.5
    if max(scaled_h, scaled_w) > (1 << 31) - 1:
        return None
    resize_h = math.floor(scaled_h)
    resize_w = math.floor(scaled_w)
    crop_y = (resize_h - output_h) // 2
    crop_x = (resize_w - output_w) // 2
    scale_y = np.float32(input_h) / np.float32(resize_h)
    scale_x = np.float32(input_w) / np.float32(resize_w)
    return (
        float(np.float32(crop_y) * scale_y),
        float(np.float32(crop_x) * scale_x),
        float(np.float32(crop_y + output_h) * scale_y),
        float(np.float32(crop_x + output_w) * scale_x),
    )


def hqresize(state):
    """HQResize operator benchmark matching C++ BenchHQResize.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    resize_type = state.get_string("resizeType")
    interpolation = state.get_string("interpolation")
    interp = get_interpolation_type(interpolation)
    antialias = bool(state.get_int64("antialias"))
    input_kind = state.get_string("inputKind")
    device_id = state.get_device()

    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("HQResize benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None
    if input_kind not in ("Tensor", "VarShape", "TensorBatch"):
        state.skip(
            "HQResize benchmark supports only Tensor, VarShape, and TensorBatch input kinds"
        )
        return None
    planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    # Channel count is an explicit axis (default 1): legacy single-channel NHWC profiles omit it, while
    # the planar-comparison profiles (NHWC/NCHW/NCHW_FAKE) set numChannels=3 to compare at parity.
    try:
        channels = int(state.get_int64("numChannels"))
    except Exception:
        channels = 1

    # NCHW_FAKE ("fake planar") is a tensor-only comparison path matching BenchHQResize.cpp.
    if is_fake_planar and input_kind != "Tensor":
        state.skip("Fake-planar (NCHW_FAKE) HQResize benchmark is tensor-only")
        return None

    roi = None
    effective_input_h = H
    effective_input_w = W
    dtype_size = get_dtype_size(dtype)
    source_bytes = N * H * W * channels * dtype_size
    vlm_prefix = "VLM_CENTER_CROP_"
    if resize_type.startswith(vlm_prefix):
        output = get_positive_square_or_shape(resize_type.removeprefix(vlm_prefix))
        if output is None:
            state.skip("VLM center-crop output must be a positive HxW or square value")
            return None
        if input_kind != "VarShape":
            state.skip("VLM center crop requires inputKind=VarShape")
            return None
        if layout != "NHWC":
            state.skip("VLM center crop requires layout=NHWC")
            return None
        if H <= 0:
            state.skip("VLM center crop requires a positive input height")
            return None
        if W < H:
            state.skip("VLM center crop requires landscape input (width >= height)")
            return None
        if dtype_str != "uint8":
            state.skip("VLM center crop requires InOutDataType=uint8")
            return None
        if channels != 3:
            state.skip("VLM center crop requires numChannels=3")
            return None
        if interpolation != "CUBIC":
            state.skip("VLM center crop requires interpolation=CUBIC")
            return None
        dst_h, dst_w = output

        sample_roi = _vlm_center_crop_roi(H, W, dst_h, dst_w)
        if sample_roi is None:
            state.skip(
                "VLM center-crop intermediate dimensions exceed the supported range"
            )
            return None
        roi = [sample_roi] * N
        effective_input_h = float(np.float32(sample_roi[2]) - np.float32(sample_roi[0]))
        effective_input_w = float(np.float32(sample_roi[3]) - np.float32(sample_roi[1]))
        read_h = min(H, max(0, math.ceil(sample_roi[2]))) - min(
            H, max(0, math.floor(sample_roi[0]))
        )
        read_w = min(W, max(0, math.ceil(sample_roi[3]))) - min(
            W, max(0, math.floor(sample_roi[1]))
        )
        source_bytes = N * read_h * read_w * channels * dtype_size
    else:
        try:
            _, dst_h, dst_w = get_resize_output_shape((N, H, W), resize_type)
        except ValueError as error:
            state.skip(str(error))
            return None
    if (
        dst_h >= effective_input_h
        and dst_w >= effective_input_w
        and (dst_h > effective_input_h or dst_w > effective_input_w)
        and antialias
    ):
        state.skip("Antialias is no-op for expanding")
        return None

    get_stream = create_stream_cache()

    if is_fake_planar:  # tensor-only: NCHW -> NHWC -> hqresize -> NHWC -> NCHW
        src_bytes = N * H * W * channels * dtype_size
        dst_bytes = N * dst_h * dst_w * channels * dtype_size
        state.add_global_memory_reads(2 * src_bytes + dst_bytes)
        state.add_global_memory_writes(src_bytes + 2 * dst_bytes)

        src = create_tensor(
            (N, channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        inter_src = create_tensor(
            (N, H, W, channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, dst_h, dst_w, channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, channels, dst_h, dst_w), dtype, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.hq_resize_into(
                inter_dst,
                inter_src,
                min_interpolation=interp,
                mag_interpolation=interp,
                antialias=antialias,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":
        state.add_global_memory_reads(source_bytes)
        state.add_global_memory_writes(N * dst_h * dst_w * channels * dtype_size)

        src_shape = (N, channels, H, W) if planar else (N, H, W, channels)
        dst_shape = (
            (N, channels, dst_h, dst_w) if planar else (N, dst_h, dst_w, channels)
        )
        src = create_tensor(
            src_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
        )
        dst = create_tensor(dst_shape, dtype, device_id, layout=layout, fill_mode=0)

        def run(launch):
            cvcuda.hq_resize_into(
                dst,
                src,
                min_interpolation=interp,
                mag_interpolation=interp,
                antialias=antialias,
                stream=get_stream(launch),
            )

        return run
    elif input_kind == "VarShape":
        if channels != 3:
            state.skip(
                "HQResize ImageBatchVarShape benchmark currently requires three channels"
            )
            return None

        state.add_global_memory_reads(source_bytes)
        state.add_global_memory_writes(N * dst_h * dst_w * channels * dtype_size)

        img_format = get_format_from_dtype(dtype_str, channels, planar=planar)
        src_batch = create_image_batch_varshape(
            (N, H, W, channels),
            0,
            img_format,
            dtype=dtype,
            device=device_id,
            fill_mode="checkerboard",
        )
        dst_batch = create_image_batch_varshape(
            (N, dst_h, dst_w, channels),
            0,
            img_format,
            dtype=dtype,
            device=device_id,
            fill_mode=0,
        )

        def run_varshape(launch):
            cvcuda.hq_resize_into(
                dst_batch,
                src_batch,
                min_interpolation=interp,
                mag_interpolation=interp,
                antialias=antialias,
                stream=get_stream(launch),
                roi=roi,
            )

        return run_varshape
    else:  # TensorBatch mode
        state.add_global_memory_reads(N * H * W * channels * dtype_size)
        state.add_global_memory_writes(N * dst_h * dst_w * channels * dtype_size)

        src_shape = (channels, H, W) if planar else (H, W, channels)
        dst_shape = (channels, dst_h, dst_w) if planar else (dst_h, dst_w, channels)
        batch_layout = "CHW" if planar else "HWC"
        src_batch = cvcuda.TensorBatch(N)
        dst_batch = cvcuda.TensorBatch(N)
        for _ in range(N):
            src_batch.pushback(
                create_tensor(
                    src_shape,
                    dtype,
                    device_id,
                    layout=batch_layout,
                    fill_mode="checkerboard",
                )
            )
            dst_batch.pushback(
                create_tensor(
                    dst_shape, dtype, device_id, layout=batch_layout, fill_mode=0
                )
            )

        def run(launch):
            cvcuda.hq_resize_into(
                dst_batch,
                src_batch,
                min_interpolation=interp,
                mag_interpolation=interp,
                antialias=antialias,
                stream=get_stream(launch),
            )

        return run


if __name__ == "__main__":
    run_benchmark("hqresize", hqresize)
