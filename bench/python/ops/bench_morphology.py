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

"""CV-CUDA Morphology operator benchmark - Python equivalent of BenchMorphology.cpp"""

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
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def morphology(state):
    """Morphology operator benchmark matching C++ BenchMorphology.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    morph_type = state.get_string("morphType")
    border = get_border_type(state.get_string("border"))
    kernel_size_str = state.get_string("kernelSize")
    iteration = int(state.get_int64("iteration"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"
    device_id = state.get_device()

    N, H, W = shape
    dtype_size = get_dtype_size(dtype_str)

    kernel_w, kernel_h = map(int, kernel_size_str.lower().split("x"))
    mask_size = (kernel_w, kernel_h)

    morph_map = {
        "ERODE": cvcuda.MorphologyType.ERODE,
        "DILATE": cvcuda.MorphologyType.DILATE,
        "OPEN": cvcuda.MorphologyType.OPEN,
        "CLOSE": cvcuda.MorphologyType.CLOSE,
    }
    morph_op = morph_map.get(morph_type, cvcuda.MorphologyType.ERODE)

    needs_workspace = (
        morph_op in (cvcuda.MorphologyType.OPEN, cvcuda.MorphologyType.CLOSE)
        or iteration > 1
    )

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "Morphology benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) Morphology benchmark is tensor-only")
        return None

    if needs_workspace:
        bw_iteration = 2 * iteration
    else:
        bw_iteration = iteration

    bytes_ = N * H * W * dtype_size * bw_iteration
    if is_fake_planar:
        state.add_global_memory_reads(3 * bytes_)
        state.add_global_memory_writes(3 * bytes_)
    else:
        state.add_global_memory_reads(bytes_)
        state.add_global_memory_writes(bytes_)

    get_stream = create_stream_cache()

    anchor = (-1, -1)

    if input_kind == "Tensor":  # Tensor mode
        if is_fake_planar:
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

            workspace = None
            if needs_workspace:
                workspace = create_tensor(
                    (N, H, W, num_channels),
                    dtype,
                    device_id,
                    layout="NHWC",
                    fill_mode=0,
                )

            def run_fake(launch):
                stream = get_stream(launch)
                cvcuda.reformat_into(inter_src, src, stream=stream)
                cvcuda.morphology_into(
                    inter_dst,
                    inter_src,
                    morph_op,
                    mask_size,
                    anchor=anchor,
                    iteration=iteration,
                    border=border,
                    workspace=workspace,
                    stream=stream,
                )
                cvcuda.reformat_into(dst, inter_dst, stream=stream)

            return run_fake

        input_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        src = create_tensor(
            input_shape,
            dtype,
            device_id,
            layout="NCHW" if is_planar else "NHWC",
            fill_mode="checkerboard",
        )
        dst = create_tensor(
            input_shape,
            dtype,
            device_id,
            layout="NCHW" if is_planar else "NHWC",
            fill_mode=0,
        )

        workspace = None
        if needs_workspace:
            workspace = create_tensor(
                input_shape,
                dtype,
                device_id,
                layout="NCHW" if is_planar else "NHWC",
                fill_mode=0,
            )

        def run(launch):
            cvcuda.morphology_into(
                dst,
                src,
                morph_op,
                mask_size,
                anchor=anchor,
                iteration=iteration,
                border=border,
                workspace=workspace,
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

        workspace = None
        if needs_workspace:
            workspace = create_image_batch_varshape(
                (N, H, W, num_channels),
                0,
                img_format,
                dtype,
                device_id,
                fill_mode=0,
            )

        with cp.cuda.Device(device_id):
            mask_data = cp.zeros((N, 2), dtype=cp.int32)
            mask_data[:, 0] = kernel_w
            mask_data[:, 1] = kernel_h
            mask_tensor = cvcuda.as_tensor(mask_data.reshape(N, 2), "NW")

            anchor_data = cp.full((N, 2), -1, dtype=cp.int32)
            anchor_tensor = cvcuda.as_tensor(anchor_data.reshape(N, 2), "NW")

        def run(launch):
            cvcuda.morphology_into(
                dst,
                src,
                morph_op,
                mask_tensor,
                anchor_tensor,
                iteration=iteration,
                border=border,
                workspace=workspace,
                stream=get_stream(launch),
            )

        return run


if __name__ == "__main__":
    run_benchmark("morphology", morphology)
