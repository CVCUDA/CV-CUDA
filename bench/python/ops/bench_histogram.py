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

"""CV-CUDA Histogram operator benchmark - Python equivalent of BenchHistogram.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype_size,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def histogram(state):
    """Histogram operator benchmark matching C++ BenchHistogram.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    mask_mode = state.get_string("maskMode")
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"

    if input_kind == "VarShape":  # ImageBatchVarShape mode
        state.skip("ImageBatchVarShape not implemented for this benchmark")
        return None

    if mask_mode not in ("none", "checkerboard"):
        raise ValueError(f"Unsupported maskMode: {mask_mode}")
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "Histogram benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if (is_planar or is_fake_planar) and input_kind != "Tensor":
        state.skip("Planar Histogram benchmark is tensor-only")
        return None

    device_id = state.get_device()

    num_bins = 256

    image_bytes = N * H * W * get_dtype_size(dtype_str)
    mask_bytes = N * H * W
    hist_bytes = N * num_bins * 4
    use_mask = mask_mode == "checkerboard"
    if is_planar or is_fake_planar:
        state.add_global_memory_reads(
            2 * image_bytes + (2 * mask_bytes if use_mask else 0)
        )
        state.add_global_memory_writes(
            image_bytes + (mask_bytes if use_mask else 0) + hist_bytes
        )
    else:
        state.add_global_memory_reads(image_bytes + (mask_bytes if use_mask else 0))
        state.add_global_memory_writes(hist_bytes)

    hist = create_tensor(
        (N, num_bins, 1), cvcuda.Type.S32, device_id, layout="HWC", fill_mode=0
    )
    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, 1, H, W), dtype_str, device_id, layout="NCHW", fill_mode="checkerboard"
        )
        inter_src = create_tensor(
            (N, H, W, 1), dtype_str, device_id, layout="NHWC", fill_mode=0
        )
        mask = None
        inter_mask = None
        if use_mask:
            mask = create_tensor(
                (N, 1, H, W),
                cvcuda.Type.U8,
                device_id,
                layout="NCHW",
                fill_mode="checkerboard",
            )
            inter_mask = create_tensor(
                (N, H, W, 1), cvcuda.Type.U8, device_id, layout="NHWC", fill_mode=0
            )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            if mask is None:
                cvcuda.histogram_into(histogram=hist, src=inter_src, stream=stream)
            else:
                cvcuda.reformat_into(inter_mask, mask, stream=stream)
                cvcuda.histogram_into(
                    histogram=hist, src=inter_src, mask=inter_mask, stream=stream
                )

        return run_fake

    tensor_shape = (N, 1, H, W) if is_planar else (N, H, W, 1)
    src = create_tensor(
        tensor_shape, dtype_str, device_id, layout=layout, fill_mode="checkerboard"
    )
    mask = None
    if use_mask:
        mask = create_tensor(
            tensor_shape,
            cvcuda.Type.U8,
            device_id,
            layout=layout,
            fill_mode="checkerboard",
        )

    def run(launch):
        if mask is None:
            cvcuda.histogram_into(
                histogram=hist,
                src=src,
                stream=get_stream(launch),
            )
        else:
            cvcuda.histogram_into(
                histogram=hist,
                src=src,
                mask=mask,
                stream=get_stream(launch),
            )

    return run


if __name__ == "__main__":
    run_benchmark("histogram", histogram)
