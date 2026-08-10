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

"""CV-CUDA Label operator benchmark - Python equivalent of BenchLabel.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    parse_shape,
    get_dtype,
    get_dtype_size,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def label(state):
    """Label operator benchmark matching C++ BenchLabel.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype = get_dtype(state.get_string("InOutDataType"))
    runChoice = state.get_string("runChoice")
    device_id = state.get_device()
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("Label benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"

    if runChoice == "DEFAULT":
        runChoice = ""

    dtype_size = get_dtype_size(dtype)
    image_bytes = N * H * W * dtype_size
    out_bytes = N * H * W * 4
    if is_fake_planar:
        state.add_global_memory_reads(2 * image_bytes + out_bytes)
        state.add_global_memory_writes(image_bytes + 2 * out_bytes)
    else:
        state.add_global_memory_reads(image_bytes)
        state.add_global_memory_writes(out_bytes)

    tensor_shape = (N, 1, H, W) if is_planar or is_fake_planar else (N, H, W, 1)
    tensor_layout = "NCHW" if is_planar or is_fake_planar else "NHWC"

    src = create_tensor(
        tensor_shape, dtype, device_id, layout=tensor_layout, fill_mode="lcg"
    )

    dst = create_tensor(
        tensor_shape, cvcuda.Type.S32, device_id, layout=tensor_layout, fill_mode=0
    )

    bgLabel = None
    minThresh = None
    maxThresh = None
    minSize = None
    mask = None

    if "BG" in runChoice:
        bgLabel = create_tensor((N,), dtype, device_id, layout="N", fill_mode="lcg")
    if "MIN" in runChoice:
        minThresh = create_tensor((N,), dtype, device_id, layout="N", fill_mode=64)
    if "MAX" in runChoice:
        maxThresh = create_tensor((N,), dtype, device_id, layout="N", fill_mode=192)
    if "ISLAND" in runChoice:
        minSize = create_tensor(
            (N,), cvcuda.Type.S32, device_id, layout="N", fill_mode=16
        )
    if "MASK" in runChoice:
        mask = create_tensor(
            tensor_shape,
            cvcuda.Type.U8,
            device_id,
            layout=tensor_layout,
            fill_mode="checkerboard",
        )

    countTensor = None
    statsTensor = None

    if "COUNT" in runChoice:
        countTensor = create_tensor(
            (N,), cvcuda.Type.S32, device_id, layout="N", fill_mode=0
        )
    if "STAT" in runChoice:
        statsTensor = create_tensor(
            (N, 10000, 7), cvcuda.Type.S32, device_id, layout="NMA", fill_mode=0
        )

    conn = cvcuda.ConnectivityType.CONNECTIVITY_4_2D
    alab = cvcuda.LABEL.FAST
    mType = cvcuda.LabelMaskType.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY

    get_stream = create_stream_cache()

    if is_fake_planar:
        inter_src = create_tensor(
            (N, H, W, 1), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, 1), cvcuda.Type.S32, device_id, layout="NHWC", fill_mode=0
        )
        inter_mask = None
        if mask is not None:
            inter_mask = create_tensor(
                (N, H, W, 1), cvcuda.Type.U8, device_id, layout="NHWC", fill_mode=0
            )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            if mask is not None:
                cvcuda.reformat_into(inter_mask, mask, stream=stream)
            cvcuda.label_into(
                inter_dst,
                countTensor,
                statsTensor,
                inter_src,
                connectivity=conn,
                assign_labels=alab,
                mask_type=mType,
                bg_label=bgLabel,
                min_thresh=minThresh,
                max_thresh=maxThresh,
                min_size=minSize,
                mask=inter_mask,
                stream=stream,
            )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    def run(launch):
        cvcuda.label_into(
            dst,
            countTensor,
            statsTensor,
            src,
            connectivity=conn,
            assign_labels=alab,
            mask_type=mType,
            bg_label=bgLabel,
            min_thresh=minThresh,
            max_thresh=maxThresh,
            min_size=minSize,
            mask=mask,
            stream=get_stream(launch),
        )

    return run


if __name__ == "__main__":
    run_benchmark("label", label)
