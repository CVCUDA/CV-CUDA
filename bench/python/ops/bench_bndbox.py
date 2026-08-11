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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_num_channels,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def bndbox(state):
    """BndBox operator benchmark matching C++ BenchBndBox.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    num_boxes = state.get_int64("numBoxes")
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"

    if input_kind == "VarShape":  # ImageBatchVarShape mode
        state.skip("ImageBatchVarShape not implemented for this benchmark")
        return None

    device_id = state.get_device()

    ch = get_num_channels(dtype_str)

    if dtype_str not in ("uchar3", "uchar4"):
        state.skip(f"Unsupported dtype for BndBox: {dtype_str}")
        return None
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("BndBox benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if (is_planar or is_fake_planar) and input_kind != "Tensor":
        state.skip("Planar BndBox benchmark is tensor-only")
        return None

    image_bytes = N * H * W * get_dtype_size(dtype_str)
    sizeof_bndbox = 28  # Matches NVCVBndBoxI: NVCVBoxI + thickness + border/fill colors
    box_bytes = N * num_boxes * sizeof_bndbox
    if is_planar or is_fake_planar:
        state.add_global_memory_reads(3 * image_bytes + box_bytes)
        state.add_global_memory_writes(3 * image_bytes)
    else:
        state.add_global_memory_reads(image_bytes + box_bytes)
        state.add_global_memory_writes(image_bytes)

    box = cvcuda.BndBoxI(
        box=(43, 21, 12, 34),
        thickness=2,
        borderColor=(0, 0, 0, 255),
        fillColor=(0, 0, 0, 0),
    )

    boxes_batch = [[box] * num_boxes for _ in range(N)]
    bboxes = cvcuda.BndBoxesI(boxes_batch)

    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, ch, H, W), dtype, device_id, layout="NCHW", fill_mode="checkerboard"
        )
        inter_src = create_tensor(
            (N, H, W, ch), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, ch), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor((N, ch, H, W), dtype, device_id, layout="NCHW", fill_mode=0)

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.bndbox_into(inter_dst, inter_src, bboxes, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    tensor_shape = (N, ch, H, W) if is_planar else (N, H, W, ch)
    src = create_tensor(
        tensor_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
    )
    dst = create_tensor(tensor_shape, dtype, device_id, layout=layout, fill_mode=0)

    def run(launch):
        cvcuda.bndbox_into(dst, src, bboxes, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("bndbox", bndbox)
