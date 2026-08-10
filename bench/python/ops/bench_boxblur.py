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
    get_num_channels,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def boxblur(state):
    """BoxBlur operator benchmark matching C++ BenchBoxBlur.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    num_boxes = state.get_int64("numBoxes")
    kernel_size = state.get_int64("kernelSize")
    box_width, box_height = parse_shape(state.get_string("boxSize"))
    box_pattern = state.get_string("boxPattern")
    dtype_str = state.get_string("InOutDataType")
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("BoxBlur benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"

    if input_kind == "VarShape":  # ImageBatchVarShape mode
        state.skip("ImageBatchVarShape not implemented for this benchmark")
        return None

    device_id = state.get_device()

    ch = get_num_channels(dtype_str)

    dtype_size = 1  # uint8
    sizeof_blurbox = 20  # Approximate size of NVCVBlurBoxI struct
    state.add_global_memory_reads(
        N * H * W * ch * dtype_size + N * num_boxes * sizeof_blurbox
    )
    state.add_global_memory_writes(N * H * W * ch * dtype_size)

    box_width = max(3, min(box_width, W))
    box_height = max(3, min(box_height, H))

    def make_blur_box(box_idx):
        x = 43
        y = 21
        if box_pattern == "grid":
            grid_cols = 1
            while grid_cols * grid_cols < num_boxes:
                grid_cols += 1
            grid_rows = (num_boxes + grid_cols - 1) // grid_cols
            col = box_idx % grid_cols
            row = box_idx // grid_cols
            max_x = max(0, W - box_width)
            max_y = max(0, H - box_height)
            x = max_x // 2 if grid_cols <= 1 else col * max_x // (grid_cols - 1)
            y = max_y // 2 if grid_rows <= 1 else row * max_y // (grid_rows - 1)
        elif box_pattern != "fixed":
            raise ValueError(f"Unexpected boxPattern = {box_pattern}")

        return cvcuda.BlurBoxI(
            box=(x, y, box_width, box_height), kernelSize=kernel_size
        )

    boxes_batch = [[make_blur_box(i) for i in range(num_boxes)] for _ in range(N)]
    blur_boxes = cvcuda.BlurBoxesI(boxes_batch)

    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, ch, H, W),
            "uint8",
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        inter_src = create_tensor(
            (N, H, W, ch), "uint8", device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, ch), "uint8", device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, ch, H, W), "uint8", device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.boxblur_into(inter_dst, inter_src, blur_boxes, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    tensor_shape = (N, ch, H, W) if is_planar else (N, H, W, ch)
    src = create_tensor(
        tensor_shape, "uint8", device_id, layout=layout, fill_mode="checkerboard"
    )
    dst = create_tensor(tensor_shape, "uint8", device_id, layout=layout, fill_mode=0)

    def run(launch):
        cvcuda.boxblur_into(dst, src, blur_boxes, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("boxblur", boxblur)
