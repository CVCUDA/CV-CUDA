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

"""CV-CUDA Stack operator benchmark - Python equivalent of BenchStack.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_num_channels,
    get_format_from_dtype,
    get_input_kind,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def stack(state):
    """Stack operator benchmark matching C++ BenchStack.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    input_kind = get_input_kind(state.get_string("inputKind"))
    try:
        layout = state.get_string("layout")
    except KeyError:
        layout = "NHWC"
    device_id = state.get_device()

    N, H, W = shape
    ch = get_num_channels(dtype_str)

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("Stack benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None
    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind != "Tensor":
        state.skip("Fake-planar (NCHW_FAKE) Stack benchmark is TensorBatch-only")
        return None

    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    state.add_global_memory_reads((3 if is_fake_planar else 1) * bytes_)
    state.add_global_memory_writes((3 if is_fake_planar else 1) * bytes_)

    get_stream = create_stream_cache()

    if is_fake_planar:
        planar_src_tensors = []
        inter_src_tensors = []
        inter_src = cvcuda.TensorBatch(N)
        for _ in range(N):
            planar_src = create_tensor(
                (ch, H, W), dtype, device_id, layout="CHW", fill_mode="checkerboard"
            )
            interleaved_src = create_tensor(
                (H, W, ch), dtype, device_id, layout="HWC", fill_mode=0
            )
            planar_src_tensors.append(planar_src)
            inter_src_tensors.append(interleaved_src)
            inter_src.pushback(interleaved_src)

        inter_dst = create_tensor(
            (N, H, W, ch), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor((N, ch, H, W), dtype, device_id, layout="NCHW", fill_mode=0)

        def run_fake(launch):
            stream = get_stream(launch)
            for planar_src, interleaved_src in zip(
                planar_src_tensors, inter_src_tensors, strict=True
            ):
                cvcuda.reformat_into(interleaved_src, planar_src, stream=stream)
            cvcuda.stack_into(inter_dst, inter_src, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    dst_shape = (N, ch, H, W) if is_planar else (N, H, W, ch)
    dst = create_tensor(dst_shape, dtype, device_id, layout=layout, fill_mode=0)

    if input_kind == "Tensor":
        src = cvcuda.TensorBatch(N)
        src_layout = "CHW" if is_planar else "HWC"
        src_shape = (ch, H, W) if is_planar else (H, W, ch)
        for _ in range(N):
            t = create_tensor(
                src_shape, dtype, device_id, layout=src_layout, fill_mode="checkerboard"
            )
            src.pushback(t)
    else:
        img_format = get_format_from_dtype(dtype_str, ch, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, ch),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )

    def run(launch):
        stream = get_stream(launch)
        cvcuda.stack_into(dst, src, stream=stream)

    return run


if __name__ == "__main__":
    run_benchmark("stack", stack)
