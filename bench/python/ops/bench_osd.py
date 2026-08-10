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


def _make_element(element_type, height, width):
    if element_type == "POINT":
        return cvcuda.Point(
            centerPos=(width // 2, height // 2),
            radius=min(width, height) // 2,
            color=(0, 0, 0, 255),
        )

    if element_type == "RECT":
        return cvcuda.BndBoxI(
            box=(width // 4, height // 4, width // 2, height // 2),
            thickness=3,
            borderColor=(255, 255, 0, 255),
            fillColor=(0, 128, 255, 0),
        )

    if element_type == "LINE":
        return cvcuda.Line(
            pos0=(width // 8, height // 8),
            pos1=(7 * width // 8, 7 * height // 8),
            thickness=4,
            color=(255, 0, 0, 255),
        )

    if element_type == "CIRCLE":
        return cvcuda.Circle(
            centerPos=(width // 2, height // 2),
            radius=min(width, height) // 4,
            thickness=4,
            borderColor=(0, 255, 255, 255),
            bgColor=(255, 0, 255, 0),
        )

    raise ValueError(f"Unsupported OSD elementType: {element_type}")


def osd(state):
    """OSD operator benchmark matching C++ BenchOSD.cpp"""

    shape = parse_shape(state.get_string("shape"))
    num_elem = state.get_int64("numElem")
    device_id = state.get_device()
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    input_kind = get_input_kind(state.get_string("inputKind"))
    element_type = state.get_string("elementType")
    try:
        layout = state.get_string("layout")
    except (KeyError, RuntimeError):
        layout = "NHWC"

    N, H, W = shape

    if dtype_str not in ("uchar3", "uchar4"):
        state.skip(f"Unsupported dtype for OSD: {dtype_str}")
        return None
    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("OSD benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if (is_planar or is_fake_planar) and input_kind != "Tensor":
        state.skip("Planar OSD benchmark is tensor-only")
        return None

    bytes_ = N * H * W * get_dtype_size(dtype_str)
    elem_bytes = num_elem * 4 * 16
    if is_planar or is_fake_planar:
        state.add_global_memory_reads(3 * bytes_ + elem_bytes)
        state.add_global_memory_writes(3 * bytes_)
    else:
        state.add_global_memory_reads(bytes_ + elem_bytes)
        state.add_global_memory_writes(bytes_)

    elements_per_batch = []
    for _ in range(N):
        batch_elements = []
        for _ in range(num_elem):
            batch_elements.append(_make_element(element_type, H, W))
        elements_per_batch.append(batch_elements)

    elements = cvcuda.Elements(elements=elements_per_batch)

    get_stream = create_stream_cache()

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

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            cvcuda.osd_into(inter_dst, inter_src, elements, stream=stream)
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    tensor_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
    src = create_tensor(
        tensor_shape, dtype, device_id, layout=layout, fill_mode="checkerboard"
    )
    dst = create_tensor(tensor_shape, dtype, device_id, layout=layout, fill_mode=0)

    def run(launch):
        stream = get_stream(launch)
        cvcuda.osd_into(dst, src, elements, stream=stream)

    return run


if __name__ == "__main__":
    run_benchmark("osd", osd)
