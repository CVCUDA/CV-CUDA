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

import cupy as cp  # noqa: E402
import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_num_channels,
    get_dtype_size,
    get_format_from_dtype,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def channelreorder(state):
    """ChannelReorder operator benchmark matching C++ BenchChannelReorder.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    input_kind = get_input_kind(state.get_string("inputKind"))
    layout = state.get_string("layout")
    order_pattern = state.get_string("orderPattern")

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip(
            "ChannelReorder benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts"
        )
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) ChannelReorder benchmark is tensor-only")
        return None

    device_id = state.get_device()

    num_channels = get_num_channels(dtype_str)

    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    state.add_global_memory_reads(3 * bytes_ if is_fake_planar else bytes_)
    state.add_global_memory_writes(3 * bytes_ if is_fake_planar else bytes_)

    if order_pattern == "rotate":
        host_order = [(c + 1) % num_channels for c in range(num_channels)]
    elif order_pattern == "zero_fill":
        host_order = [-1 if c == 1 else c for c in range(num_channels)]
    else:
        state.skip(f"Invalid orderPattern: {order_pattern}")
        return None

    get_stream = create_stream_cache()
    if input_kind == "Tensor":
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
                cvcuda.channelreorder_into(
                    inter_dst, inter_src, host_order, stream=stream
                )
                cvcuda.reformat_into(dst, inter_dst, stream=stream)

            return run_fake

        tensor_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        src = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode="checkerboard",
        )
        dst = create_tensor(tensor_shape, dtype, device_id, layout=layout, fill_mode=0)

        def run_tensor(launch):
            cvcuda.channelreorder_into(dst, src, host_order, stream=get_stream(launch))

        return run_tensor

    # Deterministic patterns mirroring the C++ bench exactly.
    with cp.cuda.Device(device_id):
        n_idx, c_idx = cp.indices((N, 4), dtype=cp.int32)
        if order_pattern == "rotate":
            order_data = (n_idx + c_idx) % num_channels
        elif order_pattern == "zero_fill":
            order_data = cp.where(c_idx == 1, -1, c_idx % num_channels).astype(cp.int32)
        else:
            state.skip(f"Invalid orderPattern: {order_pattern}")
            return None
        order = cvcuda.as_tensor(order_data, "NC")
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

    def run(launch):
        cvcuda.channelreorder_into(dst, src, order, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("channelreorder", channelreorder)
