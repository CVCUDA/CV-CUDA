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

"""CV-CUDA __OPNAMESPACE__ operator benchmark - Python equivalent of Bench__OPNAME__.cpp

TODO(make-op): this stub benchmarks the interleaved NHWC Tensor path only. Extend it to the
planar (NCHW) and fake-planar (NCHW_FAKE) layouts and the ImageBatchVarShape input kind so it
matches Bench__OPNAME__.cpp and the operator's declared support matrix. See bench_flip.py for a
complete reference.
"""

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


def __OPNAMELOW__(state):
    """__OPNAMESPACE__ operator benchmark matching C++ Bench__OPNAME__.cpp"""

    N, H, W = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"
    input_kind = get_input_kind(state.get_string("inputKind"))
    device_id = state.get_device()

    if input_kind != "Tensor":
        state.skip("TODO(make-op): implement the ImageBatchVarShape benchmark path")
        return None
    if layout != "NHWC":
        state.skip(
            "TODO(make-op): implement the planar (NCHW / NCHW_FAKE) benchmark path"
        )
        return None

    dtype_size = get_dtype_size(dtype_str)
    bytes_ = N * H * W * dtype_size
    state.add_global_memory_reads(bytes_)
    state.add_global_memory_writes(bytes_)

    get_stream = create_stream_cache()

    shape = (N, H, W, num_channels)
    src = create_tensor(
        shape, dtype, device_id, layout="NHWC", fill_mode="checkerboard"
    )
    dst = create_tensor(shape, dtype, device_id, layout="NHWC", fill_mode=0)

    def run(launch):
        cvcuda.__OPNAMELOW___into(dst, src, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("__OPNAMELOW__", __OPNAMELOW__)
