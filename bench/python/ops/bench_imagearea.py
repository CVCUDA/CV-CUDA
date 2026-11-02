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

"""CV-CUDA ImageArea operator benchmark - Python equivalent of BenchImageArea.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_num_channels,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def image_area(state):
    """ImageArea operator benchmark matching C++ BenchImageArea.cpp"""

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

    if layout not in ("NHWC", "NCHW"):
        state.skip("ImageArea benchmark supports only NHWC and NCHW layouts")
        return None

    # ImageArea is tensor-only: it produces a rank-1 [N] int32 tensor.
    if input_kind != "Tensor":
        state.skip("ImageArea benchmark is tensor-only")
        return None

    # Only the output is written; pixel values are never read by the operator.
    state.add_global_memory_writes(N * 4)

    is_planar = layout == "NCHW"
    shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
    src = create_tensor(shape, dtype, device_id, layout=layout, fill_mode=0)
    dst = cvcuda.Tensor((N,), cvcuda.Type.S32, "N")

    get_stream = create_stream_cache()

    def run(launch):
        cvcuda.image_area_into(dst, src, stream=get_stream(launch))

    return run


if __name__ == "__main__":
    run_benchmark("imagearea", image_area)
