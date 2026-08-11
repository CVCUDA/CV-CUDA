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

"""CV-CUDA SIFT operator benchmark - Python equivalent of BenchSIFT.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def sift(state):
    """SIFT operator benchmark matching C++ BenchSIFT.cpp"""

    shape = parse_shape(state.get_string("shape"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    layout = state.get_string("layout")
    expand_input = state.get_string("expandInput") == "Y"
    capacity = state.get_int64("maxCapacity")
    num_octave_layers = state.get_int64("numOctaveLayers")
    contrast_threshold = state.get_float64("contrastThreshold")
    edge_threshold = state.get_float64("edgeThreshold")
    init_sigma = state.get_float64("initSigma")
    device_id = state.get_device()

    N, H, W = shape
    if layout not in ("NHWC", "NCHW"):
        raise ValueError(f"Invalid layout = {layout}")

    input_shape = (N, 1, H, W) if layout == "NCHW" else (N, H, W, 1)

    flags = (
        cvcuda.SIFT.USE_EXPANDED_INPUT
        if expand_input
        else cvcuda.SIFT.USE_ORIGINAL_INPUT
    )

    if expand_input:
        max_w, max_h = W * 2, H * 2
    else:
        max_w, max_h = W, H

    pyr_size = (num_octave_layers + 3) * N * (max_w * max_h * 2) * 4

    dtype_size = 1
    state.add_global_memory_reads(N * H * W * dtype_size + 2 * pyr_size)
    state.add_global_memory_writes(
        2 * pyr_size + N * 4 + N * capacity * (16 + 12 + 128 * dtype_size)
    )

    if input_kind == "VarShape":
        state.skip("ImageBatchVarShape not implemented for this operator")
        return None

    src = create_tensor(input_shape, "uint8", device_id, layout=layout, fill_mode="lcg")

    feat_coords = cvcuda.Tensor((N, capacity, 4), cvcuda.Type.F32, rowalign=1)
    feat_metadata = cvcuda.Tensor((N, capacity, 3), cvcuda.Type.F32, rowalign=1)
    feat_descriptors = cvcuda.Tensor((N, capacity, 128), cvcuda.Type.U8, rowalign=1)
    num_features = cvcuda.Tensor((N, 1), cvcuda.Type.S32, rowalign=1)

    get_stream = create_stream_cache()

    def run(launch):
        stream = get_stream(launch)
        cvcuda.sift_into(
            feat_coords,
            feat_metadata,
            feat_descriptors,
            num_features,
            src,
            num_octave_layers=num_octave_layers,
            contrast_threshold=contrast_threshold,
            edge_threshold=edge_threshold,
            init_sigma=init_sigma,
            flags=flags,
            stream=stream,
        )

    return run


if __name__ == "__main__":
    run_benchmark("sift", sift)
