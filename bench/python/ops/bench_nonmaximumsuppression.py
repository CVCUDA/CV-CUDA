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

"""CV-CUDA NonMaximumSuppression operator benchmark - Python equivalent of BenchNonMaximumSuppression.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp  # noqa: E402
import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    parse_shape,
    create_tensor,
    create_stream_cache,
    run_benchmark,
    _lcg_fill,
    _LCG_SEED,
)


def nms(state):
    """NonMaximumSuppression operator benchmark matching C++ BenchNonMaximumSuppression.cpp"""

    shape = parse_shape(state.get_string("shape"))
    device_id = state.get_device()
    score_threshold = state.get_float64("scoreThreshold")
    iou_threshold = state.get_float64("iouThreshold")

    N, num_boxes = shape

    state.add_global_memory_reads(N * num_boxes * (8 + 4))
    state.add_global_memory_writes(N * num_boxes * 1 * 2)

    get_stream = create_stream_cache()

    src_bb = cvcuda.Tensor((N, num_boxes), cvcuda.Type._4S16, "NB")
    with cp.cuda.Device(device_id):
        # Deterministic per-element [10, 50] cycle mirroring the C++
        # BenchNonMaximumSuppression lambda exactly (n=batch, b=box, c=channel
        # → 10 + (b*7 + c*11 + n*3) % 41).
        n_idx, b_idx, c_idx = cp.indices((N, num_boxes, 4), dtype=cp.int64)
        bb_data = (10 + (b_idx * 7 + c_idx * 11 + n_idx * 3) % 41).astype(cp.int16)
        cp.copyto(
            cp.asarray(src_bb.cuda()).view(cp.int16).reshape(N, num_boxes, 4), bb_data
        )

    with cp.cuda.Device(device_id):
        # srcSc: deterministic LCG over float [-1, +1] (bit-identical to C++
        # RandomValues<float>() which routes through the GPU LCG fast path).
        sc_data = cp.empty((N, num_boxes), dtype=cp.float32)
        _lcg_fill(sc_data, _LCG_SEED)
        src_sc = cvcuda.as_tensor(sc_data, "NB")

    dst_mk = create_tensor(
        (N, num_boxes), cvcuda.Type.U8, device_id, layout="NB", fill_mode=0
    )

    def run(launch):
        cvcuda.nms_into(
            dst_mk,
            src_bb,
            src_sc,
            score_threshold,
            iou_threshold,
            stream=get_stream(launch),
        )

    return run


if __name__ == "__main__":
    run_benchmark("nonmaximumsuppression", nms)
