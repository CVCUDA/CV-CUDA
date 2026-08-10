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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp  # noqa: E402
import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_dtype,
    get_dtype_size,
    create_tensor,
    create_stream_cache,
    run_benchmark,
)


def pairwisematcher(state):
    """PairwiseMatcher operator benchmark matching C++ BenchPairwiseMatcher.cpp"""

    shape_str = state.get_string("shape")
    matches_per_point = state.get_int64("matchesPerPoint")
    cross_check = state.get_string("crossCheck") == "T"
    read_num_sets = state.get_string("readNumSets") == "T"
    write_distances = state.get_string("writeDistances") == "T"
    norm_type_str = state.get_string("normType")
    dtype = get_dtype(state.get_string("InOutDataType"))
    device_id = state.get_device()

    parts = shape_str.split("x")
    N = int(parts[0])
    M = int(parts[1])
    D = int(parts[2])
    shape = (N, M, D)

    norm_map = {
        "HAMMING": cvcuda.Norm.HAMMING,
        "L1": cvcuda.Norm.L1,
        "L2": cvcuda.Norm.L2,
    }
    norm_type = norm_map.get(norm_type_str, cvcuda.Norm.L2)

    dtype_size = get_dtype_size(dtype)
    max_matches = M * matches_per_point
    state.add_global_memory_reads((3 if cross_check else 2) * N * M * D * dtype_size)
    state.add_global_memory_writes(N * (4 + max_matches * (2 * 4 + 4)))

    set1 = create_tensor(shape, dtype, device_id, layout="NMD", fill_mode="lcg")
    set2 = create_tensor(shape, dtype, device_id, layout="NMD", fill_mode="lcg")

    num_set1 = None
    num_set2 = None
    if read_num_sets:
        num_set1_data = cp.full((N,), M, dtype=cp.int32)
        num_set1 = cvcuda.as_tensor(num_set1_data, "N")
        num_set2_data = cp.full((N,), M, dtype=cp.int32)
        num_set2 = cvcuda.as_tensor(num_set2_data, "N")

    get_stream = create_stream_cache()

    def run(launch):
        stream = get_stream(launch)
        cvcuda.match(
            set1,
            set2,
            num_set1=num_set1,
            num_set2=num_set2,
            num_matches=True,
            distances=write_distances,
            cross_check=cross_check,
            matches_per_point=matches_per_point,
            norm_type=norm_type,
            algo_choice=cvcuda.Matcher.BRUTE_FORCE,
            stream=stream,
        )

    return run


if __name__ == "__main__":
    run_benchmark("pairwisematcher", pairwisematcher)
