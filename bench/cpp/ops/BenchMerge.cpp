/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "BenchChannelCopy.hpp"
#include "ops/generated/BenchMergeConfig.hpp"

template<typename Pixel>
inline void merge(nvbench::state &state, nvbench::type_list<Pixel>)
try
{
    channelbench::Run<Pixel, false>(state, BENCH_MERGE_WARMUP_ITERATIONS);
}
CVCUDA_BENCH_SKIP_ERRORS(state)

NVBENCH_BENCH_TYPES(merge, NVBENCH_TYPE_AXES(BENCH_MERGE_TYPES))
BENCH_MERGE_AXES;
