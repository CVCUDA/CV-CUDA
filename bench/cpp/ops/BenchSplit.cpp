/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "BenchChannelCopy.hpp"
#include "ops/generated/BenchSplitConfig.hpp"

template<typename Pixel>
inline void split(nvbench::state &state, nvbench::type_list<Pixel>)
try
{
    channelbench::Run<Pixel, true>(state, BENCH_SPLIT_WARMUP_ITERATIONS);
}
CVCUDA_BENCH_SKIP_ERRORS(state)

NVBENCH_BENCH_TYPES(split, NVBENCH_TYPE_AXES(BENCH_SPLIT_TYPES))
BENCH_SPLIT_AXES;
