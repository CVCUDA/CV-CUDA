/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../CppBenchUtils.hpp"
#include "ops/generated/Bench__OPNAME__Config.hpp"

#include <cvcuda/Op__OPNAME__.hpp>

#include <nvbench/nvbench.cuh>

// Benchmark for the __OPNAMESPACE__ operator.
//
// TODO(make-op): this stub benchmarks the interleaved NHWC Tensor path only. Extend it to the
// planar (NCHW) and fake-planar (NCHW_FAKE) layouts and the ImageBatchVarShape input kind so the
// benchmark coverage matches the operator's declared support matrix. See .agents/guidance/MAKE_OP_GUIDELINES.md
// (BEN-* / COV-2) and BenchFlip.cpp for a complete reference (planar + fake-planar + var-shape).
template<typename T>
inline void __OPNAMELOW__(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    if (inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("TODO(make-op): implement the ImageBatchVarShape benchmark path");
        return;
    }
    if (layout != "NHWC")
    {
        state.skip("TODO(make-op): implement the planar (NCHW / NCHW_FAKE) benchmark path");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    state.add_global_memory_reads(bytes);
    state.add_global_memory_writes(bytes);

    cvcuda::__OPNAME__ op;

    nvcv::Tensor src(
        {
            {shape.x, shape.y, shape.z, ch},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor dst(
        {
            {shape.x, shape.y, shape.z, ch},
            "NHWC"
    },
        benchutils::GetDataType<BT>());

    benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

    benchutils::warmup_and_exec(state, BENCH___OPNAMEUPPER___WARMUP_ITERATIONS,
                                [&op, &src, &dst](cudaStream_t s) { op(s, src, dst); });
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(__OPNAMELOW__, NVBENCH_TYPE_AXES(BENCH___OPNAMEUPPER___TYPES))
BENCH___OPNAMEUPPER___AXES;
