/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ops/generated/BenchReformatConfig.hpp"

#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void reformat(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape        = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind    = benchutils::GetInputKind(state.get_string("inputKind"));
    const int                   rowAlignment = benchutils::GetIntParam<int>(state, "rowAlignment");

    if (rowAlignment < 0)
    {
        throw std::invalid_argument("rowAlignment must be non-negative");
    }

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Reformat op;

    // clang-format off

    if (inputKind == benchutils::InputKind::Tensor)
    {
        using BT = typename nvcv::cuda::BaseType<T>;
        int  ch  = nvcv::cuda::NumElements<T>;

        nvcv::MemAlignment bufferAlignment;
        bufferAlignment.rowAddr(rowAlignment);

        nvcv::Tensor src({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>(), bufferAlignment);
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>(), bufferAlignment);

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_REFORMAT_WARMUP_ITERATIONS,
            [&op, &src, &dst](cudaStream_t s) { op(s, src, dst); });
    }
    else
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(reformat, NVBENCH_TYPE_AXES(BENCH_REFORMAT_TYPES))
BENCH_REFORMAT_AXES;
