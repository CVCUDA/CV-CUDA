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
#include "ops/generated/BenchMinAreaRectConfig.hpp"

#include <cvcuda/OpMinAreaRect.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void minarearect(nvbench::state &state, nvbench::type_list<T>)
try
{
    int2                        shape     = benchutils::GetShape<2, int2>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    int                         numPoints = benchutils::GetIntParam<int>(state, "numPoints");

    if (numPoints < 0 || numPoints > shape.y)
    {
        throw std::invalid_argument("numPoints must be 0 or in [1, max_points]");
    }

    state.add_global_memory_reads(shape.x * shape.y * sizeof(T));
    state.add_global_memory_writes(shape.x * 8 * sizeof(float) + shape.x * sizeof(int));

    cvcuda::MinAreaRect op(shape.x);

    // clang-format off

    if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src({{shape.x, shape.y, 2}, "NWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, 8}, "NW"}, nvcv::TYPE_F32);
        nvcv::Tensor points({{1, shape.x}, "NW"}, nvcv::TYPE_S32);

        benchutils::FillTensor<T>(src, benchutils::LcgValues<T>());
        benchutils::FillTensor<float>(dst, [](const long4_16a &) { return 0.f; });  // Output only, zero-fill
        // numPoints=0 preserves the original deterministic [10, 100] cycle.
        benchutils::FillTensor<int>(points, [numPoints](const long4_16a &c) {
            return numPoints > 0 ? numPoints : 10 + static_cast<int>(c.y % 91);
        });

        benchutils::warmup_and_exec(state, BENCH_MINAREARECT_WARMUP_ITERATIONS,
            [&op, &src, &dst, &points, &shape](cudaStream_t s) { op(s, src, dst, points, shape.x); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(minarearect, NVBENCH_TYPE_AXES(BENCH_MINAREARECT_TYPES))
BENCH_MINAREARECT_AXES;
