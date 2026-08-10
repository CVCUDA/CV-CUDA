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
#include "ops/generated/BenchPairwiseMatcherConfig.hpp"

#include <cvcuda/OpPairwiseMatcher.hpp>

#include <nvbench/nvbench.cuh>

template<typename ST>
inline void pairwisematcher(nvbench::state &state, nvbench::type_list<ST>)
try
{
    int3 shape = benchutils::GetShape<3, int3>(state.get_string("shape"));

    int matchesPerPoint = benchutils::GetIntParam<int>(state, "matchesPerPoint");

    bool crossCheck     = state.get_string("crossCheck") == "T";
    bool readNumSets    = state.get_string("readNumSets") == "T";
    bool writeDistances = state.get_string("writeDistances") == "T";

    NVCVNormType normType = benchutils::GetNormType(state.get_string("normType"));

    NVCVPairwiseMatcherType algoChoice;

    if (state.get_string("algoChoice") == "BRUTE_FORCE")
    {
        algoChoice = NVCV_BRUTE_FORCE;
    }
    else
    {
        throw std::invalid_argument("Unexpected algorithm choice = " + state.get_string("algoChoice"));
    }

    int maxMatches = shape.y * matchesPerPoint;

    cvcuda::PairwiseMatcher op(algoChoice);

    state.add_global_memory_reads((crossCheck ? 3 : 2) * shape.x * shape.y * shape.z * sizeof(ST));
    state.add_global_memory_writes(shape.x * (sizeof(int) + maxMatches * (2 * sizeof(int) + sizeof(float))));

    // clang-format off

    nvcv::Tensor set1({{shape.x, shape.y, shape.z}, "NMD"}, benchutils::GetDataType<ST>());
    nvcv::Tensor set2({{shape.x, shape.y, shape.z}, "NMD"}, benchutils::GetDataType<ST>());

    nvcv::Tensor matches({{shape.x, maxMatches, 2}, "NMD"}, nvcv::TYPE_S32);

    nvcv::Tensor numMatches({{shape.x}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor numSet1;
    nvcv::Tensor numSet2;
    nvcv::Tensor distances;

    if (readNumSets)
    {
        numSet1 = nvcv::Tensor({{shape.x}, "N"}, nvcv::TYPE_S32);
        numSet2 = nvcv::Tensor({{shape.x}, "N"}, nvcv::TYPE_S32);

        benchutils::FillTensor<int>(numSet1, [&shape](auto &){ return shape.y; });
        benchutils::FillTensor<int>(numSet2, [&shape](auto &){ return shape.y; });
    }
    if (writeDistances)
    {
        distances = nvcv::Tensor({{shape.x, maxMatches}, "NM"}, nvcv::TYPE_F32);
    }

    benchutils::FillTensor<ST>(set1, benchutils::LcgValues<ST>());
    benchutils::FillTensor<ST>(set2, benchutils::LcgValues<ST>());

    benchutils::warmup_and_exec(state, BENCH_PAIRWISEMATCHER_WARMUP_ITERATIONS,
        [&op, &set1, &set2, &numSet1, &numSet2, &matches, &numMatches, &distances, &crossCheck,
         &matchesPerPoint, &normType](cudaStream_t s) {
            op(s, set1, set2, numSet1, numSet2, matches, numMatches, distances, crossCheck,
               matchesPerPoint, normType);
        });
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(pairwisematcher, NVBENCH_TYPE_AXES(BENCH_PAIRWISEMATCHER_TYPES))
BENCH_PAIRWISEMATCHER_AXES;
