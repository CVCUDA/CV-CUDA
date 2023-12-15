/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "BenchUtils.hpp"

#include <cvcuda/OpPairwiseMatcher.hpp>

#include <nvbench/nvbench.cuh>

template<typename ST>
inline void PairwiseMatcher(nvbench::state &state, nvbench::type_list<ST>)
try
{
    long3 shape = benchutils::GetShape<3>(state.get_string("shape"));

    int matchesPerPoint = static_cast<int>(state.get_int64("matchesPerPoint"));

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

    nvcv::Tensor numSet1, numSet2, distances;

    if (readNumSets)
    {
        numSet1 = nvcv::Tensor({{shape.x}, "N"}, nvcv::TYPE_S32);
        numSet2 = nvcv::Tensor({{shape.x}, "N"}, nvcv::TYPE_S32);

        benchutils::FillTensor<int>(numSet1, [&shape](const long4 &){ return shape.y; });
        benchutils::FillTensor<int>(numSet2, [&shape](const long4 &){ return shape.y; });
    }
    if (writeDistances)
    {
        distances = nvcv::Tensor({{shape.x, maxMatches}, "NM"}, nvcv::TYPE_F32);
    }

    benchutils::FillTensor<ST>(set1, benchutils::RandomValues<ST>());
    benchutils::FillTensor<ST>(set2, benchutils::RandomValues<ST>());

    state.exec(nvbench::exec_tag::sync,
               [&op, &set1, &set2, &numSet1, &numSet2, &matches, &numMatches, &distances, &crossCheck,
                &matchesPerPoint, &normType](nvbench::launch &launch)
               {
                   op(launch.get_stream(), set1, set2, numSet1, numSet2, matches, numMatches, distances, crossCheck,
                      matchesPerPoint, normType);
               });
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using PairwiseMatcherTypes = nvbench::type_list<uint8_t, uint32_t, float>;

NVBENCH_BENCH_TYPES(PairwiseMatcher, NVBENCH_TYPE_AXES(PairwiseMatcherTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x10000x32"})
    .add_int64_axis("matchesPerPoint", {1})
    .add_string_axis("crossCheck", {"T"})
    .add_string_axis("readNumSets", {"F"})
    .add_string_axis("writeDistances", {"T"})
    .add_string_axis("normType", {"HAMMING"})
    .add_string_axis("algoChoice", {"BRUTE_FORCE"});
