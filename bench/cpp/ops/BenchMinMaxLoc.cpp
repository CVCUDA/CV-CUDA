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
#include "ops/generated/BenchMinMaxLocConfig.hpp"

#include <cvcuda/OpMinMaxLoc.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline nvcv::DataType GetValueDataType()
{
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
    {
        return nvcv::TYPE_S32;
    }
    else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>)
    {
        return nvcv::TYPE_U32;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return nvcv::TYPE_F32;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return nvcv::TYPE_F64;
    }
    else
    {
        throw std::invalid_argument("Unsupported MinMaxLoc input data type");
    }
}

template<typename T>
inline size_t GetValueDataSize()
{
    if constexpr (std::is_integral_v<T>)
    {
        return sizeof(int32_t);
    }
    else
    {
        return sizeof(T);
    }
}

template<typename T>
inline void minmaxloc(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    int                         maxLocs   = benchutils::GetIntParam<int>(state, "maxLocations");
    std::string                 runChoice = state.get_string("runChoice");
    std::string                 layout    = benchutils::GetStringParam(state, "layout", "NHWC");

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";

    if (layout != "NHWC" && !isPlanar && !isFakePlanar)
    {
        state.skip("MinMaxLoc benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (inputKind != benchutils::InputKind::Tensor && layout != "NHWC")
    {
        state.skip("Planar MinMaxLoc benchmark is tensor-only");
        return;
    }

    bool runMin = runChoice == "MIN" || runChoice == "MIN_MAX";
    bool runMax = runChoice == "MAX" || runChoice == "MIN_MAX";

    if (!runMin && !runMax)
    {
        throw std::invalid_argument("runChoice must be MIN, MAX, or MIN_MAX");
    }

    // clang-format off

    nvcv::DataType valueType = GetValueDataType<T>();

    nvcv::Tensor minVal({{shape.x}, "N"}, valueType);
    nvcv::Tensor minLoc({{shape.x, maxLocs}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{shape.x}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor maxVal({{shape.x}, "N"}, valueType);
    nvcv::Tensor maxLoc({{shape.x, maxLocs}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMax({{shape.x}, "N"}, nvcv::TYPE_S32);

    // clang-format on

    // R/W bandwidth rationale:
    // 1 read to find min/max + 1 read to collect their locations
    // 1 or 2 writes of min/max values, locations (2S32) and quantity (S32)
    const size_t inputBytes = static_cast<size_t>(shape.x) * shape.y * shape.z * sizeof(T);
    state.add_global_memory_reads(inputBytes * (isFakePlanar ? 3 : 2));
    state.add_global_memory_writes(shape.x * (GetValueDataSize<T>() + maxLocs * sizeof(int2) + sizeof(int))
                                       * (static_cast<int>(runMin) + static_cast<int>(runMax))
                                   + (isFakePlanar ? inputBytes : 0));

    cvcuda::MinMaxLoc op;

    const nvcv::Tensor noTensor{nullptr};

    auto runOp = [&op, &runChoice, &minVal, &minLoc, &numMin, &maxVal, &maxLoc, &numMax, &noTensor](cudaStream_t s,
                                                                                                    const auto  &src)
    {
        if (runChoice == "MIN")
        {
            op(s, src, minVal, minLoc, numMin, noTensor, noTensor, noTensor);
        }
        else if (runChoice == "MAX")
        {
            op(s, src, noTensor, noTensor, noTensor, maxVal, maxLoc, numMax);
        }
        else
        {
            op(s, src, minVal, minLoc, numMin, maxVal, maxLoc, numMax);
        }
    };

    // clang-format off

    if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = (isPlanar || isFakePlanar)
                             ? nvcv::Tensor({{shape.x, 1, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<T>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::LcgValues<T>());

        if (isFakePlanar)
        {
            nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
            cvcuda::Reformat reformatOp;

            benchutils::warmup_and_exec(state, BENCH_MINMAXLOC_WARMUP_ITERATIONS,
                [&runOp, &reformatOp, &src, &interSrc](cudaStream_t s) {
                    reformatOp(s, src, interSrc);
                    runOp(s, interSrc);
                });
            return;
        }

        benchutils::warmup_and_exec(state, BENCH_MINMAXLOC_WARMUP_ITERATIONS,
            [&runOp, &src](cudaStream_t s) {
                runOp(s, src);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                      benchutils::LcgValues<T>());

        benchutils::warmup_and_exec(state, BENCH_MINMAXLOC_WARMUP_ITERATIONS,
            [&runOp, &src](cudaStream_t s) {
                runOp(s, src);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(minmaxloc, NVBENCH_TYPE_AXES(BENCH_MINMAXLOC_TYPES))
BENCH_MINMAXLOC_AXES;
