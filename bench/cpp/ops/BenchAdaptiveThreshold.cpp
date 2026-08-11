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
#include "ops/generated/BenchAdaptiveThresholdConfig.hpp"

#include <cvcuda/OpAdaptiveThreshold.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void adaptivethreshold(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    int                         blockSize = benchutils::GetIntParam<int>(state, "blockSize");

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("AdaptiveThreshold benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if ((isPlanar || isFakePlanar) && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Planar AdaptiveThreshold benchmark rows are tensor-only");
        return;
    }

    NVCVThresholdType         threshType = NVCV_THRESH_BINARY;
    NVCVAdaptiveThresholdType adaptType  = NVCV_ADAPTIVE_THRESH_GAUSSIAN_C;

    double maxValue = 123.;
    double c        = -2.3;

    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    if (isFakePlanar)
    {
        state.add_global_memory_reads(3 * bytes);
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(bytes);
        state.add_global_memory_writes(bytes);
    }

    cvcuda::AdaptiveThreshold op(blockSize, shape.x);

    // clang-format off

    if (isFakePlanar)
    {
        nvcv::Tensor src     ({{shape.x, 1, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<T>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst     ({{shape.x, 1, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::CheckerboardValues<T>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_ADAPTIVETHRESHOLD_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &maxValue, &adaptType, &threshType, &blockSize, &c](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, maxValue, adaptType, threshType, blockSize, c);
                reformatOp(s, interDst, dst);
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{shape.x, 1, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<T>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, 1, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<T>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::CheckerboardValues<T>());

        benchutils::warmup_and_exec(state, BENCH_ADAPTIVETHRESHOLD_WARMUP_ITERATIONS,
            [&op, &src, &dst, &maxValue, &adaptType, &threshType, &blockSize, &c](cudaStream_t s) {
                op(s, src, dst, maxValue, adaptType, threshType, blockSize, c);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                      benchutils::CheckerboardValues<T>());
        // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
        benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });

        nvcv::Tensor maxValueTensor({{shape.x}, "N"}, nvcv::TYPE_F64);
        nvcv::Tensor blockSizeTensor({{shape.x}, "N"}, nvcv::TYPE_S32);
        nvcv::Tensor cTensor({{shape.x}, "N"}, nvcv::TYPE_F64);

        benchutils::FillTensor<double>(maxValueTensor, [&maxValue](const long4_16a &){ return maxValue; });
        benchutils::FillTensor<int>(blockSizeTensor, [&blockSize](const long4_16a &){ return blockSize; });
        benchutils::FillTensor<double>(cTensor, [&c](const long4_16a &){ return c; });

        benchutils::warmup_and_exec(state, BENCH_ADAPTIVETHRESHOLD_WARMUP_ITERATIONS,
            [&op, &src, &dst, &maxValueTensor, &adaptType, &threshType, &blockSizeTensor, &cTensor](cudaStream_t s) {
                op(s, src, dst, maxValueTensor, adaptType, threshType, blockSizeTensor, cTensor);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json
NVBENCH_BENCH_TYPES(adaptivethreshold, NVBENCH_TYPE_AXES(BENCH_ADAPTIVETHRESHOLD_TYPES))
BENCH_ADAPTIVETHRESHOLD_AXES;
