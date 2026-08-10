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
#include "ops/generated/BenchConvertToConfig.hpp"

#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

inline nvcv::DataType GetConvertToDataType(const std::string &dtype)
{
    if (dtype == "uint8")
    {
        return nvcv::TYPE_U8;
    }
    else if (dtype == "uint16")
    {
        return nvcv::TYPE_U16;
    }
    else if (dtype == "int16")
    {
        return nvcv::TYPE_S16;
    }
    else if (dtype == "int32")
    {
        return nvcv::TYPE_S32;
    }
    else if (dtype == "float32")
    {
        return nvcv::TYPE_F32;
    }
    else if (dtype == "float64")
    {
        return nvcv::TYPE_F64;
    }

    throw std::invalid_argument("Invalid outDataType = " + dtype);
}

inline long GetConvertToTypeSize(const std::string &dtype)
{
    if (dtype == "uint8")
    {
        return 1;
    }
    else if (dtype == "uint16" || dtype == "int16")
    {
        return 2;
    }
    else if (dtype == "int32" || dtype == "float32")
    {
        return 4;
    }
    else if (dtype == "float64")
    {
        return 8;
    }

    throw std::invalid_argument("Invalid outDataType = " + dtype);
}

inline std::pair<double, double> GetConvertToScale(const std::string &scaleMode)
{
    if (scaleMode == "affine")
    {
        return {0.123, 0.456};
    }
    else if (scaleMode == "identity")
    {
        return {1.0, 0.0};
    }

    throw std::invalid_argument("Invalid scaleMode = " + scaleMode);
}

template<typename T>
inline void convertto(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape       = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        outDataType = state.get_string("outDataType");
    auto                        scaleMode   = state.get_string("scaleMode");
    auto                        layout      = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind   = benchutils::GetInputKind(state.get_string("inputKind"));

    auto [alpha, beta] = GetConvertToScale(scaleMode);

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("ConvertTo benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    // ConvertTo is tensor-only; the native planar (NCHW) and fake-planar (NCHW_FAKE) paths are too.
    if ((isPlanar || isFakePlanar) && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Planar ConvertTo benchmark is tensor-only");
        return;
    }

    const long srcBytes = shape.x * shape.y * shape.z * sizeof(T);
    const long dstBytes = static_cast<long>(shape.x) * shape.y * shape.z * ch * GetConvertToTypeSize(outDataType);
    if (isFakePlanar)
    {
        // reformat(NCHW→NHWC) + convert + reformat(NHWC→NCHW): reads src twice + dst once,
        // writes the interleaved src once + dst twice.
        state.add_global_memory_reads(2 * srcBytes + dstBytes);
        state.add_global_memory_writes(srcBytes + 2 * dstBytes);
    }
    else
    {
        state.add_global_memory_reads(srcBytes);
        state.add_global_memory_writes(dstBytes);
    }

    cvcuda::ConvertTo op;

    // clang-format off

    if (isFakePlanar) // tensor-only: planar→interleaved→convert→interleaved→planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, GetConvertToDataType(outDataType));
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, GetConvertToDataType(outDataType));

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_CONVERTTO_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &alpha, &beta](cudaStream_t s) {
                reformatOp(s, src, interSrc);       // NCHW → NHWC
                op(s, interSrc, interDst, alpha, beta); // interleaved convert
                reformatOp(s, interDst, dst);       // NHWC → NCHW
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, GetConvertToDataType(outDataType))
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, GetConvertToDataType(outDataType));

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_CONVERTTO_WARMUP_ITERATIONS,
            [&op, &src, &dst, &alpha, &beta](cudaStream_t s) { op(s, src, dst, alpha, beta); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(convertto, NVBENCH_TYPE_AXES(BENCH_CONVERTTO_TYPES))
BENCH_CONVERTTO_AXES;
