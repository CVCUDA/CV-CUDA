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
#include "ops/generated/BenchJointBilateralFilterConfig.hpp"

#include <cvcuda/OpJointBilateralFilter.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void jointbilateralfilter(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape      = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind  = benchutils::GetInputKind(state.get_string("inputKind"));
    auto                        layout     = benchutils::GetStringParam(state, "layout", "NHWC");
    int                         diameter   = benchutils::GetIntParam<int>(state, "diameter");
    auto                        sigmaSpace = static_cast<float>(state.get_float64("sigmaSpace"));
    float                       sigmaColor = -1.f;

    using BT = typename nvcv::cuda::BaseType<T>;
    int ch   = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("JointBilateralFilter benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) JointBilateralFilter benchmark is tensor-only");
        return;
    }

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::JointBilateralFilter op;

    // clang-format off

    if (isFakePlanar) // tensor-only: planar->interleaved->jointbilateralfilter->interleaved->planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());

        cvcuda::Reformat reformatOp;
        benchutils::warmup_and_exec(state, BENCH_JOINTBILATERALFILTER_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &diameter, &sigmaColor, &sigmaSpace, &borderType](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interSrc, interDst, diameter, sigmaColor, sigmaSpace, borderType);
                reformatOp(s, interDst, dst);
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor color = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());
        benchutils::FillTensor<BT>(color, benchutils::LcgValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_JOINTBILATERALFILTER_WARMUP_ITERATIONS,
            [&op, &src, &color, &dst, &diameter, &sigmaColor, &sigmaSpace, &borderType](cudaStream_t s) {
                op(s, src, color, dst, diameter, sigmaColor, sigmaSpace, borderType);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape color(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
            color.pushBack(src.begin(), src.end());
            benchutils::FillPlanarImageBatchLike<T>(dst, src);
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                          benchutils::LcgValues<T>());
            // color can alias src (guidance = input is valid use case)
            color.pushBack(src.begin(), src.end());
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        nvcv::Tensor diameterTensor({{shape.x}, "N"}, nvcv::TYPE_S32);
        nvcv::Tensor sigmaSpaceTensor({{shape.x}, "N"}, nvcv::TYPE_F32);
        nvcv::Tensor sigmaColorTensor({{shape.x}, "N"}, nvcv::TYPE_F32);

        benchutils::FillTensor<int>(diameterTensor, [&diameter](const long4_16a &){ return diameter; });
        benchutils::FillTensor<float>(sigmaSpaceTensor, [&sigmaSpace](const long4_16a &){ return sigmaSpace; });
        benchutils::FillTensor<float>(sigmaColorTensor, [&sigmaColor](const long4_16a &){ return sigmaColor; });

        benchutils::warmup_and_exec(state, BENCH_JOINTBILATERALFILTER_WARMUP_ITERATIONS,
            [&op, &src, &color, &dst, &diameterTensor, &sigmaColorTensor, &sigmaSpaceTensor, &borderType](cudaStream_t s) {
                op(s, src, color, dst, diameterTensor, sigmaColorTensor, sigmaSpaceTensor, borderType);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(jointbilateralfilter, NVBENCH_TYPE_AXES(BENCH_JOINTBILATERALFILTER_TYPES))
BENCH_JOINTBILATERALFILTER_AXES;
