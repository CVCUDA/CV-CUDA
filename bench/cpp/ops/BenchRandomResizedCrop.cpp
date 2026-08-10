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
#include "ops/generated/BenchRandomResizedCropConfig.hpp"

#include <cvcuda/OpRandomResizedCrop.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void randomresizedcrop(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3                       srcShape  = benchutils::GetShape<3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    long3 dstShape;

    if (state.get_string("resizeType") == "EXPAND")
    {
        dstShape = long3{srcShape.x, srcShape.y * 2, srcShape.z * 2};
    }
    else if (state.get_string("resizeType") == "CONTRACT")
    {
        dstShape = long3{srcShape.x, srcShape.y / 2, srcShape.z / 2};
    }
    else
    {
        throw std::invalid_argument("Invalid resizeType = " + state.get_string("resizeType"));
    }

    double   minScale = 0.08;
    double   maxScale = 1.0;
    double   minRatio = 0.5;
    double   maxRatio = 2.0;
    uint32_t seed     = 1234;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("RandomResizedCrop benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) RandomResizedCrop benchmark is tensor-only");
        return;
    }

    const long srcBytes = srcShape.x * srcShape.y * srcShape.z * sizeof(T);
    const long dstBytes = dstShape.x * dstShape.y * dstShape.z * sizeof(T);
    if (isFakePlanar)
    {
        // reformat(NCHW->NHWC) + random-resized-crop + reformat(NHWC->NCHW):
        // reads src twice + dst once, writes the interleaved src once + dst twice.
        state.add_global_memory_reads(2 * srcBytes + dstBytes);
        state.add_global_memory_writes(srcBytes + 2 * dstBytes);
    }
    else
    {
        state.add_global_memory_reads(srcBytes);
        state.add_global_memory_writes(dstBytes);
    }

    cvcuda::RandomResizedCrop op(minScale, maxScale, minRatio, maxRatio, static_cast<int32_t>(srcShape.x), seed);
    using BT = typename nvcv::cuda::BaseType<T>;
    int ch   = nvcv::cuda::NumElements<T>;

    // clang-format off

    if (isFakePlanar) // tensor-only: planar -> interleaved -> random-resized-crop -> interleaved -> planar
    {
        nvcv::Tensor src     ({{srcShape.x, ch, srcShape.y, srcShape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{srcShape.x, srcShape.y, srcShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{dstShape.x, dstShape.y, dstShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{dstShape.x, ch, dstShape.y, dstShape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_RANDOMRESIZEDCROP_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &interpType](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, interpType);
                reformatOp(s, interDst, dst);
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{srcShape.x, ch, srcShape.y, srcShape.z}, "NCHW"},
                                            benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{srcShape.x, srcShape.y, srcShape.z, ch}, "NHWC"},
                                            benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{dstShape.x, ch, dstShape.y, dstShape.z}, "NCHW"},
                                            benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{dstShape.x, dstShape.y, dstShape.z, ch}, "NHWC"},
                                            benchutils::GetDataType<BT>());


        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_RANDOMRESIZEDCROP_WARMUP_ITERATIONS,
            [&op, &src, &dst, &interpType](cudaStream_t s) { op(s, src, dst, interpType); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(static_cast<int32_t>(srcShape.x));
        nvcv::ImageBatchVarShape dst(static_cast<int32_t>(dstShape.x));

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{0, 0});
            benchutils::FillPlanarImageBatch<T>(dst, long2{dstShape.z, dstShape.y}, long2{0, 0});
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{0, 0},
                                          benchutils::CheckerboardValues<T>());
            benchutils::FillImageBatch<T>(dst, long2{dstShape.z, dstShape.y}, long2{0, 0},
                                          [](const long4_16a &) { return T{0}; });
        }

        benchutils::warmup_and_exec(state, BENCH_RANDOMRESIZEDCROP_WARMUP_ITERATIONS,
            [&op, &src, &dst, &interpType](cudaStream_t s) { op(s, src, dst, interpType); });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(randomresizedcrop, NVBENCH_TYPE_AXES(BENCH_RANDOMRESIZEDCROP_TYPES))
BENCH_RANDOMRESIZEDCROP_AXES;
