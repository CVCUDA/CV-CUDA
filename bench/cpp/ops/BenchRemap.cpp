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
#include "ops/generated/BenchRemapConfig.hpp"

#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpRemap.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void remap(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3                       srcShape  = benchutils::GetShape<3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    long3                       dstShape  = srcShape;
    long3                       mapShape;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("Remap benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    // NCHW_FAKE ("fake planar") is the tensor-only convert-remap-convert comparison baseline.
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) remap benchmark is tensor-only");
        return;
    }

    NVCVInterpolationType srcInterp;
    NVCVInterpolationType mapInterp;
    NVCVBorderType        borderType;
    NVCVRemapMapValueType mapValueType;

    bool   alignCorners{true};
    float4 borderValue{0, 0, 0, 0};

    if (state.get_string("mapType") == "DENSE")
    {
        srcInterp    = NVCV_INTERP_NEAREST;
        mapInterp    = NVCV_INTERP_NEAREST;
        borderType   = NVCV_BORDER_CONSTANT;
        mapValueType = NVCV_REMAP_ABSOLUTE_NORMALIZED;
        mapShape     = srcShape;
    }
    else if (state.get_string("mapType") == "RELATIVE")
    {
        srcInterp    = NVCV_INTERP_CUBIC;
        mapInterp    = NVCV_INTERP_CUBIC;
        borderType   = NVCV_BORDER_REFLECT101;
        mapValueType = NVCV_REMAP_RELATIVE_NORMALIZED;
        mapShape     = long3{srcShape.x, 4, 4};
    }
    else
    {
        throw std::invalid_argument("Invalid mapType = " + state.get_string("mapType"));
    }

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(T)
                                  + mapShape.x * mapShape.y * mapShape.z * sizeof(float2));
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::Remap op;

    // clang-format off

    nvcv::Tensor map({{mapShape.x, mapShape.y, mapShape.z, 1}, "NHWC"}, nvcv::TYPE_2F32);

    // Map tensor stays random: its values determine the source-pixel access
    // pattern (cache behaviour), not just sample data — checkerboard would
    // give pattern-perfect cache hits and skew kernel timing.
    benchutils::FillTensor<float2>(map, benchutils::LcgValues<float2>());

    using BT = typename nvcv::cuda::BaseType<T>;
    int  ch  = nvcv::cuda::NumElements<T>;

    if (isFakePlanar) // tensor-only: planar->interleaved->remap->interleaved->planar
    {
        nvcv::Tensor src     ({{srcShape.x, ch, srcShape.y, srcShape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{srcShape.x, srcShape.y, srcShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{dstShape.x, dstShape.y, dstShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{dstShape.x, ch, dstShape.y, dstShape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_REMAP_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &map, &srcInterp, &mapInterp, &mapValueType, &alignCorners, &borderType, &borderValue](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, map, srcInterp, mapInterp, mapValueType, alignCorners, borderType, borderValue);
                reformatOp(s, interDst, dst);
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{srcShape.x, ch, srcShape.y, srcShape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{srcShape.x, srcShape.y, srcShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{dstShape.x, ch, dstShape.y, dstShape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{dstShape.x, dstShape.y, dstShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_REMAP_WARMUP_ITERATIONS,
            [&op, &src, &dst, &map, &srcInterp, &mapInterp, &mapValueType, &alignCorners, &borderType, &borderValue](cudaStream_t s) {
                op(s, src, dst, map, srcInterp, mapInterp, mapValueType, alignCorners, borderType, borderValue);
            });
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
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        benchutils::warmup_and_exec(state, BENCH_REMAP_WARMUP_ITERATIONS,
            [&op, &src, &dst, &map, &srcInterp, &mapInterp, &mapValueType, &alignCorners, &borderType, &borderValue](cudaStream_t s) {
                op(s, src, dst, map, srcInterp, mapInterp, mapValueType, alignCorners, borderType, borderValue);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(remap, NVBENCH_TYPE_AXES(BENCH_REMAP_TYPES))
BENCH_REMAP_AXES;
