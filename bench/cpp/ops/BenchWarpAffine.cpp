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
#include "ops/generated/BenchWarpAffineConfig.hpp"

#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpWarpAffine.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void warpaffine(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    NVCVBorderType        borderType = benchutils::GetBorderType(state.get_string("border"));
    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    int flags = interpType | ((state.get_string("inverseMap") == "Y") ? NVCV_WARP_INVERSE_MAP : 0);

    float4 borderValue{0, 0, 0, 0};

    NVCVAffineTransform transMatrix{2.f, 2.f, 0.f, 3.f, 1.f, 0.f};

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("WarpAffine benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    // NCHW_FAKE ("fake planar") is a tensor-only comparison path: planar data is reformatted to
    // interleaved, warped with the interleaved kernel, and reformatted back to planar — all timed
    // together — so the native planar path (NCHW) can be shown to be faster than this naive
    // convert→warp→convert pipeline.
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) WarpAffine benchmark is tensor-only");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;
    int ch   = nvcv::cuda::NumElements<T>;

    // WarpAffine preserves size, so src and dst hold the same number of bytes.
    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    if (isFakePlanar)
    {
        state.add_global_memory_reads(3 * bytes + 6 * sizeof(float));
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(bytes + 6 * sizeof(float));
        state.add_global_memory_writes(bytes);
    }

    cvcuda::WarpAffine op(shape.x);

    // clang-format off

    if (isFakePlanar) // tensor-only: planar→interleaved→warp→interleaved→planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_WARPAFFINE_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &transMatrix, &flags, &borderType, &borderValue](cudaStream_t s) {
                reformatOp(s, src, interSrc);                                       // NCHW → NHWC
                op(s, interSrc, interDst, transMatrix, flags, borderType, borderValue); // interleaved warp
                reformatOp(s, interDst, dst);                                       // NHWC → NCHW
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_WARPAFFINE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &transMatrix, &flags, &borderType, &borderValue](cudaStream_t s) {
                op(s, src, dst, transMatrix, flags, borderType, borderValue);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
            benchutils::FillPlanarImageBatch<T>(dst, long2{shape.z, shape.y}, long2{0, 0});
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                          benchutils::CheckerboardValues<T>());
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        nvcv::Tensor transMatrixTensor({{shape.x, 6}, "NW"}, nvcv::TYPE_F32);

        benchutils::FillTensor<float>(transMatrixTensor, [&transMatrix](const long4_16a &c){ return transMatrix[c.y]; });

        benchutils::warmup_and_exec(state, BENCH_WARPAFFINE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &transMatrixTensor, &flags, &borderType, &borderValue](cudaStream_t s) {
                op(s, src, dst, transMatrixTensor, flags, borderType, borderValue);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(warpaffine, NVBENCH_TYPE_AXES(BENCH_WARPAFFINE_TYPES))
BENCH_WARPAFFINE_AXES;
