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
#include "ops/generated/BenchWarpPerspectiveConfig.hpp"

#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpWarpPerspective.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void warpperspective(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    NVCVBorderType        borderType = benchutils::GetBorderType(state.get_string("border"));
    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    int flags = interpType | ((state.get_string("inverseMap") == "Y") ? NVCV_WARP_INVERSE_MAP : 0);

    float4 borderValue{0, 0, 0, 0};

    NVCVPerspectiveTransform transMatrix{0.27f, 0.16f, 0.00f, -0.11f, 0.61f, 0.65f, -0.09f, 0.06f, 1.00f};

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("WarpPerspective benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) WarpPerspective benchmark is tensor-only");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;
    int ch   = nvcv::cuda::NumElements<T>;

    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    if (isFakePlanar)
    {
        state.add_global_memory_reads(3 * bytes + 9 * sizeof(float));
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(bytes + 9 * sizeof(float));
        state.add_global_memory_writes(bytes);
    }

    cvcuda::WarpPerspective op(shape.x);

    // clang-format off

    if (isFakePlanar) // tensor-only: planar -> interleaved -> warp -> interleaved -> planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_WARPPERSPECTIVE_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &transMatrix, &flags, &borderType, &borderValue](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, transMatrix, flags, borderType, borderValue);
                reformatOp(s, interDst, dst);
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

        benchutils::warmup_and_exec(state, BENCH_WARPPERSPECTIVE_WARMUP_ITERATIONS,
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

        nvcv::Tensor transMatrixTensor({{shape.x, 9}, "NW"}, nvcv::TYPE_F32);

        benchutils::FillTensor<float>(transMatrixTensor, [&transMatrix](auto &c){ return transMatrix[c.y]; });

        benchutils::warmup_and_exec(state, BENCH_WARPPERSPECTIVE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &transMatrixTensor, &flags, &borderType, &borderValue](cudaStream_t s) {
                op(s, src, dst, transMatrixTensor, flags, borderType, borderValue);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(warpperspective, NVBENCH_TYPE_AXES(BENCH_WARPPERSPECTIVE_TYPES))
BENCH_WARPPERSPECTIVE_AXES;
