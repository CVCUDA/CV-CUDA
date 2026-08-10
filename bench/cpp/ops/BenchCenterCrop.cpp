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
#include "ops/generated/BenchCenterCropConfig.hpp"

#include <cvcuda/OpCenterCrop.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void centercrop(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3                       srcShape  = benchutils::GetShape<3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("CenterCrop benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    // CenterCrop is tensor-only; the native planar (NCHW) and fake-planar (NCHW_FAKE) paths are too.
    if ((isPlanar || isFakePlanar) && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Planar CenterCrop benchmark is tensor-only");
        return;
    }

    nvcv::Size2D cropSize;

    if (state.get_string("cropType") == "SAME")
    {
        cropSize = nvcv::Size2D{(int)srcShape.z, (int)srcShape.y};
    }
    else if (state.get_string("cropType") == "QUARTER")
    {
        cropSize = nvcv::Size2D{(int)srcShape.z / 2, (int)srcShape.y / 2};
    }
    else
    {
        throw std::invalid_argument("Invalid cropType = " + state.get_string("cropType"));
    }

    long3 dstShape{srcShape.x, cropSize.h, cropSize.w};

    const long fullBytes = srcShape.x * srcShape.y * srcShape.z * sizeof(T);
    const long cropBytes = dstShape.x * dstShape.y * dstShape.z * sizeof(T);
    if (isFakePlanar)
    {
        // reformat(NCHW→NHWC) + crop + reformat(NHWC→NCHW).
        state.add_global_memory_reads(fullBytes + 2 * cropBytes);
        state.add_global_memory_writes(fullBytes + 2 * cropBytes);
    }
    else
    {
        state.add_global_memory_reads(cropBytes);
        state.add_global_memory_writes(cropBytes);
    }

    cvcuda::CenterCrop op;

    // clang-format off

    if (isFakePlanar) // tensor-only: planar→interleaved→crop→interleaved→planar
    {
        nvcv::Tensor src     ({{srcShape.x, ch, srcShape.y, srcShape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{srcShape.x, srcShape.y, srcShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{dstShape.x, dstShape.y, dstShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{dstShape.x, ch, dstShape.y, dstShape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_CENTERCROP_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &cropSize](cudaStream_t s) {
                reformatOp(s, src, interSrc);       // NCHW → NHWC
                op(s, interSrc, interDst, cropSize); // interleaved crop
                reformatOp(s, interDst, dst);       // NHWC → NCHW
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

        benchutils::warmup_and_exec(state, BENCH_CENTERCROP_WARMUP_ITERATIONS,
            [&op, &src, &dst, &cropSize](cudaStream_t s) { op(s, src, dst, cropSize); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(centercrop, NVBENCH_TYPE_AXES(BENCH_CENTERCROP_TYPES))
BENCH_CENTERCROP_AXES;
