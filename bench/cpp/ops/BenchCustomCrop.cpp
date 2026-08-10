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
#include "ops/generated/BenchCustomCropConfig.hpp"

#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void customcrop(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        cropMode  = state.get_string("cropMode");
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("CustomCrop benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    // CustomCrop is tensor-only; the native planar (NCHW) and fake-planar (NCHW_FAKE) paths are too.
    if ((isPlanar || isFakePlanar) && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Planar CustomCrop benchmark is tensor-only");
        return;
    }

    NVCVRectI cropRect{0, 0, shape.z, shape.y};
    if (cropMode == "center_half")
    {
        cropRect = NVCVRectI{shape.z / 4, shape.y / 4, shape.z / 2, shape.y / 2};
    }
    else if (cropMode != "full")
    {
        throw std::invalid_argument("Invalid cropMode = " + cropMode);
    }

    const long fullBytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    const long cropBytes = static_cast<long>(shape.x) * cropRect.height * cropRect.width * sizeof(T);
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

    cvcuda::CustomCrop op;

    // clang-format off

    if (isFakePlanar) // tensor-only: planar→interleaved→crop→interleaved→planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, cropRect.height, cropRect.width, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, cropRect.height, cropRect.width}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_CUSTOMCROP_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &cropRect](cudaStream_t s) {
                reformatOp(s, src, interSrc);          // NCHW → NHWC
                op(s, interSrc, interDst, cropRect);   // interleaved crop
                reformatOp(s, interDst, dst);          // NHWC → NCHW
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, cropRect.height, cropRect.width}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, cropRect.height, cropRect.width, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_CUSTOMCROP_WARMUP_ITERATIONS,
            [&op, &src, &dst, &cropRect](cudaStream_t s) { op(s, src, dst, cropRect); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(customcrop, NVBENCH_TYPE_AXES(BENCH_CUSTOMCROP_TYPES))
BENCH_CUSTOMCROP_AXES;
