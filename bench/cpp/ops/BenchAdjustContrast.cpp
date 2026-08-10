/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ops/generated/BenchAdjustContrastConfig.hpp"

#include <cvcuda/OpAdjustContrast.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

// Benchmark for the Adjust Contrast operator across the interleaved (NHWC), native planar (NCHW),
// fake-planar (NCHW_FAKE), and ImageBatchVarShape paths. AdjustContrast is a two-pass operator: a
// per-image grayscale-mean reduction followed by the clamped affine blend.
template<typename T>
inline void adjustcontrast(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("AdjustContrast benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) adjust_contrast benchmark is tensor-only");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;

    // Fixed contrast factor > 1: blends away from the grayscale mean and forces the [0, bound] clamp
    // on both ends, matching the correctness matrix's work path.
    const double contrastFactor = 1.5;

    int ch = nvcv::cuda::NumElements<T>;

    // AdjustContrast reads the image once for the mean reduction and once for the blend, and writes
    // it once (plus the reformat traffic on the fake-planar path).
    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    if (isFakePlanar)
    {
        state.add_global_memory_reads(4 * bytes);
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(2 * bytes);
        state.add_global_memory_writes(bytes);
    }

    cvcuda::AdjustContrast op;

    // clang-format off
    if (isFakePlanar) // tensor-only: planar->interleaved->adjust->interleaved->planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_ADJUSTCONTRAST_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, contrastFactor](cudaStream_t s) {
                reformatOp(s, src, interSrc);                   // NCHW -> NHWC
                op(s, interSrc, interDst, contrastFactor);      // interleaved adjust
                reformatOp(s, interDst, dst);                   // NHWC -> NCHW
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

        benchutils::warmup_and_exec(state, BENCH_ADJUSTCONTRAST_WARMUP_ITERATIONS,
            [&op, &src, &dst, contrastFactor](cudaStream_t s) { op(s, src, dst, contrastFactor); });
    }
    else // ImageBatchVarShape
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
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        benchutils::warmup_and_exec(state, BENCH_ADJUSTCONTRAST_WARMUP_ITERATIONS,
            [&op, &src, &dst, contrastFactor](cudaStream_t s) { op(s, src, dst, contrastFactor); });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(adjustcontrast, NVBENCH_TYPE_AXES(BENCH_ADJUSTCONTRAST_TYPES))
BENCH_ADJUSTCONTRAST_AXES;
