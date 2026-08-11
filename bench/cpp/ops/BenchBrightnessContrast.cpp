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
#include "ops/generated/BenchBrightnessContrastConfig.hpp"

#include <cvcuda/OpBrightnessContrast.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <type_traits>

template<typename T>
inline void brightnesscontrast(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("BrightnessContrast benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) BrightnessContrast benchmark is tensor-only");
        return;
    }
    if (isPlanar && inputKind == benchutils::InputKind::VarShape && ch == 4 && std::is_same_v<BT, unsigned char>)
    {
        state.skip("uchar4 planar ImageBatchVarShape is not supported by the Python image API");
        return;
    }

    const long bytes    = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    const long argBytes = static_cast<long>(shape.x) * sizeof(float) * 4;
    if (isFakePlanar)
    {
        state.add_global_memory_reads(3 * bytes + argBytes);
        state.add_global_memory_writes(3 * bytes);
    }
    else
    {
        state.add_global_memory_reads(bytes + argBytes);
        state.add_global_memory_writes(bytes);
    }

    cvcuda::BrightnessContrast op;

    // clang-format off

    nvcv::Tensor brightness({{shape.x}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor contrast({{shape.x}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor brightnessShift({{shape.x}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor contrastCenter({{shape.x}, "N"}, nvcv::TYPE_F32);

    // All 4 param tensors use the deterministic LCG GPU fast path so the
    // bytes match the Python bench's fill_mode="lcg" exactly.
    benchutils::FillTensor<float>(brightness, benchutils::LcgValues<float>());
    benchutils::FillTensor<float>(contrast, benchutils::LcgValues<float>());
    benchutils::FillTensor<float>(brightnessShift, benchutils::LcgValues<float>());
    benchutils::FillTensor<float>(contrastCenter, benchutils::LcgValues<float>());

    if (isFakePlanar) // tensor-only: planar->interleaved->brightnesscontrast->interleaved->planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_BRIGHTNESSCONTRAST_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &brightness, &contrast, &brightnessShift, &contrastCenter](cudaStream_t s) {
                reformatOp(s, src, interSrc); // NCHW -> NHWC
                op(s, interSrc, interDst, brightness, contrast, brightnessShift, contrastCenter);
                reformatOp(s, interDst, dst); // NHWC -> NCHW
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

        benchutils::warmup_and_exec(state, BENCH_BRIGHTNESSCONTRAST_WARMUP_ITERATIONS,
            [&op, &src, &dst, &brightness, &contrast, &brightnessShift, &contrastCenter](cudaStream_t s) {
                op(s, src, dst, brightness, contrast, brightnessShift, contrastCenter);
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

        benchutils::warmup_and_exec(state, BENCH_BRIGHTNESSCONTRAST_WARMUP_ITERATIONS,
            [&op, &src, &dst, &brightness, &contrast, &brightnessShift, &contrastCenter](cudaStream_t s) {
                op(s, src, dst, brightness, contrast, brightnessShift, contrastCenter);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(brightnesscontrast, NVBENCH_TYPE_AXES(BENCH_BRIGHTNESSCONTRAST_TYPES))
BENCH_BRIGHTNESSCONTRAST_AXES;
