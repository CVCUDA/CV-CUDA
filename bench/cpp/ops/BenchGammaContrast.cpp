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
#include "ops/generated/BenchGammaContrastConfig.hpp"

#include <cvcuda/OpGammaContrast.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <type_traits>

template<typename T>
inline void gammacontrast(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3       shape        = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto       layout       = benchutils::GetStringParam(state, "layout", "NHWC");
    const auto inputKindStr = state.get_string("inputKind");

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    // ScalarGamma benches the host-scalar gamma/gain overload: same dense-tensor input as Tensor,
    // but gamma/gain are kernel launch arguments -- no device gamma tensor is staged or read.
    const bool isScalarGamma = inputKindStr == "ScalarGamma";
    const bool isVarShape    = inputKindStr == "VarShape";
    if (inputKindStr != "Tensor" && !isVarShape && !isScalarGamma)
    {
        state.skip("GammaContrast benchmark supports only Tensor, VarShape, and ScalarGamma input kinds");
        return;
    }
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("GammaContrast benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKindStr != "Tensor")
    {
        state.skip("Fake-planar (NCHW_FAKE) GammaContrast benchmark is tensor-only");
        return;
    }
    if (isPlanar && isVarShape && std::is_same_v<BT, uint8_t> && ch == 4)
    {
        state.skip("RGBA8p varshape is unsupported by the Python image API");
        return;
    }

    const long imageBytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    const long gammaBytes = isScalarGamma ? 0 : static_cast<long>(shape.x) * ch * sizeof(float);
    if (isFakePlanar)
    {
        state.add_global_memory_reads(3 * imageBytes + gammaBytes);
        state.add_global_memory_writes(3 * imageBytes);
    }
    else
    {
        state.add_global_memory_reads(imageBytes + gammaBytes);
        state.add_global_memory_writes(imageBytes);
    }

    cvcuda::GammaContrast op(shape.x, ch);

    // clang-format off

    nvcv::Tensor gamma({{shape.x * ch}, "N"}, nvcv::TYPE_F32);

    // gamma: deterministic constant 0.75 (Python bench mirrors this). The ScalarGamma rows pass the
    // same 0.75 by value with gain 1.0 -- bit-exact with the Tensor rows -- and never read this tensor.
    benchutils::FillTensor<float>(gamma, [](const long4_16a &) { return 0.75f; });

    if (isFakePlanar) // tensor-only: planar->interleaved->gammacontrast->interleaved->planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_GAMMACONTRAST_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &gamma](cudaStream_t s) {
                reformatOp(s, src, interSrc); // NCHW -> NHWC
                op(s, interSrc, interDst, gamma);
                reformatOp(s, interDst, dst); // NHWC -> NCHW
            });
    }
    else if (inputKindStr == "Tensor" || isScalarGamma)
    {
        nvcv::Tensor src = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());

        if (isScalarGamma)
        {
            benchutils::warmup_and_exec(state, BENCH_GAMMACONTRAST_WARMUP_ITERATIONS,
                [&op, &src, &dst](cudaStream_t s) { op(s, src, dst, 0.75f, 1.0f); });
        }
        else
        {
            benchutils::warmup_and_exec(state, BENCH_GAMMACONTRAST_WARMUP_ITERATIONS,
                [&op, &src, &dst, &gamma](cudaStream_t s) { op(s, src, dst, gamma); });
        }
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatchLcg<T>(src, long2{shape.z, shape.y}, long2{0, 0});
            benchutils::FillPlanarImageBatchLike<T>(dst, src);
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                          benchutils::LcgValues<T>());
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        benchutils::warmup_and_exec(state, BENCH_GAMMACONTRAST_WARMUP_ITERATIONS,
            [&op, &src, &dst, &gamma](cudaStream_t s) { op(s, src, dst, gamma); });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(gammacontrast, NVBENCH_TYPE_AXES(BENCH_GAMMACONTRAST_TYPES))
BENCH_GAMMACONTRAST_AXES;
