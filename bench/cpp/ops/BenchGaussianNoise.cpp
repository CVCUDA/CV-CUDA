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
#include "ops/generated/BenchGaussianNoiseConfig.hpp"

#include <cvcuda/OpGaussianNoise.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void gaussiannoise(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    using BT                              = typename nvcv::cuda::BaseType<T>;
    int ch                                = nvcv::cuda::NumElements<T>;

    bool perCh        = ch > 1;
    bool isPlanar     = layout == "NCHW";
    bool isFakePlanar = layout == "NCHW_FAKE";

    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("GaussianNoise benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) GaussianNoise benchmark is tensor-only");
        return;
    }

    unsigned long long int seed = 12345;

    const int64_t bytes = static_cast<int64_t>(shape.x) * shape.y * shape.z * sizeof(T);
    state.add_global_memory_reads((isFakePlanar ? 3 : 1) * bytes);
    state.add_global_memory_writes((isFakePlanar ? 3 : 1) * bytes);

    cvcuda::GaussianNoise op(shape.x);

    // clang-format off

    nvcv::Tensor mu({{shape.x}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigma({{shape.x}, "N"}, nvcv::TYPE_F32);

    benchutils::FillTensor<float>(mu, benchutils::LcgValues<float>(.0f, 1.f));
    benchutils::FillTensor<float>(sigma, benchutils::LcgValues<float>(.05f, .1f));

    if (isFakePlanar)
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_GAUSSIANNOISE_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &mu, &sigma, &perCh, &seed](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, mu, sigma, perCh, seed);
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

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_GAUSSIANNOISE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &mu, &sigma, &perCh, &seed](cudaStream_t s) {
                op(s, src, dst, mu, sigma, perCh, seed);
            });
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

        benchutils::warmup_and_exec(state, BENCH_GAUSSIANNOISE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &mu, &sigma, &perCh, &seed](cudaStream_t s) {
                op(s, src, dst, mu, sigma, perCh, seed);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(gaussiannoise, NVBENCH_TYPE_AXES(BENCH_GAUSSIANNOISE_TYPES))
BENCH_GAUSSIANNOISE_AXES;
