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
#include "ops/generated/BenchCLAHEConfig.hpp"

#include <cvcuda/OpCLAHE.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void clahe(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    int                         tilesX    = benchutils::GetIntParam<int>(state, "tilesX");
    int                         tilesY    = benchutils::GetIntParam<int>(state, "tilesY");
    int                         clip10    = benchutils::GetIntParam<int>(state, "clip10");

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("CLAHE benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) CLAHE benchmark is tensor-only");
        return;
    }

    const auto clipLimit = static_cast<float>(static_cast<double>(clip10) / 10.0);

    using BT = typename nvcv::cuda::BaseType<T>;
    int ch   = nvcv::cuda::NumElements<T>;

    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    state.add_global_memory_reads(isFakePlanar ? 3 * bytes : bytes);
    state.add_global_memory_writes(isFakePlanar ? 3 * bytes : bytes);

    cvcuda::CLAHE op(shape.x, tilesX, tilesY);

    // clang-format off

    if (isFakePlanar)
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::LcgValues<BT>());

        cvcuda::Reformat reformatOp;
        benchutils::warmup_and_exec(state, BENCH_CLAHE_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, clipLimit](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, clipLimit);
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

        benchutils::warmup_and_exec(state, BENCH_CLAHE_WARMUP_ITERATIONS,
            [&op, &src, &dst, clipLimit](cudaStream_t s) { op(s, src, dst, clipLimit); });
    }
    else
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);
        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                      benchutils::LcgValues<T>());
        benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });

        benchutils::warmup_and_exec(state, BENCH_CLAHE_WARMUP_ITERATIONS,
            [&op, &src, &dst, clipLimit](cudaStream_t s) { op(s, src, dst, clipLimit); });
    }

    // clang-format on
}

CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(clahe, NVBENCH_TYPE_AXES(BENCH_CLAHE_TYPES))
BENCH_CLAHE_AXES;
