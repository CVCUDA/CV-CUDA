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
#include "ops/generated/BenchStackConfig.hpp"

#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpStack.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

template<typename BT>
inline void RunFakePlanarStackBench(nvbench::state &state, cvcuda::Stack &op, const int3 &shape, int ch)
{
    nvcv::TensorBatch         interSrc(nvcv::TensorBatch::CalcRequirements(shape.x));
    std::vector<nvcv::Tensor> planarSrcTensors;
    std::vector<nvcv::Tensor> interSrcTensors;
    planarSrcTensors.reserve(shape.x);
    interSrcTensors.reserve(shape.x);

    for (int i = 0; i < shape.x; ++i)
    {
        planarSrcTensors.emplace_back(nvcv::Tensor(
            {
                {ch, shape.y, shape.z},
                "CHW"
        },
            benchutils::GetDataType<BT>()));
        interSrcTensors.emplace_back(nvcv::Tensor(
            {
                {shape.y, shape.z, ch},
                "HWC"
        },
            benchutils::GetDataType<BT>()));
        benchutils::FillTensor<BT>(planarSrcTensors.back(), benchutils::CheckerboardValues<BT>());
        interSrc.pushBack(interSrcTensors.back());
    }

    nvcv::Tensor interDst(
        {
            {shape.x, shape.y, shape.z, ch},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor dst(
        {
            {shape.x, ch, shape.y, shape.z},
            "NCHW"
    },
        benchutils::GetDataType<BT>());
    cvcuda::Reformat reformatOp;

    benchutils::warmup_and_exec(
        state, BENCH_STACK_WARMUP_ITERATIONS,
        [&op, &reformatOp, &planarSrcTensors, &interSrcTensors, &interSrc, &interDst, &dst](cudaStream_t s)
        {
            for (size_t i = 0; i < planarSrcTensors.size(); ++i)
            {
                reformatOp(s, planarSrcTensors[i], interSrcTensors[i]); // CHW -> HWC
            }
            op(s, interSrc, interDst);    // interleaved Stack
            reformatOp(s, interDst, dst); // NHWC -> NCHW
        });
}

template<typename T>
inline void stack(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("Stack benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Fake-planar (NCHW_FAKE) Stack benchmark is TensorBatch-only");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    state.add_global_memory_reads((isFakePlanar ? 3 : 1) * bytes);
    state.add_global_memory_writes((isFakePlanar ? 3 : 1) * bytes);

    cvcuda::Stack op;

    // clang-format off

    if (isFakePlanar)
    {
        RunFakePlanarStackBench<BT>(state, op, shape, ch);
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::TensorBatch src(nvcv::TensorBatch::CalcRequirements(shape.x));

        for (int i = 0 ; i < shape.x; i++)
        {
            nvcv::Tensor srcIn = isPlanar
                                   ? nvcv::Tensor({{ch, shape.y, shape.z}, "CHW"}, benchutils::GetDataType<BT>())
                                   : nvcv::Tensor({{shape.y, shape.z, ch}, "HWC"}, benchutils::GetDataType<BT>());
            benchutils::FillTensor<BT>(srcIn, benchutils::CheckerboardValues<BT>());
            src.pushBack(srcIn);
        }

        benchutils::warmup_and_exec(state, BENCH_STACK_WARMUP_ITERATIONS,
            [&op, &src, &dst](cudaStream_t s) { op(s, src, dst); });
    }
    else
    {
        nvcv::Tensor dst = isPlanar
                             ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                             : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::ImageBatchVarShape src(shape.x);
        if (isPlanar && ch > 1)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                          benchutils::CheckerboardValues<T>());
        }

        benchutils::warmup_and_exec(state, BENCH_STACK_WARMUP_ITERATIONS,
            [&op, &src, &dst](cudaStream_t s) { op(s, src, dst); });
    }

}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(stack, NVBENCH_TYPE_AXES(BENCH_STACK_TYPES))
BENCH_STACK_AXES;
