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
#include "ops/generated/BenchPadAndStackConfig.hpp"

#include <cvcuda/OpPadAndStack.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

template<typename T>
inline void padandstack(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    int                         pad       = benchutils::GetIntParam<int>(state, "pad");
    const bool                  isPlanar  = layout == "NCHW";
    if (layout != "NHWC" && layout != "NCHW")
    {
        state.skip("PadAndStack benchmark supports only NHWC and NCHW layouts");
        return;
    }
    if (isPlanar && nvcv::cuda::NumElements<T> == 2)
    {
        state.skip("Planar PadAndStack benchmark does not support 2-channel layouts");
        return;
    }
    if (isPlanar && nvcv::cuda::NumElements<T> == 1)
    {
        state.skip("Single-channel PadAndStack has no distinct planar image-batch format");
        return;
    }

    auto numBatches = shape.x;
    auto srcHeight  = shape.y;
    auto srcWidth   = shape.z;

    int dstHeight = srcHeight + 2 * pad;
    int dstWidth  = srcWidth + 2 * pad;

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));
    float          borderValue{0.f};

    // Memory accounting: read all src images + top/left, write dst tensor
    state.add_global_memory_reads(numBatches * srcHeight * srcWidth * sizeof(T) + numBatches * sizeof(int) * 2);
    state.add_global_memory_writes(numBatches * dstHeight * dstWidth * sizeof(T));

    // PadAndStack only supports ImageBatchVarShape input
    if (inputKind == benchutils::InputKind::Tensor)
    {
        state.skip("PadAndStack only supports ImageBatchVarShape input (inputKind=VarShape)");
        return;
    }

    nvcv::ImageBatchVarShape src(numBatches);
    if (isPlanar)
    {
        benchutils::FillPlanarImageBatch<T>(src, long2{srcWidth, srcHeight}, long2{0, 0});
    }
    else
    {
        benchutils::FillImageBatch<T>(src, long2{srcWidth, srcHeight}, long2{0, 0},
                                      benchutils::CheckerboardValues<T>());
    }

    using BT               = typename nvcv::cuda::BaseType<T>;
    constexpr int channels = nvcv::cuda::NumElements<T>;
    nvcv::Tensor  dst
        = isPlanar ? nvcv::Tensor(
              {
                  {numBatches, channels, dstHeight, dstWidth},
                  "NCHW"
    },
              benchutils::GetDataType<BT>())
                   : nvcv::Tensor({{numBatches, dstHeight, dstWidth, channels}, "NHWC"}, benchutils::GetDataType<BT>());

    // Create top/left padding tensors (OLD style - 4D NHWC)
    nvcv::Tensor top(
        {
            {numBatches, 1, 1, 1},
            "NHWC"
    },
        nvcv::TYPE_S32);
    nvcv::Tensor left(
        {
            {numBatches, 1, 1, 1},
            "NHWC"
    },
        nvcv::TYPE_S32);

    // Fill top/left using FillTensor (OLD style)
    benchutils::FillTensor<int>(top, [dstHeight, srcHeight](const long4_16a &) { return (dstHeight - srcHeight) / 2; });
    benchutils::FillTensor<int>(left, [dstWidth, srcWidth](const long4_16a &) { return (dstWidth - srcWidth) / 2; });

    // Create operator
    cvcuda::PadAndStack op;

    benchutils::warmup_and_exec(state, BENCH_PADANDSTACK_WARMUP_ITERATIONS,
                                [&op, &src, &dst, &top, &left, &borderType, &borderValue](cudaStream_t s)
                                { op(s, src, dst, top, left, borderType, borderValue); });
}

CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json
NVBENCH_BENCH_TYPES(padandstack, NVBENCH_TYPE_AXES(BENCH_PADANDSTACK_TYPES))
BENCH_PADANDSTACK_AXES;
