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
#include "ops/generated/BenchConv2DConfig.hpp"

#include <cvcuda/OpConv2D.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void conv2d(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    int2 kernelSize = nvcv::cuda::StaticCast<int>(benchutils::GetShape<2>(state.get_string("kernelSize")));

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    if (layout != "NHWC" && layout != "NCHW")
    {
        state.skip("Conv2D benchmark supports only NHWC and NCHW layouts");
        return;
    }

    constexpr int ch       = nvcv::cuda::NumElements<T>;
    const bool    isPlanar = layout == "NCHW";
    if (isPlanar && ch == 1)
    {
        state.skip("Single-channel Conv2D has no distinct planar image-batch layout");
        return;
    }
    if (isPlanar && ch == 2)
    {
        state.skip("Planar Conv2D benchmark does not support 2-channel layouts");
        return;
    }

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Conv2D op;

    // clang-format off

    nvcv::Tensor kernelAnchor({{shape.x}, "N"}, nvcv::TYPE_2S32);

    benchutils::FillTensor<int2>(kernelAnchor, [](auto &){ return int2{-1, -1}; });

    if (inputKind == benchutils::InputKind::Tensor)
    {
        throw std::invalid_argument("Tensor not implemented for this operator");
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);
        nvcv::ImageBatchVarShape kernel(shape.x);

        if (isPlanar)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
            benchutils::FillPlanarImageBatchLike<T>(dst, src);
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0},
                                          benchutils::CheckerboardValues<T>());
            // Use FillImageBatchLike to ensure dst has same per-sample shapes as src
            benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
        }

        // Use default-range LcgValues so the GPU LCG fast path runs on the
        // device; the Python bench's fill_mode="lcg" produces bit-identical
        // float32 weights via the same kernel.
        benchutils::FillImageBatch<float>(kernel, long2{kernelSize.x, kernelSize.y}, long2{0, 0},
                                          benchutils::LcgValues<float>());

        benchutils::warmup_and_exec(state, BENCH_CONV2D_WARMUP_ITERATIONS,
            [&op, &src, &dst, &kernel, &kernelAnchor, &borderType](cudaStream_t s) {
                op(s, src, dst, kernel, kernelAnchor, borderType);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(conv2d, NVBENCH_TYPE_AXES(BENCH_CONV2D_TYPES))
BENCH_CONV2D_AXES;
