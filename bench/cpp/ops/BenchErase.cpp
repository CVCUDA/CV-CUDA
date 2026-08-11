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
#include "ops/generated/BenchEraseConfig.hpp"

#include <cvcuda/OpErase.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <string_view>

inline bool GetEraseRandomMode(const std::string &randomMode)
{
    if (randomMode == "random")
    {
        return true;
    }
    else if (randomMode == "constant")
    {
        return false;
    }

    throw std::invalid_argument("Invalid randomMode = " + randomMode);
}

template<typename BT>
inline const char *GetEraseSkipReason(std::string_view layout, benchutils::InputKind inputKind, bool regionMode,
                                      int channels)
{
    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";

    if (layout != "NHWC" && !isPlanar && !isFakePlanar)
        return "Erase benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts";
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
        return "Fake-planar (NCHW_FAKE) erase benchmark is tensor-only";
    if (regionMode && inputKind == benchutils::InputKind::VarShape)
        return "Torchvision Erase region benchmark is tensor-only";
    if (!regionMode && isPlanar && channels == 2)
        return "Planar Erase benchmark does not support 2-channel layouts";
    if (isPlanar && inputKind == benchutils::InputKind::VarShape && channels == 1)
        return "Single-channel varshape Erase has no distinct planar image layout";
    if (isPlanar && inputKind == benchutils::InputKind::VarShape && std::is_same_v<BT, uint8_t> && channels == 4)
        return "RGBA8p varshape is unsupported by the Python image API";

    return nullptr;
}

template<typename T>
inline void erase(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape      = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        randomMode = state.get_string("randomMode");
    auto                        layout     = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind  = benchutils::GetInputKind(state.get_string("inputKind"));
    int                         numErase   = benchutils::GetIntParam<int>(state, "numErase");

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    const bool regionMode = randomMode == "torchvision";
    bool       random     = regionMode ? false : GetEraseRandomMode(randomMode);
    int        seed       = 0;

    const bool isPlanar     = layout == "NCHW";
    const bool isFakePlanar = layout == "NCHW_FAKE";
    if (const char *reason = GetEraseSkipReason<BT>(layout, inputKind, regionMode, ch))
    {
        state.skip(reason);
        return;
    }

    const long imageBytes   = static_cast<long>(shape.x) * shape.y * shape.z * sizeof(T);
    const int  regionHeight = std::max(shape.y / 4, 1);
    const int  regionWidth  = std::max(shape.z / 4, 1);
    const long paramBytes
        = regionMode ? static_cast<long>(ch) * regionHeight * regionWidth * sizeof(float)
                     : static_cast<long>(numErase) * (sizeof(int2) + sizeof(int3) + ch * sizeof(float) + sizeof(int));
    if (isFakePlanar)
    {
        state.add_global_memory_reads(3 * imageBytes + paramBytes);
        state.add_global_memory_writes(3 * imageBytes);
    }
    else
    {
        state.add_global_memory_reads(imageBytes + paramBytes);
        state.add_global_memory_writes(imageBytes);
    }

    cvcuda::Erase op(numErase);

    // clang-format off

    nvcv::Tensor anchor({{numErase}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor erasing({{numErase}, "N"}, nvcv::TYPE_3S32);
    nvcv::Tensor values({{numErase * ch}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor imgIdx({{numErase}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor regionValues({{ch, regionHeight, regionWidth}, "CHW"}, nvcv::TYPE_F32);

    int eraseMask = (1 << ch) - 1;

    benchutils::FillTensor<int2>(anchor, [](const long4_16a &){ return int2{0, 0}; });
    benchutils::FillTensor<int3>(erasing, [&eraseMask](const long4_16a &){ return int3{10, 10, eraseMask}; });
    benchutils::FillTensor<float>(values, [](const long4_16a &){ return 1.f; });
    benchutils::FillTensor<int>(imgIdx, [](const long4_16a &){ return 0; });
    benchutils::FillTensor<float>(regionValues, [](const long4_16a &){ return 1.f; });

    if (isFakePlanar) // tensor-only: planar→interleaved→erase→interleaved→planar
    {
        nvcv::Tensor src     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor interDst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst     ({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_ERASE_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &anchor, &erasing, &values, &imgIdx, &regionValues, regionMode, regionHeight, regionWidth, &random, &seed](cudaStream_t s) {
                reformatOp(s, src, interSrc); // NCHW → NHWC
                if (regionMode)
                    op(s, interSrc, interDst, 0, 0, regionHeight, regionWidth, regionValues);
                else
                    op(s, interSrc, interDst, anchor, erasing, values, imgIdx, random, seed);
                reformatOp(s, interDst, dst); // NHWC → NCHW
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

        benchutils::warmup_and_exec(state, BENCH_ERASE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &anchor, &erasing, &values, &imgIdx, &regionValues, regionMode, regionHeight, regionWidth, &random, &seed](cudaStream_t s) {
                if (regionMode)
                    op(s, src, dst, 0, 0, regionHeight, regionWidth, regionValues);
                else
                    op(s, src, dst, anchor, erasing, values, imgIdx, random, seed);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

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

        benchutils::warmup_and_exec(state, BENCH_ERASE_WARMUP_ITERATIONS,
            [&op, &src, &dst, &anchor, &erasing, &values, &imgIdx, &random, &seed](cudaStream_t s) {
                op(s, src, dst, anchor, erasing, values, imgIdx, random, seed);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(erase, NVBENCH_TYPE_AXES(BENCH_ERASE_TYPES))
BENCH_ERASE_AXES;
