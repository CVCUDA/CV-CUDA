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
#include "ops/generated/BenchCropFlipNormalizeReformatConfig.hpp"

#include <cvcuda/OpCropFlipNormalizeReformat.hpp>
#include <cvcuda/OpNormalize.h>

#include <nvbench/nvbench.cuh>

inline uint32_t GetCropFlipNormalizeReformatFlags(const std::string &flagsMode)
{
    if (flagsMode == "normal")
    {
        return 0;
    }
    else if (flagsMode == "stddev")
    {
        return CVCUDA_NORMALIZE_SCALE_IS_STDDEV;
    }

    throw std::invalid_argument("Invalid flagsMode = " + flagsMode);
}

template<typename T>
inline void cropflipnormalizereformat(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3                       srcShape  = benchutils::GetShape<3>(state.get_string("shape"));
    auto                        cropMode  = state.get_string("cropMode");
    auto                        flagsMode = state.get_string("flagsMode");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    const std::string           srcLayout = state.get_string("srcLayout");
    const std::string           dstLayout = state.get_string("layout");
    long3                       dstShape  = srcShape;

    if (cropMode != "full" && cropMode != "padded16")
    {
        throw std::invalid_argument("Invalid cropMode = " + cropMode);
    }

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    if ((srcLayout != "NHWC" && srcLayout != "NCHW") || (dstLayout != "NHWC" && dstLayout != "NCHW"))
    {
        state.skip("CropFlipNormalizeReformat benchmark supports only NHWC and NCHW source/output layouts");
        return;
    }

    const bool srcPlanar = srcLayout == "NCHW";
    const bool dstPlanar = dstLayout == "NCHW";

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    float borderValue{0.f};

    float    globalScale = 1.234f;
    float    globalShift = 2.345f;
    float    epsilon     = 12.34f;
    uint32_t flags       = GetCropFlipNormalizeReformatFlags(flagsMode);

    long3 baseShape{srcShape.x, 1, 1};
    long3 scaleShape{srcShape.x, 1, 1};
    long3 cropShape{srcShape.x, 1, 1};

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(T)
                                  + baseShape.x * baseShape.y * baseShape.z * sizeof(float)
                                  + scaleShape.x * scaleShape.y * scaleShape.z * sizeof(float)
                                  + cropShape.x * cropShape.y * cropShape.z * sizeof(int) * 4);
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::CropFlipNormalizeReformat op;

    // clang-format off

    nvcv::Tensor dst = dstPlanar
        ? nvcv::Tensor({{dstShape.x, ch, dstShape.y, dstShape.z}, "NCHW"}, benchutils::GetDataType<BT>())
        : nvcv::Tensor({{dstShape.x, dstShape.y, dstShape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

    nvcv::Tensor flipCode({{srcShape.x}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor base({{baseShape.x, baseShape.y, baseShape.z, 1}, "NHWC"}, nvcv::TYPE_F32);
    nvcv::Tensor scale({{scaleShape.x, scaleShape.y, scaleShape.z, 1}, "NHWC"}, nvcv::TYPE_F32);

    nvcv::Tensor crop({{cropShape.x, cropShape.y, cropShape.z, 4}, "NHWC"}, nvcv::TYPE_S32);

    benchutils::FillTensor<int>(flipCode, [](auto &){ return -1; });

    // base, scale: default-range LcgValues<float>() so the GPU LCG fast
    // path produces float [-1, +1] bytes identical to the Python bench's
    // create_tensor(..., fill_mode="lcg").
    benchutils::FillTensor<float>(base, benchutils::LcgValues<float>());
    benchutils::FillTensor<float>(scale, benchutils::LcgValues<float>());

    // Always crop entire source image for easy bandwidth calculations
    benchutils::FillTensor<int>(crop, [&srcShape, &cropMode](const long4_16a &c)
    {
        if (c.w == 0)
        {
            return cropMode == "padded16" ? -16 : 0;
        }
        else if (c.w == 1)
        {
            return cropMode == "padded16" ? -16 : 0;
        }
        else if (c.w == 2)
        {
            return (int)srcShape.z;
        }
        else if (c.w == 3)
        {
            return (int)srcShape.y;
        }
        return 0;
    });

    if (inputKind == benchutils::InputKind::Tensor)
    {
        throw std::invalid_argument("Tensor not implemented for this operator");
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(static_cast<int32_t>(srcShape.x));

        if (srcPlanar && ch > 1)
        {
            benchutils::FillPlanarImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{0, 0});
        }
        else
        {
            benchutils::FillImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{0, 0},
                                          benchutils::CheckerboardValues<T>());
        }

        benchutils::warmup_and_exec(state, BENCH_CROPFLIPNORMALIZEREFORMAT_WARMUP_ITERATIONS,
            [&op, &src, &dst, &crop, &borderType, &borderValue, &flipCode, &base, &scale, &globalScale,
             &globalShift, &epsilon, &flags](cudaStream_t s) {
                op(s, src, dst, crop, borderType, borderValue, flipCode, base, scale, globalScale,
                   globalShift, epsilon, flags);
            });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(cropflipnormalizereformat, NVBENCH_TYPE_AXES(BENCH_CROPFLIPNORMALIZEREFORMAT_TYPES))
BENCH_CROPFLIPNORMALIZEREFORMAT_AXES;
