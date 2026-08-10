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
#include "ops/generated/BenchAdvCvtColorConfig.hpp"

#include <cvcuda/OpAdvCvtColor.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

enum class ConversionShape
{
    kInterleaved444,
    kRgbToNv,
    kNvToRgb
};

struct ConversionInfo
{
    NVCVColorConversionCode code;
    ConversionShape         shape;
};

struct ConversionTensorShape
{
    int srcH;
    int dstH;
    int srcC;
    int dstC;
};

inline ConversionInfo GetConversionInfo(const std::string &code) // NOSONAR: S3776 is misattributed to this lookup.
{
    // clang-format off
    static const std::map<std::string, ConversionInfo, std::less<>> codeMap {
        {     "BGR2YUV", {NVCV_COLOR_BGR2YUV,      ConversionShape::kInterleaved444}},
        {     "RGB2YUV", {NVCV_COLOR_RGB2YUV,      ConversionShape::kInterleaved444}},
        {     "YUV2BGR", {NVCV_COLOR_YUV2BGR,      ConversionShape::kInterleaved444}},
        {     "YUV2RGB", {NVCV_COLOR_YUV2RGB,      ConversionShape::kInterleaved444}},
        {"RGB2YUV_NV12", {NVCV_COLOR_RGB2YUV_NV12, ConversionShape::kRgbToNv}},
        {"BGR2YUV_NV21", {NVCV_COLOR_BGR2YUV_NV21, ConversionShape::kRgbToNv}},
        {"YUV2RGB_NV12", {NVCV_COLOR_YUV2RGB_NV12, ConversionShape::kNvToRgb}},
        {"YUV2BGR_NV21", {NVCV_COLOR_YUV2BGR_NV21, ConversionShape::kNvToRgb}},
    };
    // clang-format on

    if (auto it = codeMap.find(code); it != codeMap.end())
    {
        return it->second;
    }

    throw std::invalid_argument("Unrecognized AdvCvtColor conversion code");
}

inline bool IsPlanarLayout(std::string_view layout)
{
    return layout == "NCHW";
}

inline bool IsFakePlanarLayout(std::string_view layout)
{
    return layout == "NCHW_FAKE";
}

inline bool IsSupportedLayout(std::string_view layout)
{
    return layout == "NHWC" || IsPlanarLayout(layout) || IsFakePlanarLayout(layout);
}

inline bool ValidateLayout(nvbench::state &state, std::string_view layout, benchutils::InputKind inputKind)
{
    if (!IsSupportedLayout(layout))
    {
        state.skip("AdvCvtColor benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return false;
    }
    if ((IsPlanarLayout(layout) || IsFakePlanarLayout(layout)) && inputKind != benchutils::InputKind::Tensor)
    {
        state.skip("Planar AdvCvtColor benchmark is tensor-only");
        return false;
    }
    return true;
}

inline ConversionTensorShape GetConversionTensorShape(ConversionShape conversionShape, int3 shape, int rgbChannels)
{
    ConversionTensorShape tensorShape{shape.y, shape.y, rgbChannels, rgbChannels};

    switch (conversionShape)
    {
    case ConversionShape::kInterleaved444:
        if (rgbChannels != 3)
        {
            throw std::invalid_argument("Interleaved 444 conversion requires uchar3");
        }
        tensorShape.srcC = 3;
        tensorShape.dstC = 3;
        break;
    case ConversionShape::kRgbToNv:
        if (rgbChannels != 3 && rgbChannels != 4)
        {
            throw std::invalid_argument("RGB/BGR to NV conversion requires uchar3 or uchar4");
        }
        if (shape.y % 2 != 0 || shape.z % 2 != 0)
        {
            throw std::invalid_argument("NV conversion requires even height and width");
        }
        tensorShape.dstH = shape.y * 3 / 2;
        tensorShape.dstC = 1;
        break;
    case ConversionShape::kNvToRgb:
        if (rgbChannels != 3 && rgbChannels != 4)
        {
            throw std::invalid_argument("NV to RGB/BGR conversion requires uchar3 or uchar4");
        }
        if (shape.y % 2 != 0 || shape.z % 2 != 0)
        {
            throw std::invalid_argument("NV conversion requires even height and width");
        }
        tensorShape.srcH = shape.y * 3 / 2;
        tensorShape.srcC = 1;
        break;
    }

    return tensorShape;
}

template<typename T>
inline void advcvtcolor(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));

    using BT = typename nvcv::cuda::BaseType<T>;

    int rgbChannels = nvcv::cuda::NumElements<T>;

    if (!ValidateLayout(state, layout, inputKind))
    {
        return;
    }
    const bool isPlanar     = IsPlanarLayout(layout);
    const bool isFakePlanar = IsFakePlanarLayout(layout);

    ConversionInfo          conversion = GetConversionInfo(state.get_string("code"));
    NVCVColorConversionCode code       = conversion.code;
    nvcv::ColorSpec         colorSpec{NVCV_COLOR_SPEC_BT2020};

    ConversionTensorShape tensorShape = GetConversionTensorShape(conversion.shape, shape, rgbChannels);

    int64_t srcBytes = shape.x * tensorShape.srcH * shape.z * tensorShape.srcC * sizeof(BT);
    int64_t dstBytes = shape.x * tensorShape.dstH * shape.z * tensorShape.dstC * sizeof(BT);
    if (isFakePlanar)
    {
        state.add_global_memory_reads(2 * srcBytes + dstBytes);
        state.add_global_memory_writes(srcBytes + 2 * dstBytes);
    }
    else
    {
        state.add_global_memory_reads(srcBytes);
        state.add_global_memory_writes(dstBytes);
    }

    cvcuda::AdvCvtColor op;
    auto                dtype = benchutils::GetDataType<BT>();

    auto makeTensor = [dtype, &shape](int height, int channels, std::string_view tensorLayout)
    {
        if (tensorLayout == "NCHW")
        {
            return nvcv::Tensor(
                {
                    {shape.x, channels, height, shape.z},
                    "NCHW"
            },
                dtype);
        }
        return nvcv::Tensor(
            {
                {shape.x, height, shape.z, channels},
                "NHWC"
        },
            dtype);
    };

    // clang-format off

    if (isFakePlanar) // tensor-only: planar→interleaved→advcvtcolor→interleaved→planar
    {
        nvcv::Tensor src      = makeTensor(tensorShape.srcH, tensorShape.srcC, "NCHW");
        nvcv::Tensor interSrc = makeTensor(tensorShape.srcH, tensorShape.srcC, "NHWC");
        nvcv::Tensor interDst = makeTensor(tensorShape.dstH, tensorShape.dstC, "NHWC");
        nvcv::Tensor dst      = makeTensor(tensorShape.dstH, tensorShape.dstC, "NCHW");

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        cvcuda::Reformat reformatOp;

        benchutils::warmup_and_exec(state, BENCH_ADVCVTCOLOR_WARMUP_ITERATIONS,
            [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &code, &colorSpec](cudaStream_t s) {
                reformatOp(s, src, interSrc);
                op(s, interSrc, interDst, code, colorSpec);
                reformatOp(s, interDst, dst);
            });
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = makeTensor(tensorShape.srcH, tensorShape.srcC, isPlanar ? "NCHW" : "NHWC");
        nvcv::Tensor dst = makeTensor(tensorShape.dstH, tensorShape.dstC, isPlanar ? "NCHW" : "NHWC");

        benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

        benchutils::warmup_and_exec(state, BENCH_ADVCVTCOLOR_WARMUP_ITERATIONS,
            [&op, &src, &dst, &code, &colorSpec](cudaStream_t s) { op(s, src, dst, code, colorSpec); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(advcvtcolor, NVBENCH_TYPE_AXES(BENCH_ADVCVTCOLOR_TYPES))
BENCH_ADVCVTCOLOR_AXES;
