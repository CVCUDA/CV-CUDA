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
#include "ops/generated/BenchCvtColorConfig.hpp"

#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <map>
#include <stdexcept>
#include <string_view>
#include <tuple>

using ConvCodeToFormat = std::tuple<NVCVColorConversionCode, NVCVImageFormat, NVCVImageFormat>;
using CodeMap          = std::map<std::string, ConvCodeToFormat, std::less<>>;

template<typename BT>
inline float bytesPerPixel(NVCVImageFormat imgFormat);

namespace {

enum class CvtColorLayout
{
    NHWC,
    NCHW,
    NCHWFake
};

CvtColorLayout GetCvtColorLayout(std::string_view layout, benchutils::InputKind inputKind)
{
    if (layout == "NHWC")
    {
        return CvtColorLayout::NHWC;
    }
    if (layout == "NCHW")
    {
        return CvtColorLayout::NCHW;
    }
    if (layout == "NCHW_FAKE")
    {
        if (inputKind == benchutils::InputKind::VarShape)
        {
            throw std::invalid_argument("Fake-planar (NCHW_FAKE) CvtColor benchmark is tensor-only");
        }
        return CvtColorLayout::NCHWFake;
    }

    throw std::invalid_argument("CvtColor benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
}

bool IsPlanar(CvtColorLayout layout)
{
    return layout == CvtColorLayout::NCHW;
}

bool IsFakePlanar(CvtColorLayout layout)
{
    return layout == CvtColorLayout::NCHWFake;
}

int NumFormatChannels(nvcv::ImageFormat format)
{
    return format.numPlanes() == 1 ? format.planeNumChannels(0) : format.numPlanes();
}

nvcv::ImageFormat PlanarVarShapeFormat(nvcv::ImageFormat format)
{
    if (format == NVCV_IMAGE_FORMAT_RGB8)
    {
        return nvcv::FMT_RGB8p;
    }
    if (format == NVCV_IMAGE_FORMAT_BGR8)
    {
        return nvcv::FMT_BGR8p;
    }
    if (format == NVCV_IMAGE_FORMAT_RGBA8)
    {
        return nvcv::FMT_RGBA8p;
    }
    if (format == NVCV_IMAGE_FORMAT_RGBf16)
    {
        return nvcv::FMT_RGBf16p;
    }
    if (format == NVCV_IMAGE_FORMAT_BGRf16)
    {
        return nvcv::FMT_BGRf16p;
    }
    if (format == NVCV_IMAGE_FORMAT_RGBAf16)
    {
        return nvcv::FMT_RGBAf16p;
    }
    if (format == NVCV_IMAGE_FORMAT_LAB8)
    {
        return nvcv::FMT_LAB8p;
    }
    if (format == NVCV_IMAGE_FORMAT_LABf16)
    {
        return nvcv::FMT_LABf16p;
    }
    if (format == NVCV_IMAGE_FORMAT_RGBf32)
    {
        return nvcv::FMT_RGBf32p;
    }
    if (format == NVCV_IMAGE_FORMAT_BGRf32)
    {
        return nvcv::FMT_BGRf32p;
    }
    if (format == NVCV_IMAGE_FORMAT_RGBAf32)
    {
        return nvcv::FMT_RGBAf32p;
    }
    if (format == NVCV_IMAGE_FORMAT_LABf32)
    {
        return nvcv::FMT_LABf32p;
    }

    throw std::invalid_argument("Planar CvtColor var-shape benchmark supports only RGB/BGR/RGBA/Lab formats");
}

// HSV/YUV/Y f16 formats are not predefined by NVCV; build them like the operator tests do.
#define BENCH_CVTCOLOR_FORMAT_HSVf16 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(HSV, UNDEFINED, PL, FLOAT, XYZ0, ASSOCIATED, X16_Y16_Z16)
#define BENCH_CVTCOLOR_FORMAT_YUVf16 NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, FLOAT, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define BENCH_CVTCOLOR_FORMAT_Yf16   NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, FLOAT, X000, ASSOCIATED, X16)

// Map the 8-bit formats keyed by the code map to the equivalent format for the benched base
// type. The floating-point configs bench the same conversion codes on dtype-matched images.
// The subsampled NV12 formats stay 8-bit only (no floating-point equivalent).
template<typename BaseT>
NVCVImageFormat FormatForBaseType(NVCVImageFormat format)
{
    if constexpr (std::is_same_v<BaseT, __half>)
    {
        switch (format)
        {
        case NVCV_IMAGE_FORMAT_RGB8:
            return NVCV_IMAGE_FORMAT_RGBf16;
        case NVCV_IMAGE_FORMAT_BGR8:
            return NVCV_IMAGE_FORMAT_BGRf16;
        case NVCV_IMAGE_FORMAT_RGBA8:
            return NVCV_IMAGE_FORMAT_RGBAf16;
        case NVCV_IMAGE_FORMAT_HSV8:
            return BENCH_CVTCOLOR_FORMAT_HSVf16;
        case NVCV_IMAGE_FORMAT_LAB8:
            return NVCV_IMAGE_FORMAT_LABf16;
        case NVCV_IMAGE_FORMAT_YUV8:
            return BENCH_CVTCOLOR_FORMAT_YUVf16;
        case NVCV_IMAGE_FORMAT_Y8:
            return BENCH_CVTCOLOR_FORMAT_Yf16;
        default:
            throw std::invalid_argument("No F16 equivalent format (subsampled YUV formats are 8-bit only)");
        }
    }
    else if constexpr (std::is_same_v<BaseT, float>)
    {
        switch (format)
        {
        case NVCV_IMAGE_FORMAT_RGB8:
            return NVCV_IMAGE_FORMAT_RGBf32;
        case NVCV_IMAGE_FORMAT_BGR8:
            return NVCV_IMAGE_FORMAT_BGRf32;
        case NVCV_IMAGE_FORMAT_RGBA8:
            return NVCV_IMAGE_FORMAT_RGBAf32;
        case NVCV_IMAGE_FORMAT_LAB8:
            return NVCV_IMAGE_FORMAT_LABf32;
        default:
            throw std::invalid_argument("No F32 equivalent format (this conversion remains 8-bit/F16 only)");
        }
    }
    return format;
}

template<typename BT>
void AddCvtColorMemoryTraffic(nvbench::state &state, size_t pixelCount, NVCVImageFormat inFormat,
                              NVCVImageFormat outFormat, CvtColorLayout layout)
{
    const auto srcBytes = static_cast<size_t>(static_cast<double>(pixelCount) * bytesPerPixel<BT>(inFormat));
    const auto dstBytes = static_cast<size_t>(static_cast<double>(pixelCount) * bytesPerPixel<BT>(outFormat));

    if (IsFakePlanar(layout))
    {
        state.add_global_memory_reads(2 * srcBytes + dstBytes);
        state.add_global_memory_writes(srcBytes + 2 * dstBytes);
    }
    else
    {
        state.add_global_memory_reads(srcBytes);
        state.add_global_memory_writes(dstBytes);
    }
}

template<typename BT>
nvcv::Tensor CreatePlanarTensor(int numImages, int imgWidth, int imgHeight, nvcv::ImageFormat format)
{
    return nvcv::Tensor(
        {
            {numImages, NumFormatChannels(format), imgHeight, imgWidth},
            "NCHW"
    },
        benchutils::GetDataType<BT>());
}

template<typename BaseT, int Channels>
void FillCvtColorImageBatch(nvcv::ImageBatchVarShape &batch, int3 shape, nvcv::ImageFormat format, bool planar,
                            bool checker)
{
    using PixelT = std::conditional_t<Channels == 1, BaseT, nvcv::cuda::MakeType<BaseT, Channels>>;

    if (planar)
    {
        benchutils::FillPlanarImageBatch<PixelT>(batch, long2{shape.z, shape.y}, long2{0, 0}, checker);
        return;
    }

    const BaseT hi = checker ? benchutils::DefaultRangeMax<BaseT>() : static_cast<BaseT>(0);
    benchutils::FillImageBatch<PixelT>(batch, long2{shape.z, shape.y}, long2{0, 0},
                                       benchutils::CheckerboardValues<PixelT>(hi), format);
}

template<typename BaseT>
void FillCvtColorImageBatch(nvcv::ImageBatchVarShape &batch, int3 shape, nvcv::ImageFormat format, bool planar,
                            bool checker)
{
    switch (NumFormatChannels(format))
    {
    case 1:
        return FillCvtColorImageBatch<BaseT, 1>(batch, shape, format, planar, checker);
    case 3:
        return FillCvtColorImageBatch<BaseT, 3>(batch, shape, format, planar, checker);
    case 4:
        return FillCvtColorImageBatch<BaseT, 4>(batch, shape, format, planar, checker);
    default:
        throw std::invalid_argument("Unsupported CvtColor image channel count");
    }
}

} // namespace

inline static ConvCodeToFormat str2Frmt(const std::string &str)
{
    // clang-format off
    static const CodeMap codeMap {
        {     "RGB2BGR", {NVCV_COLOR_RGB2BGR,      NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_BGR8 }},
        {    "RGB2RGBA", {NVCV_COLOR_RGB2RGBA,     NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_RGBA8}},
        {    "RGBA2RGB", {NVCV_COLOR_RGBA2RGB,     NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_RGB8 }},
        {    "RGB2GRAY", {NVCV_COLOR_RGB2GRAY,     NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_Y8   }},
        {    "GRAY2RGB", {NVCV_COLOR_GRAY2RGB,     NVCV_IMAGE_FORMAT_Y8,    NVCV_IMAGE_FORMAT_RGB8 }},
        {     "RGB2HSV", {NVCV_COLOR_RGB2HSV,      NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_HSV8 }},
        {     "HSV2RGB", {NVCV_COLOR_HSV2RGB,      NVCV_IMAGE_FORMAT_HSV8,  NVCV_IMAGE_FORMAT_RGB8 }},
        {     "BGR2Lab", {NVCV_COLOR_BGR2Lab,      NVCV_IMAGE_FORMAT_BGR8,  NVCV_IMAGE_FORMAT_LAB8 }},
        {     "RGB2Lab", {NVCV_COLOR_RGB2Lab,      NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_LAB8 }},
        {     "Lab2BGR", {NVCV_COLOR_Lab2BGR,      NVCV_IMAGE_FORMAT_LAB8,  NVCV_IMAGE_FORMAT_BGR8 }},
        {     "Lab2RGB", {NVCV_COLOR_Lab2RGB,      NVCV_IMAGE_FORMAT_LAB8,  NVCV_IMAGE_FORMAT_RGB8 }},
        {    "LBGR2Lab", {NVCV_COLOR_LBGR2Lab,     NVCV_IMAGE_FORMAT_BGR8,  NVCV_IMAGE_FORMAT_LAB8 }},
        {    "LRGB2Lab", {NVCV_COLOR_LRGB2Lab,     NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_LAB8 }},
        {    "Lab2LBGR", {NVCV_COLOR_Lab2LBGR,     NVCV_IMAGE_FORMAT_LAB8,  NVCV_IMAGE_FORMAT_BGR8 }},
        {    "Lab2LRGB", {NVCV_COLOR_Lab2LRGB,     NVCV_IMAGE_FORMAT_LAB8,  NVCV_IMAGE_FORMAT_RGB8 }},
        {     "RGB2YUV", {NVCV_COLOR_RGB2YUV,      NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_YUV8 }},
        {     "YUV2RGB", {NVCV_COLOR_YUV2RGB,      NVCV_IMAGE_FORMAT_YUV8,  NVCV_IMAGE_FORMAT_RGB8 }},
        {"RGB2YUV_NV12", {NVCV_COLOR_RGB2YUV_NV12, NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_NV12 }},
        {"YUV2RGB_NV12", {NVCV_COLOR_YUV2RGB_NV12, NVCV_IMAGE_FORMAT_NV12,  NVCV_IMAGE_FORMAT_RGB8 }},
    };
    // clang-format on

    if (auto it = codeMap.find(str); it != codeMap.end())
    {
        return it->second;
    }
    else
    {
        throw std::invalid_argument("Unrecognized color code");
    }
}

template<typename BT>
inline float bytesPerPixel(NVCVImageFormat imgFormat)
{
#define BPP_CASE(frmt, bytes) \
    case frmt:                \
        return bytes * sizeof(BT)

    switch (imgFormat)
    {
        BPP_CASE(NVCV_IMAGE_FORMAT_RGB8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_BGR8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_HSV8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_LAB8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_RGBA8, 4);
        BPP_CASE(NVCV_IMAGE_FORMAT_YUV8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_NV12, 1.5f);
        BPP_CASE(NVCV_IMAGE_FORMAT_Y8, 1);
    default:
        throw std::invalid_argument("Unrecognized format");
    }
#undef BPP_CASE
}

// Adapted from src/util/TensorDataUtils.hpp
inline static nvcv::Tensor CreateTensor(int numImages, int imgWidth, int imgHeight, const nvcv::ImageFormat &imgFormat)
{
    if (imgFormat == NVCV_IMAGE_FORMAT_NV12 || imgFormat == NVCV_IMAGE_FORMAT_NV12_ER
        || imgFormat == NVCV_IMAGE_FORMAT_NV21 || imgFormat == NVCV_IMAGE_FORMAT_NV21_ER)
    {
        if (imgHeight % 2 != 0 || imgWidth % 2 != 0)
        {
            throw std::invalid_argument("Invalid height");
        }

        int height420 = (imgHeight * 3) / 2;

        return nvcv::Tensor(numImages, {imgWidth, height420}, nvcv::ImageFormat(NVCV_IMAGE_FORMAT_Y8));
    }
    else
    {
        return nvcv::Tensor(numImages, {imgWidth, imgHeight}, imgFormat);
    }
}

inline static bool IsSubsampledFormat(nvcv::ImageFormat format)
{
    const auto chromaSubsampling = format.chromaSubsampling();
    return chromaSubsampling != nvcv::ChromaSubsampling::NONE && chromaSubsampling != nvcv::ChromaSubsampling::CSS_444;
}

inline static bool HasSubsampledFormat(nvcv::ImageFormat inFormat, nvcv::ImageFormat outFormat)
{
    return IsSubsampledFormat(inFormat) || IsSubsampledFormat(outFormat);
}

template<typename BaseT>
void RunFakePlanarBenchmark(nvbench::state &state, cvcuda::CvtColor &op, int3 shape, nvcv::ImageFormat inFormat,
                            nvcv::ImageFormat outFormat, NVCVColorConversionCode code)
{
    nvcv::Tensor src      = CreatePlanarTensor<BaseT>(shape.x, shape.z, shape.y, inFormat);
    nvcv::Tensor interSrc = CreateTensor(shape.x, shape.z, shape.y, inFormat);
    nvcv::Tensor interDst = CreateTensor(shape.x, shape.z, shape.y, outFormat);
    nvcv::Tensor dst      = CreatePlanarTensor<BaseT>(shape.x, shape.z, shape.y, outFormat);

    benchutils::FillTensor<BaseT>(src, benchutils::CheckerboardValues<BaseT>());

    cvcuda::Reformat reformatOp;

    benchutils::warmup_and_exec(state, BENCH_CVTCOLOR_WARMUP_ITERATIONS,
                                [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &code](cudaStream_t s)
                                {
                                    reformatOp(s, src, interSrc);
                                    op(s, interSrc, interDst, code);
                                    reformatOp(s, interDst, dst);
                                });
}

template<typename BaseT>
void RunTensorBenchmark(nvbench::state &state, cvcuda::CvtColor &op, int3 shape, nvcv::ImageFormat inFormat,
                        nvcv::ImageFormat outFormat, CvtColorLayout layout, NVCVColorConversionCode code)
{
    nvcv::Tensor src = IsPlanar(layout) ? CreatePlanarTensor<BaseT>(shape.x, shape.z, shape.y, inFormat)
                                        : CreateTensor(shape.x, shape.z, shape.y, inFormat);
    nvcv::Tensor dst = IsPlanar(layout) ? CreatePlanarTensor<BaseT>(shape.x, shape.z, shape.y, outFormat)
                                        : CreateTensor(shape.x, shape.z, shape.y, outFormat);

    benchutils::FillTensor<BaseT>(src, benchutils::CheckerboardValues<BaseT>());

    benchutils::warmup_and_exec(state, BENCH_CVTCOLOR_WARMUP_ITERATIONS,
                                [&op, &src, &dst, &code](cudaStream_t s) { op(s, src, dst, code); });
}

template<typename BaseT>
inline static void RunVarShapeBenchmark(nvbench::state &state, cvcuda::CvtColor &op, int3 shape,
                                        nvcv::ImageFormat inFormat, nvcv::ImageFormat outFormat, CvtColorLayout layout,
                                        NVCVColorConversionCode code)
{
    nvcv::ImageBatchVarShape src(shape.x);
    nvcv::ImageBatchVarShape dst(shape.x);

    if (IsPlanar(layout))
    {
        inFormat  = PlanarVarShapeFormat(inFormat);
        outFormat = PlanarVarShapeFormat(outFormat);
    }
    FillCvtColorImageBatch<BaseT>(src, shape, inFormat, IsPlanar(layout), true);
    FillCvtColorImageBatch<BaseT>(dst, shape, outFormat, IsPlanar(layout), false);

    benchutils::warmup_and_exec(state, BENCH_CVTCOLOR_WARMUP_ITERATIONS,
                                [&op, &src, &dst, &code](cudaStream_t s) { op(s, src, dst, code); });
}

template<typename T>
inline void cvtcolor(nvbench::state &state, nvbench::type_list<T>)
try
{
    using BT = typename nvcv::cuda::BaseType<T>;

    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    auto                        layoutStr = benchutils::GetStringParam(state, "layout", "NHWC");
    using BaseT                           = nvcv::cuda::BaseType<BT>;

    auto [code, inFormatValue, outFormatValue] = str2Frmt(state.get_string("code"));

    CvtColorLayout layout = GetCvtColorLayout(layoutStr, inputKind);
    if (inputKind == benchutils::InputKind::VarShape)
    {
        if (HasSubsampledFormat(nvcv::ImageFormat{inFormatValue}, nvcv::ImageFormat{outFormatValue}))
        {
            state.skip("Skipping formats that have subsampled planes for the varshape benchmark");
            return;
        }
        if (inFormatValue == NVCV_IMAGE_FORMAT_YUV8 || outFormatValue == NVCV_IMAGE_FORMAT_YUV8)
        {
            state.skip("Skipping YUV8 format for varshape benchmark (planar format limitation)");
            return;
        }
    }

    // Tensors and images use the dtype-matched formats; the memory-traffic accounting below
    // keeps the 8-bit map keys (it scales channel counts by sizeof(BaseT) itself).
    nvcv::ImageFormat inFormat{FormatForBaseType<BaseT>(inFormatValue)};
    nvcv::ImageFormat outFormat{FormatForBaseType<BaseT>(outFormatValue)};

    if ((IsPlanar(layout) || IsFakePlanar(layout)) && HasSubsampledFormat(inFormat, outFormat))
    {
        state.skip("Skipping subsampled YUV CvtColor formats for planar benchmarks");
        return;
    }

    const size_t pixelCount
        = static_cast<size_t>(shape.x) * static_cast<size_t>(shape.y) * static_cast<size_t>(shape.z);
    AddCvtColorMemoryTraffic<BaseT>(state, pixelCount, inFormatValue, outFormatValue, layout);

    cvcuda::CvtColor op;

    if (IsFakePlanar(layout))
    {
        RunFakePlanarBenchmark<BaseT>(state, op, shape, inFormat, outFormat, code);
    }
    else if (inputKind == benchutils::InputKind::Tensor)
    {
        RunTensorBenchmark<BaseT>(state, op, shape, inFormat, outFormat, layout, code);
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        RunVarShapeBenchmark<BaseT>(state, op, shape, inFormat, outFormat, layout, code);
    }
}

CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(cvtcolor, NVBENCH_TYPE_AXES(BENCH_CVTCOLOR_TYPES))
BENCH_CVTCOLOR_AXES;
