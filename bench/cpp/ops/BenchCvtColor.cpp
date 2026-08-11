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
    if (format == NVCV_IMAGE_FORMAT_RGBA8)
    {
        return nvcv::FMT_RGBA8p;
    }

    throw std::invalid_argument("Planar CvtColor var-shape benchmark supports only RGB8p/RGBA8p formats");
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

void FillPlanarPlaneData(std::vector<uint8_t> &planeData, int imageIndex, int planeIndex, bool checker)
{
    if (!checker)
    {
        return;
    }

    for (size_t idx = 0; idx < planeData.size(); ++idx)
    {
        planeData[idx] = static_cast<uint8_t>(((imageIndex + planeIndex + idx) & 1) ? 255 : 0);
    }
}

void FillPlanarImage(nvcv::Image &image, int imageIndex, int channels, bool checker)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    CVCUDA_CHECK_DATA(data);

    for (int p = 0; p < channels; ++p)
    {
        const auto          &plane = data->plane(p);
        std::vector<uint8_t> planeData(static_cast<size_t>(plane.rowStride) * plane.height);
        FillPlanarPlaneData(planeData, imageIndex, p, checker);
        CUDA_CHECK_ERROR(cudaMemcpy2D(plane.basePtr, plane.rowStride, planeData.data(), plane.rowStride,
                                      plane.rowStride, plane.height, cudaMemcpyHostToDevice));
    }
}

void FillPlanarImageBatch(nvcv::ImageBatchVarShape &batch, int3 shape, nvcv::ImageFormat format, bool checker)
{
    const int channels = NumFormatChannels(format);
    for (int i = 0; i < shape.x; ++i)
    {
        nvcv::Image image(nvcv::Size2D{shape.z, shape.y}, format);
        FillPlanarImage(image, i, channels, checker);

        batch.pushBack(image);
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

inline static void FillInterleavedImageBatch(nvcv::ImageBatchVarShape &src, nvcv::ImageBatchVarShape &dst,
                                             std::vector<nvcv::Image> &imgSrc, std::vector<nvcv::Image> &imgDst,
                                             std::vector<std::vector<uint8_t>> &srcVec, int3 shape,
                                             nvcv::ImageFormat inFormat, nvcv::ImageFormat outFormat)
{
    for (int i = 0; i < shape.x; i++)
    {
        imgSrc.emplace_back(nvcv::Size2D{shape.z, shape.y}, inFormat);
        imgDst.emplace_back(nvcv::Size2D{shape.z, shape.y}, outFormat);

        int srcRowStride = imgSrc[i].size().w * inFormat.planePixelStrideBytes(0);
        int srcBufSize   = imgSrc[i].size().h * srcRowStride;
        srcVec[i].resize(srcBufSize);
        for (int idx = 0; idx < srcBufSize; idx++)
        {
            srcVec[i][idx] = static_cast<uint8_t>(((i + idx) & 1) ? 255 : 0);
        }

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        CUDA_CHECK_ERROR(cudaMemcpy2D(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                      srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice));
    }
    src.pushBack(imgSrc.begin(), imgSrc.end());
    dst.pushBack(imgDst.begin(), imgDst.end());
}

inline static void RunVarShapeBenchmark(nvbench::state &state, cvcuda::CvtColor &op, int3 shape,
                                        nvcv::ImageFormat inFormat, nvcv::ImageFormat outFormat, CvtColorLayout layout,
                                        NVCVColorConversionCode code)
{
    if (HasSubsampledFormat(inFormat, outFormat))
    {
        state.skip("Skipping formats that have subsampled planes for the varshape benchmark");
        return;
    }
    // Also skip YUV8 (planar in Python as YUV8p, not supported with ImageBatchVarShape)
    if (inFormat == NVCV_IMAGE_FORMAT_YUV8 || outFormat == NVCV_IMAGE_FORMAT_YUV8)
    {
        state.skip("Skipping YUV8 format for varshape benchmark (planar format limitation)");
        return;
    }

    std::vector<nvcv::Image>          imgSrc;
    std::vector<nvcv::Image>          imgDst;
    nvcv::ImageBatchVarShape          src(shape.x);
    nvcv::ImageBatchVarShape          dst(shape.x);
    std::vector<std::vector<uint8_t>> srcVec(shape.x);

    // Per-byte checkerboard fill (varied without paying random-distribution cost).
    if (IsPlanar(layout))
    {
        FillPlanarImageBatch(src, shape, PlanarVarShapeFormat(inFormat), true);
        FillPlanarImageBatch(dst, shape, PlanarVarShapeFormat(outFormat), false);
    }
    else
    {
        FillInterleavedImageBatch(src, dst, imgSrc, imgDst, srcVec, shape, inFormat, outFormat);
    }

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

    nvcv::ImageFormat inFormat{inFormatValue};
    nvcv::ImageFormat outFormat{outFormatValue};
    CvtColorLayout    layout = GetCvtColorLayout(layoutStr, inputKind);

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
        RunVarShapeBenchmark(state, op, shape, inFormat, outFormat, layout, code);
    }
}

CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(cvtcolor, NVBENCH_TYPE_AXES(BENCH_CVTCOLOR_TYPES))
BENCH_CVTCOLOR_AXES;
