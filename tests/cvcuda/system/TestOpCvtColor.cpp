/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvtColorUtils.hpp"
#include "Definitions.hpp"
#include "PlanarParityUtils.hpp"
#include "TestUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstddef>
#include <cstring>

namespace test = nvcv::test;
namespace util = nvcv::util;
namespace cuda = nvcv::cuda;

using std::vector;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

#define NVCV_IMAGE_FORMAT_RGBS8  NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X8_Y8_Z8)
#define NVCV_IMAGE_FORMAT_BGRS8  NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, ZYX1, ASSOCIATED, X8_Y8_Z8)
#define NVCV_IMAGE_FORMAT_RGBAS8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8)
#define NVCV_IMAGE_FORMAT_BGRAS8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, ZYXW, ASSOCIATED, X8_Y8_Z8_W8)
#define NVCV_IMAGE_FORMAT_YS8    NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, SIGNED, X000, ASSOCIATED, X8)
#define NVCV_IMAGE_FORMAT_YS8_ER NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, NONE, PL, UNSIGNED, X000, ASSOCIATED, X8)

#define NVCV_IMAGE_FORMAT_BGR16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, ZYX1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGB16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_BGRA16 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, ZYXW, ASSOCIATED, X16_Y16_Z16_W16)
#define NVCV_IMAGE_FORMAT_RGBA16 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)
#define NVCV_IMAGE_FORMAT_YUV16 NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)

#define NVCV_IMAGE_FORMAT_YS16   NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, SIGNED, X000, ASSOCIATED, X16)
#define NVCV_IMAGE_FORMAT_BGRS16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, ZYX1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGBS16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_BGRAS16 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, ZYXW, ASSOCIATED, X16_Y16_Z16_W16)
#define NVCV_IMAGE_FORMAT_RGBAS16 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)

#define NVCV_IMAGE_FORMAT_YUVf16 NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, FLOAT, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_Yf16   NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, FLOAT, X000, ASSOCIATED, X16)

#define NVCV_IMAGE_FORMAT_YS32   NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, SIGNED, X000, ASSOCIATED, X32)
#define NVCV_IMAGE_FORMAT_BGRS32 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, ZYX1, ASSOCIATED, X32_Y32_Z32)
#define NVCV_IMAGE_FORMAT_RGBS32 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X32_Y32_Z32)
#define NVCV_IMAGE_FORMAT_BGRAS32 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, ZYXW, ASSOCIATED, X32_Y32_Z32_W32)
#define NVCV_IMAGE_FORMAT_RGBAS32 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZW, ASSOCIATED, X32_Y32_Z32_W32)

#define NVCV_IMAGE_FORMAT_YUVf32 NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, FLOAT, XYZ1, ASSOCIATED, X32_Y32_Z32)
#define NVCV_IMAGE_FORMAT_Yf32   NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, FLOAT, X000, ASSOCIATED, X32)
#define NVCV_IMAGE_FORMAT_HSVf32 NVCV_DETAIL_MAKE_COLOR_FMT1(HSV, UNDEFINED, PL, FLOAT, XYZ0, ASSOCIATED, X32_Y32_Z32)

#define NVCV_IMAGE_FORMAT_BGRf64 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, ZYX1, ASSOCIATED, X64_Y64_Z64)
#define NVCV_IMAGE_FORMAT_RGBf64 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZ1, ASSOCIATED, X64_Y64_Z64)
#define NVCV_IMAGE_FORMAT_BGRAf64 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, ZYXW, ASSOCIATED, X64_Y64_Z64_W64)
#define NVCV_IMAGE_FORMAT_RGBAf64 \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZW, ASSOCIATED, X64_Y64_Z64_W64)
#define NVCV_IMAGE_FORMAT_HSVf64 NVCV_DETAIL_MAKE_COLOR_FMT1(HSV, UNDEFINED, PL, FLOAT, XYZ0, ASSOCIATED, X64_Y64_Z64)
#define NVCV_IMAGE_FORMAT_Yf64   NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, FLOAT, X000, ASSOCIATED, X64)

// clang-format off

//--------------------------------------------------------------------------------------------------------------------//
static bool IsInterleavedYuv422(nvcv::ImageFormat fmt)
{
    return fmt == NVCV_IMAGE_FORMAT_UYVY || fmt == NVCV_IMAGE_FORMAT_UYVY_ER || fmt == NVCV_IMAGE_FORMAT_YUYV
        || fmt == NVCV_IMAGE_FORMAT_YUYV_ER;
}

static bool IsSemiPlanarYuv420(nvcv::ImageFormat fmt)
{
    return fmt == NVCV_IMAGE_FORMAT_NV12 || fmt == NVCV_IMAGE_FORMAT_NV12_ER || fmt == NVCV_IMAGE_FORMAT_NV21
        || fmt == NVCV_IMAGE_FORMAT_NV21_ER;
}

static bool IsBGR(nvcv::Swizzle swizzle)
{
    return swizzle == nvcv::Swizzle::S_ZYXW || swizzle == nvcv::Swizzle::S_ZYX1
        || swizzle == nvcv::Swizzle::S_ZYX0;
}

static bool IsHsvToRgb(NVCVColorConversionCode code)
{
    return code == NVCV_COLOR_HSV2BGR || code == NVCV_COLOR_HSV2BGR_FULL || code == NVCV_COLOR_HSV2RGB
        || code == NVCV_COLOR_HSV2RGB_FULL;
}

static bool IsFullHsvToRgb(NVCVColorConversionCode code)
{
    return code == NVCV_COLOR_HSV2BGR_FULL || code == NVCV_COLOR_HSV2RGB_FULL;
}

template<typename T, bool full>
static void GenerateHsvSource(vector<T> &srcVec, int srcWdth, int srcHght, int imgs, size_t numPixels, RandEng &randEng)
{
    constexpr size_t minCntAllHSV = 90 * 256 * 256; // Minimum # of pixels to call generateAllHSV.
    constexpr double minMultHSV   = -0.5;           // Set hue range multiplier to be outside normal range
    constexpr double maxMultHSV   = 1.5;            // to test robustness to wrapped hue values.

    if (numPixels >= minCntAllHSV)
    {
        generateAllHSV<T, full>(srcVec, srcWdth, srcHght, imgs);
        return;
    }

    generateRandHSV<T, full>(srcVec, randEng, minMultHSV, maxMultHSV);
}

template<typename T>
static void PopulateSource(vector<T> &srcVec, int srcWdth, int srcHght, int imgs, int srcChannels, size_t numPixels,
                           bool srcRGBA, bool srcBGR, NVCVColorConversionCode code, RandEng &randEng)
{
    constexpr size_t minCntAllRGB = 128 * 256 * 256; // Minimum # of pixels to call generateAllRGB.

    if (srcChannels <= 2)
    {
        generateRandVec(srcVec, randEng);
        return;
    }

    if (IsHsvToRgb(code))
    {
        if (IsFullHsvToRgb(code))
        {
            GenerateHsvSource<T, true>(srcVec, srcWdth, srcHght, imgs, numPixels, randEng);
        }
        else
        {
            GenerateHsvSource<T, false>(srcVec, srcWdth, srcHght, imgs, numPixels, randEng);
        }
        return;
    }

    if (numPixels >= minCntAllRGB)
    {
        generateAllRGB(srcVec, srcWdth, srcHght, imgs, srcRGBA, srcBGR);
        return;
    }

    generateRandTestRGB(srcVec, randEng, srcRGBA, srcBGR);
}

template<typename T>
static bool BuildBasicColorReference(vector<T> &refVec, const vector<T> &srcVec, NVCVColorConversionCode code,
                                     size_t numPixels, bool srcRGBA, bool srcBGR, bool dstRGBA, bool dstBGR)
{
    switch (code) // NOSONAR: reference implementation covers each supported conversion code.
    {
    case NVCV_COLOR_BGR2BGRA:
    case NVCV_COLOR_BGRA2BGR:
        changeAlpha<T>(refVec, srcVec, numPixels, srcRGBA, dstRGBA);
        return true;

    case NVCV_COLOR_BGR2RGBA:
    case NVCV_COLOR_RGBA2BGR:
    case NVCV_COLOR_BGR2RGB:
    case NVCV_COLOR_BGRA2RGBA:
        convertRGBtoBGR<T>(refVec, srcVec, numPixels, srcRGBA, dstRGBA);
        return true;

    case NVCV_COLOR_BGR2GRAY:
    case NVCV_COLOR_RGB2GRAY:
    case NVCV_COLOR_BGRA2GRAY:
    case NVCV_COLOR_RGBA2GRAY:
        convertRGBtoGray<T>(refVec, srcVec, numPixels, srcRGBA, srcBGR);
        return true;

    case NVCV_COLOR_GRAY2BGR:
    case NVCV_COLOR_GRAY2BGRA:
        convertGrayToRGB<T>(refVec, srcVec, numPixels, dstRGBA);
        return true;

    case NVCV_COLOR_BGR2HSV:
    case NVCV_COLOR_RGB2HSV:
        convertRGBtoHSV<T, false>(refVec, srcVec, numPixels, srcRGBA, srcBGR);
        return true;

    case NVCV_COLOR_HSV2BGR:
    case NVCV_COLOR_HSV2RGB:
        convertHSVtoRGB<T, false>(refVec, srcVec, numPixels, dstRGBA, dstBGR);
        return true;

    case NVCV_COLOR_BGR2HSV_FULL:
    case NVCV_COLOR_RGB2HSV_FULL:
        convertRGBtoHSV<T, true>(refVec, srcVec, numPixels, srcRGBA, srcBGR);
        return true;

    case NVCV_COLOR_HSV2BGR_FULL:
    case NVCV_COLOR_HSV2RGB_FULL:
        convertHSVtoRGB<T, true>(refVec, srcVec, numPixels, dstRGBA, dstBGR);
        return true;

    case NVCV_COLOR_BGR2YUV:
    case NVCV_COLOR_RGB2YUV:
        convertRGBtoYUV_PAL<T>(refVec, srcVec, numPixels, srcRGBA, srcBGR);
        return true;

    case NVCV_COLOR_YUV2BGR:
    case NVCV_COLOR_YUV2RGB:
        convertYUVtoRGB_PAL<T>(refVec, srcVec, numPixels, dstRGBA, dstBGR);
        return true;

    default:
        return false;
    }
}

template<typename T>
static bool BuildYuvToColorReference(vector<T> &refVec, const vector<T> &srcVec, NVCVColorConversionCode code, int wdth,
                                     int hght, int imgs, size_t numPixels, bool dstRGBA, bool dstBGR)
{
    switch (code) // NOSONAR: reference implementation covers each supported YUV conversion code.
    {
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2RGBA_NV12:
    case NVCV_COLOR_YUV2BGRA_NV12:
        convertNV12toRGB<T>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, false);
        return true;

    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2BGR_NV21:
    case NVCV_COLOR_YUV2RGBA_NV21:
    case NVCV_COLOR_YUV2BGRA_NV21:
        convertNV12toRGB<T>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, true);
        return true;

    case NVCV_COLOR_YUV2RGB_YV12:
    case NVCV_COLOR_YUV2BGR_YV12:
    case NVCV_COLOR_YUV2RGBA_YV12:
    case NVCV_COLOR_YUV2BGRA_YV12:
        convertYUVtoRGB_420<T>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, true);
        return true;

    case NVCV_COLOR_YUV2RGB_IYUV:
    case NVCV_COLOR_YUV2BGR_IYUV:
    case NVCV_COLOR_YUV2RGBA_IYUV:
    case NVCV_COLOR_YUV2BGRA_IYUV:
        convertYUVtoRGB_420<T>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, false);
        return true;

    case NVCV_COLOR_YUV2GRAY_420:
        convertYUVtoGray_420<T>(refVec, srcVec, wdth, hght, imgs);
        return true;

    case NVCV_COLOR_YUV2RGB_UYVY:
    case NVCV_COLOR_YUV2BGR_UYVY:
    case NVCV_COLOR_YUV2RGBA_UYVY:
    case NVCV_COLOR_YUV2BGRA_UYVY:
        convertYUVtoRGB_422<T, false>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, false);
        return true;

    case NVCV_COLOR_YUV2RGB_YUY2:
    case NVCV_COLOR_YUV2BGR_YUY2:
    case NVCV_COLOR_YUV2RGBA_YUY2:
    case NVCV_COLOR_YUV2BGRA_YUY2:
        convertYUVtoRGB_422<T, true>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, false);
        return true;

    case NVCV_COLOR_YUV2RGB_YVYU:
    case NVCV_COLOR_YUV2BGR_YVYU:
    case NVCV_COLOR_YUV2RGBA_YVYU:
    case NVCV_COLOR_YUV2BGRA_YVYU:
        convertYUVtoRGB_422<T, true>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, true);
        return true;

    case NVCV_COLOR_YUV2GRAY_UYVY:
        convertYUVtoGray_422<T, false>(refVec, srcVec, numPixels);
        return true;

    case NVCV_COLOR_YUV2GRAY_YUY2:
        convertYUVtoGray_422<T, true>(refVec, srcVec, numPixels);
        return true;

    default:
        return false;
    }
}

template<typename T>
static bool BuildColorToYuvReference(vector<T> &refVec, const vector<T> &srcVec, NVCVColorConversionCode code, int wdth,
                                     int hght, int imgs, bool srcRGBA, bool srcBGR)
{
    switch (code)
    {
    case NVCV_COLOR_RGB2YUV_I420:
    case NVCV_COLOR_BGR2YUV_I420:
    case NVCV_COLOR_RGBA2YUV_I420:
    case NVCV_COLOR_BGRA2YUV_I420:
        convertRGBtoYUV_420<T>(refVec, srcVec, wdth, hght, imgs, srcRGBA, srcBGR, false);
        return true;

    case NVCV_COLOR_RGB2YUV_YV12:
    case NVCV_COLOR_BGR2YUV_YV12:
    case NVCV_COLOR_RGBA2YUV_YV12:
    case NVCV_COLOR_BGRA2YUV_YV12:
        convertRGBtoYUV_420<T>(refVec, srcVec, wdth, hght, imgs, srcRGBA, srcBGR, true);
        return true;

    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_RGBA2YUV_NV12:
    case NVCV_COLOR_BGRA2YUV_NV12:
        convertRGBtoNV12<T>(refVec, srcVec, wdth, hght, imgs, srcRGBA, srcBGR, false);
        return true;

    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_RGBA2YUV_NV21:
    case NVCV_COLOR_BGRA2YUV_NV21:
        convertRGBtoNV12<T>(refVec, srcVec, wdth, hght, imgs, srcRGBA, srcBGR, true);
        return true;

    default:
        return false;
    }
}

template<typename T>
static bool BuildCvtColorReference(vector<T> &refVec, const vector<T> &srcVec, NVCVColorConversionCode code,
                                   size_t numPixels, int wdth, int hght, int imgs, bool srcRGBA, bool srcBGR,
                                   bool dstRGBA, bool dstBGR)
{
    return BuildBasicColorReference(refVec, srcVec, code, numPixels, srcRGBA, srcBGR, dstRGBA, dstBGR)
        || BuildYuvToColorReference(refVec, srcVec, code, wdth, hght, imgs, numPixels, dstRGBA, dstBGR)
        || BuildColorToYuvReference(refVec, srcVec, code, wdth, hght, imgs, srcRGBA, srcBGR);
}

//--------------------------------------------------------------------------------------------------------------------//
template<typename T>
static void verifyOutput(nvcv::Tensor srcTensor, nvcv::ImageFormat srcFrmt,
                         nvcv::Tensor dstTensor, nvcv::ImageFormat dstFrmt,
                         NVCVColorConversionCode code, int wdth, int hght, int imgs, double maxDiff)
{
    auto srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);
    ASSERT_TRUE(dstData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    int srcChannels = srcAccess->numChannels();
    int dstChannels = dstAccess->numChannels();

    ASSERT_LE(srcChannels, 4);
    ASSERT_LE(dstChannels, 4);

    int srcWdth = wdth;
    int srcHght = hght;
    int dstWdth = wdth;
    int dstHght = hght;

    if (IsInterleavedYuv422(srcFrmt))
        srcWdth = srcWdth << 1;
    if (IsSemiPlanarYuv420(srcFrmt))
        srcHght = (srcHght * 3) >> 1;
    ASSERT_EQ(srcWdth, srcAccess->numCols());
    ASSERT_EQ(srcHght, srcAccess->numRows());

    if (IsInterleavedYuv422(dstFrmt))
        dstWdth = dstWdth << 1;
    if (IsSemiPlanarYuv420(dstFrmt))
        dstHght = (dstHght * 3) >> 1;
    ASSERT_EQ(dstWdth, dstAccess->numCols());
    ASSERT_EQ(dstHght, dstAccess->numRows());

    int srcRowElems = srcChannels * srcWdth;
    int dstRowElems = dstChannels * dstWdth;

    size_t numPixels = (size_t)imgs * (size_t)wdth * (size_t)hght;
    size_t srcElems  = (size_t)imgs * (size_t)srcWdth * (size_t)srcHght * (size_t)srcChannels;
    size_t dstElems  = (size_t)imgs * (size_t)dstWdth * (size_t)dstHght * (size_t)dstChannels;

    size_t srcPitchCPU = srcRowElems * sizeof(T);
    size_t dstPitchCPU = dstRowElems * sizeof(T);

    nvcv::Swizzle srcSwizzle = srcFrmt.swizzle();
    nvcv::Swizzle dstSwizzle = dstFrmt.swizzle();

    vector<T> srcVec(srcElems);
    vector<T> refVec(dstElems);

    bool srcBGR  = IsBGR(srcSwizzle);
    bool dstBGR  = IsBGR(dstSwizzle);
    bool srcRGBA = (srcChannels == 4);
    bool dstRGBA = (dstChannels == 4);

    RandEng randEng(0);

    PopulateSource(srcVec, srcWdth, srcHght, imgs, srcChannels, numPixels, srcRGBA, srcBGR, code, randEng);

    // Copy source from image vector to device tensor.
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->basePtr(), srcAccess->rowStride(), srcVec.data(), srcPitchCPU,
                                        srcPitchCPU, (size_t)imgs * (size_t)srcHght, cudaMemcpyHostToDevice));

    bool success = BuildCvtColorReference(refVec, srcVec, code, numPixels, wdth, hght, imgs, srcRGBA, srcBGR, dstRGBA,
                                          dstBGR);
    if (!success)
    {
        std::cerr << "**** ERROR: Color conversion not implemented for conversion code " << code << ". ****\n\n";
    }

    if (success)
    {
        // Run color conversion operator.
        cudaStream_t stream;

        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        cvcuda::CvtColor convertColor;

        EXPECT_NO_THROW(convertColor(stream, srcTensor, dstTensor, code));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

        // Copy destination tensor back to host.
        vector<T> dstVec(dstElems);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dstVec.data(), dstPitchCPU, dstData->basePtr(), dstAccess->rowStride(),
                                            dstPitchCPU, (size_t)imgs * (size_t)dstHght, cudaMemcpyDeviceToHost));

        constexpr unsigned int maxErrCnt = 16;

        // Compare "gold" reference to computed output.
        if (dstFrmt == NVCV_IMAGE_FORMAT_HSV8 || dstFrmt == NVCV_IMAGE_FORMAT_HSVf32)
        {
            const bool full = (code == NVCV_COLOR_BGR2HSV_FULL || code == NVCV_COLOR_RGB2HSV_FULL);
            double     range = 360.0;
            if constexpr (sizeof(T) == 1)
            {
                range = full ? 256.0 : 180.0;
            }

            EXPECT_NEAR_HSV_VEC_CNT(refVec, dstVec, range, maxDiff, maxErrCnt, success);
        }
        else
            EXPECT_NEAR_VEC_CNT(refVec, dstVec, maxDiff, maxErrCnt, success);
    }
    else
    {
        GTEST_SKIP() << "Waived: this test hasn't been implemented.";
    }
}

//--------------------------------------------------------------------------------------------------------------------//

#define ERR2_3 (2.0 / 1024.0) // 0.0009765625    --> approximates 2e-3 but can be exactly represented in floating point.
#define ERR1_3 (1.0 / 1024.0) // 0.0009765625    --> approximates 1e-3 but can be exactly represented in floating point.
#define ERR1_4 (1.0 / 8192.0) // 0.0001220703125 --> approximates 1e-4 but can be exactly represented in floating point.

NVCV_TEST_SUITE_P(OpCvtColor,
test::ValueList<int, int, int, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode, double>
{
    //  W,   H,  N,  Input Format,               Output Format,              Convert Code,         maxDiff
    { 177, 113,  1,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_BGR2BGRA,      0.0},
    { 113, 176,  2,  NVCV_IMAGE_FORMAT_BGRA8,    NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_BGRA2BGR,      0.0},
    { 335, 432,  2,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_RGB2RGBA,      0.0},
    { 431, 336,  2,  NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_BGR2RGBA,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_RGB2BGRA,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_BGRA8,    NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_RGBA2BGRA,     0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_BGRA8,    NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_BGRA2RGBA,     0.0},
    {  38,  52,  4,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_RGB2BGR,       0.0},
    {  52,  38,  7,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_BGR2RGB,       0.0},

    { 177, 113,  1,  NVCV_IMAGE_FORMAT_BGRS8,    NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_COLOR_BGR2BGRA,      0.0},
    { 113, 176,  2,  NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_IMAGE_FORMAT_BGRS8,    NVCV_COLOR_BGRA2BGR,      0.0},
    { 335, 432,  2,  NVCV_IMAGE_FORMAT_RGBS8,    NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_COLOR_RGB2RGBA,      0.0},
    { 431, 336,  2,  NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_IMAGE_FORMAT_RGBS8,    NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRS8,    NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_COLOR_BGR2RGBA,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_IMAGE_FORMAT_BGRS8,    NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGBS8,    NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_COLOR_RGB2BGRA,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_IMAGE_FORMAT_RGBS8,    NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_COLOR_RGBA2BGRA,     0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_COLOR_BGRA2RGBA,     0.0},

    { 177, 113,  1,  NVCV_IMAGE_FORMAT_BGR16,    NVCV_IMAGE_FORMAT_BGRA16,   NVCV_COLOR_BGR2BGRA,      0.0},
    { 113, 176,  2,  NVCV_IMAGE_FORMAT_BGRA16,   NVCV_IMAGE_FORMAT_BGR16,    NVCV_COLOR_BGRA2BGR,      0.0},
    { 335, 432,  2,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_RGBA16,   NVCV_COLOR_RGB2RGBA,      0.0},
    { 431, 336,  2,  NVCV_IMAGE_FORMAT_RGBA16,   NVCV_IMAGE_FORMAT_RGB16,    NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGR16,    NVCV_IMAGE_FORMAT_RGBA16,   NVCV_COLOR_BGR2RGBA,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_RGBA16,   NVCV_IMAGE_FORMAT_BGR16,    NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_BGRA16,   NVCV_COLOR_RGB2BGRA,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_BGRA16,   NVCV_IMAGE_FORMAT_RGB16,    NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBA16,   NVCV_IMAGE_FORMAT_BGRA16,   NVCV_COLOR_RGBA2BGRA,     0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_BGRA16,   NVCV_IMAGE_FORMAT_RGBA16,   NVCV_COLOR_BGRA2RGBA,     0.0},

    { 177, 113,  1,  NVCV_IMAGE_FORMAT_BGRS16,   NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_COLOR_BGR2BGRA,      0.0},
    { 113, 176,  2,  NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_IMAGE_FORMAT_BGRS16,   NVCV_COLOR_BGRA2BGR,      0.0},
    { 335, 432,  2,  NVCV_IMAGE_FORMAT_RGBS16,   NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_COLOR_RGB2RGBA,      0.0},
    { 431, 336,  2,  NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_IMAGE_FORMAT_RGBS16,   NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRS16,   NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_COLOR_BGR2RGBA,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_IMAGE_FORMAT_BGRS16,   NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGBS16,   NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_COLOR_RGB2BGRA,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_IMAGE_FORMAT_RGBS16,   NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_COLOR_RGBA2BGRA,     0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_COLOR_BGRA2RGBA,     0.0},
    {  38,  52,  2,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_BGR16,    NVCV_COLOR_RGB2BGR,       0.0},
    {  52,  38,  4,  NVCV_IMAGE_FORMAT_BGR16,    NVCV_IMAGE_FORMAT_RGB16,    NVCV_COLOR_BGR2RGB,       0.0},

    { 177, 113,  1,  NVCV_IMAGE_FORMAT_BGRS32,   NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_COLOR_BGR2BGRA,      0.0},
    { 113, 176,  2,  NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_IMAGE_FORMAT_BGRS32,   NVCV_COLOR_BGRA2BGR,      0.0},
    { 335, 432,  2,  NVCV_IMAGE_FORMAT_RGBS32,   NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_COLOR_RGB2RGBA,      0.0},
    { 431, 336,  2,  NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_IMAGE_FORMAT_RGBS32,   NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRS32,   NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_COLOR_BGR2RGBA,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_IMAGE_FORMAT_BGRS32,   NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGBS32,   NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_COLOR_RGB2BGRA,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_IMAGE_FORMAT_RGBS32,   NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_COLOR_RGBA2BGRA,     0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_COLOR_BGRA2RGBA,     0.0},
    {  38,  58,  2,  NVCV_IMAGE_FORMAT_RGBS32,   NVCV_IMAGE_FORMAT_BGRS32,   NVCV_COLOR_RGB2BGR,       0.0},
    {  52,  22,  4,  NVCV_IMAGE_FORMAT_BGRS32,   NVCV_IMAGE_FORMAT_RGBS32,   NVCV_COLOR_BGR2RGB,       0.0},

    // Conversions that add alpha to output tensor are not allowed for f16 type.
    { 113, 176,  2,  NVCV_IMAGE_FORMAT_BGRAf16,  NVCV_IMAGE_FORMAT_BGRf16,   NVCV_COLOR_BGRA2BGR,      0.0},
    { 431, 336,  2,  NVCV_IMAGE_FORMAT_RGBAf16,  NVCV_IMAGE_FORMAT_RGBf16,   NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_RGBAf16,  NVCV_IMAGE_FORMAT_BGRf16,   NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_BGRAf16,  NVCV_IMAGE_FORMAT_RGBf16,   NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAf16,  NVCV_IMAGE_FORMAT_BGRAf16,  NVCV_COLOR_RGBA2BGRA,     0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_BGRAf16,  NVCV_IMAGE_FORMAT_RGBAf16,  NVCV_COLOR_BGRA2RGBA,     0.0},

    { 129,  61,  4,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_BGR2BGRA,      0.0},
    { 129,  61,  4,  NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_BGRA2BGR,      0.0},
    {  63,  31,  3,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_COLOR_RGB2RGBA,      0.0},
    {  63,  31,  3,  NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_RGBA2RGB,      0.0},
    {  42, 111,  2,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_COLOR_BGR2RGBA,      0.0},
    {  42, 111,  2,  NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_RGBA2BGR,      0.0},
    {  21,  72,  2,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_RGB2BGRA,      0.0},
    {  21,  72,  2,  NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_BGRA2RGB,      0.0},
    {  23,  31,  3,  NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_RGBA2BGRA,     0.0},
    {  23,  31,  3,  NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_COLOR_BGRA2RGBA,     0.0},
    {  44,  88,  1,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_RGB2BGR,       0.0},
    {  54,  77,  3,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_BGR2RGB,       0.0},

    { 177, 113,  1,  NVCV_IMAGE_FORMAT_BGRf64,   NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_COLOR_BGR2BGRA,      0.0},
    { 113, 176,  2,  NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_IMAGE_FORMAT_BGRf64,   NVCV_COLOR_BGRA2BGR,      0.0},
    { 335, 432,  2,  NVCV_IMAGE_FORMAT_RGBf64,   NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_COLOR_RGB2RGBA,      0.0},
    { 431, 336,  2,  NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_IMAGE_FORMAT_RGBf64,   NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRf64,   NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_COLOR_BGR2RGBA,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_IMAGE_FORMAT_BGRf64,   NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGBf64,   NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_COLOR_RGB2BGRA,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_IMAGE_FORMAT_RGBf64,   NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_COLOR_RGBA2BGRA,     0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_COLOR_BGRA2RGBA,     0.0},
    {  44,  77,  1,  NVCV_IMAGE_FORMAT_RGBf64,   NVCV_IMAGE_FORMAT_BGRf64,   NVCV_COLOR_RGB2BGR,       0.0},
    {  77,  44,  3,  NVCV_IMAGE_FORMAT_BGRf64,   NVCV_IMAGE_FORMAT_RGBf64,   NVCV_COLOR_BGR2RGB,       0.0},

    {  23,  21, 63,  NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_GRAY2BGR,      0.0},
    {  21,  22, 63,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_COLOR_BGR2GRAY,      1.0},
    { 401, 202,  5,  NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_GRAY2RGB,      0.0},
    { 201, 402,  5,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_COLOR_RGB2GRAY,      1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_COLOR_RGB2GRAY,      1.0},
    {  44,  45,  2,  NVCV_IMAGE_FORMAT_YS8,      NVCV_IMAGE_FORMAT_BGRS8,    NVCV_COLOR_GRAY2BGR,      0.0},
    {  44,  45,  4,  NVCV_IMAGE_FORMAT_YS8,      NVCV_IMAGE_FORMAT_RGBS8,    NVCV_COLOR_GRAY2RGB,      0.0},

    {  32,  22,  4,  NVCV_IMAGE_FORMAT_Y16,      NVCV_IMAGE_FORMAT_BGR16,    NVCV_COLOR_GRAY2BGR,      0.0},
    {  32,  20,  4,  NVCV_IMAGE_FORMAT_YS16,     NVCV_IMAGE_FORMAT_BGRS16,   NVCV_COLOR_GRAY2BGR,      0.0},
    {  32,  21,  4,  NVCV_IMAGE_FORMAT_BGR16,    NVCV_IMAGE_FORMAT_Y16,      NVCV_COLOR_BGR2GRAY,      2.0},
    {  54,  66,  5,  NVCV_IMAGE_FORMAT_Y16,      NVCV_IMAGE_FORMAT_RGB16,    NVCV_COLOR_GRAY2RGB,      0.0},
    {  54,  66,  5,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_Y16,      NVCV_COLOR_RGB2GRAY,      2.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_Y16,      NVCV_COLOR_RGB2GRAY,      2.0},

    {  52,  22,  2,  NVCV_IMAGE_FORMAT_Yf16,     NVCV_IMAGE_FORMAT_BGRf16,   NVCV_COLOR_GRAY2BGR,   ERR1_4},

    {  44,  17,  7,  NVCV_IMAGE_FORMAT_YS32,     NVCV_IMAGE_FORMAT_BGRS32,   NVCV_COLOR_GRAY2BGR,      0.0},
    {  88,  34,  4,  NVCV_IMAGE_FORMAT_YS32,     NVCV_IMAGE_FORMAT_RGBS32,   NVCV_COLOR_GRAY2RGB,      0.0},

    {  64,  21,  3,  NVCV_IMAGE_FORMAT_Yf32,     NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_GRAY2BGR,   ERR1_4},
    {  64,  21,  4,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_Yf32,     NVCV_COLOR_BGR2GRAY,   ERR1_4},
    { 121,  66,  6,  NVCV_IMAGE_FORMAT_Yf32,     NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_GRAY2RGB,   ERR1_4},
    { 121,  66,  5,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_Yf32,     NVCV_COLOR_RGB2GRAY,   ERR1_4},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_Yf32,     NVCV_COLOR_RGB2GRAY,   ERR1_4},

    {  68,  33,  3,  NVCV_IMAGE_FORMAT_Yf64,     NVCV_IMAGE_FORMAT_BGRf64,   NVCV_COLOR_GRAY2BGR,   ERR1_4},
    { 127,  28,  6,  NVCV_IMAGE_FORMAT_Yf64,     NVCV_IMAGE_FORMAT_RGBf64,   NVCV_COLOR_GRAY2RGB,   ERR1_4},

    // codes 9 to 11
    {  25,  21,  5,   NVCV_IMAGE_FORMAT_Y8,       NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_GRAY2BGRA,     0.0},
    {  44,  21,  5,   NVCV_IMAGE_FORMAT_Y8,       NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_GRAY2RGBA,     0.0},
    {  21,  22, 11,   NVCV_IMAGE_FORMAT_BGRA8,    NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_BGRA2GRAY,     1.0},
    {  28,  22,  7,   NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_RGBA2GRAY,     1.0},

    {  14,  7,  8,   NVCV_IMAGE_FORMAT_Y16,      NVCV_IMAGE_FORMAT_BGRA16,   NVCV_COLOR_GRAY2BGRA,     0.0},
    {  22, 17,  8,   NVCV_IMAGE_FORMAT_Y16,      NVCV_IMAGE_FORMAT_RGBA16,   NVCV_COLOR_GRAY2RGBA,     0.0},
    {  17,  22,  6,  NVCV_IMAGE_FORMAT_BGRA16,   NVCV_IMAGE_FORMAT_Y16,      NVCV_COLOR_BGRA2GRAY,     2.0},
    {  57,  27,  6,  NVCV_IMAGE_FORMAT_RGBA16,   NVCV_IMAGE_FORMAT_Y16,      NVCV_COLOR_RGBA2GRAY,     2.0},

    {  45,  17, 6,   NVCV_IMAGE_FORMAT_YS32,     NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_COLOR_GRAY2BGRA,     0.0},
    {  45,  28, 3,   NVCV_IMAGE_FORMAT_YS32,     NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_COLOR_GRAY2RGBA,     0.0},

    {  25, 11,  2,   NVCV_IMAGE_FORMAT_Yf32,     NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_GRAY2BGRA,  ERR1_4},
    {  14,  8, 10,   NVCV_IMAGE_FORMAT_Yf32,     NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_COLOR_GRAY2RGBA,  ERR1_4},
    {  8,  14,  7,   NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_IMAGE_FORMAT_Yf32,     NVCV_COLOR_BGRA2GRAY,  ERR1_4},
    {  22, 87,  4,   NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_IMAGE_FORMAT_Yf32,     NVCV_COLOR_RGBA2GRAY,  ERR1_4},

    {  77,  21, 7,   NVCV_IMAGE_FORMAT_Yf64,     NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_COLOR_GRAY2BGRA,  ERR1_4},
    {  62,  27, 7,   NVCV_IMAGE_FORMAT_Yf64,     NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_COLOR_GRAY2RGBA,  ERR1_4},

    // Codes 12 to 39 are not implemented
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV,       1.0},
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_HSV2BGR,       1.0},
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_HSV8,    NVCV_IMAGE_FORMAT_BGRA8,     NVCV_COLOR_HSV2BGR,       1.0},
    { 366,  14,  5,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_RGB2HSV,       1.0},
    { 366,  14,  5,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_HSV2RGB,       1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_RGB2HSV,       1.0},
    {2880,4096,  1,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_HSV2RGB,       1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV,       1.0},
    {4096,4096,  2,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_HSV2BGR,       1.0},

    // Hue computation differs slightly because CUDA kernel adds FLT_EPSILON to denominator for 'diff' division.
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_HSVf32,   NVCV_COLOR_BGR2HSV,    ERR2_3},
    {  33, 525,  3,  NVCV_IMAGE_FORMAT_HSVf32,   NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_HSV2BGR,    ERR1_4},
    {  33, 525,  4,  NVCV_IMAGE_FORMAT_HSVf32,  NVCV_IMAGE_FORMAT_BGRAf32,   NVCV_COLOR_HSV2BGR,    ERR1_4},
    { 365,  14,  5,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_HSVf32,   NVCV_COLOR_RGB2HSV,    ERR2_3},
    { 367, 223,  2,  NVCV_IMAGE_FORMAT_HSVf32,   NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_HSV2RGB,    ERR1_4},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_HSVf32,   NVCV_COLOR_RGB2HSV,    ERR2_3},
    {5760,4096,  1,  NVCV_IMAGE_FORMAT_HSVf32,   NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_RGB2HSV,    ERR2_3},

    // // Codes 42 to 53 and 56 to 65 and 68 to 69 are not implemented
    { 112, 157,  4,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV_FULL,  1.0},
    { 112, 157,  4,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_HSV2BGR_FULL,  1.0},
    { 333,  13,  3,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_RGB2HSV_FULL,  1.0},
    { 333,  13,  3,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_HSV2RGB_FULL,  1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_RGB2HSV_FULL,  1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_RGB2HSV_FULL,  1.0},

    // Codes 72 to 81 are not implemented
    { 133,  22,  2,  NVCV_IMAGE_FORMAT_YUV8,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR,       1.0},
    { 133,  22,  2,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_YUV8,     NVCV_COLOR_BGR2YUV,       1.0},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_YUV8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB,       1.0},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_YUV8,     NVCV_COLOR_RGB2YUV,       1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_YUV8,     NVCV_COLOR_RGB2YUV,       1.0},

    { 133,  21,  3,  NVCV_IMAGE_FORMAT_YUV16,    NVCV_IMAGE_FORMAT_BGR16,    NVCV_COLOR_YUV2BGR,       1.0},
    { 133,  21,  3,  NVCV_IMAGE_FORMAT_BGR16,    NVCV_IMAGE_FORMAT_YUV16,    NVCV_COLOR_BGR2YUV,       2.0},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_YUV16,    NVCV_IMAGE_FORMAT_RGB16,    NVCV_COLOR_YUV2RGB,       1.0},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_YUV16,    NVCV_COLOR_RGB2YUV,       2.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_YUV16,    NVCV_COLOR_RGB2YUV,       2.0},

    { 133,  21,  3,  NVCV_IMAGE_FORMAT_YUVf32,   NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_YUV2BGR,    ERR1_4},
    { 133,  21,  3,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_YUVf32,   NVCV_COLOR_BGR2YUV,    ERR1_4},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_YUVf32,   NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_YUV2RGB,    ERR1_4},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_YUVf32,   NVCV_COLOR_RGB2YUV,    ERR1_4},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_YUVf32,   NVCV_COLOR_RGB2YUV,    ERR1_4},
    // Codes 86 to 89 are not implemented
    // Codes 90 to 147 dealing with subsampled planes (NV12, etc. formats) are postponed (see comment below)
    //     Codes 109, 110, 113, 114 dealing with VYUY format are not implemented
    //     Codes 125, 126 dealing alpha premultiplication are not implemented
    //     Codes 135 to 139 dealing edge-aware demosaicing are not implemented

    { 120,  20,  2,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_I420,  2.0},
    { 120,  20,  2,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_RGB2YUV_I420,  1.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_I420,  2.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_BGR2YUV_I420,  1.0},
    {  80, 120,  4,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_I420, 2.0},
    {  80, 120,  4,  NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_RGBA2YUV_I420, 1.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_I420, 2.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_BGRA8,    NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_BGRA2YUV_I420, 1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_RGB2YUV_I420,  1.0},

    { 140,  80,  6,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_YV12,  2.0},
    { 140,  80,  6,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_RGB2YUV_YV12,  1.0},
    { 160,  60,  5,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_YV12,  2.0},
    { 160,  60,  5,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_BGR2YUV_YV12,  1.0},
    {  60, 100,  4,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_YV12, 2.0},
    {  60, 100,  4,  NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_RGBA2YUV_YV12, 1.0},
    {  80,  80,  3,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_YV12, 2.0},
    {  80,  80,  3,  NVCV_IMAGE_FORMAT_BGRA8,    NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_BGRA2YUV_YV12, 1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_RGB2YUV_YV12,  1.0},

    // NV12, ... makes varShape raise an error:
    // "NVCV_ERROR_NOT_IMPLEMENTED: Batch image format must not have subsampled planes, but it is: X"
    { 120,  20,  2,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_NV12,  2.0},
    { 120,  20,  2,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_RGB2YUV_NV12,  1.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_NV12,  2.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_BGR2YUV_NV12,  1.0},
    {  80, 120,  4,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_NV12, 2.0},
    {  80, 120,  4,  NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_RGBA2YUV_NV12, 1.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_NV12, 2.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_BGRA8,    NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_BGRA2YUV_NV12, 1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV12,     NVCV_COLOR_RGB2YUV_NV12,  1.0},

    { 140,  80,  6,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_NV21,  2.0},
    { 140,  80,  6,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_RGB2YUV_NV21,  1.0},
    { 160,  60,  5,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_NV21,  2.0},
    { 160,  60,  5,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_BGR2YUV_NV21,  1.0},
    {  60, 100,  4,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_NV21, 2.0},
    {  60, 100,  4,  NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_RGBA2YUV_NV21, 1.0},
    {  80,  80,  3,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_NV21, 2.0},
    {  80,  80,  3,  NVCV_IMAGE_FORMAT_BGRA8,    NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_BGRA2YUV_NV21, 1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_RGB2YUV_NV21,  1.0},
    {4096,4096,  3,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_RGB2YUV_NV21,  1.0},
    {4096,4096,  4,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_NV21,  2.0},

    {  80, 120,  2,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_YUV2GRAY_420,  0.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_YUV2GRAY_420,  0.0},

    { 120,  20,  2,  NVCV_IMAGE_FORMAT_UYVY,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_UYVY,  2.0},
    { 120,  20,  2,  NVCV_IMAGE_FORMAT_UYVY,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_UYVY,  2.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_UYVY,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_UYVY, 2.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_UYVY,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_UYVY, 2.0},

    {  80, 120,  4,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_YUY2,  2.0},
    {  80, 120,  4,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_YUY2,  2.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_YVYU,  2.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_YVYU,  2.0},
    {  80, 120,  4,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_YUY2, 2.0},
    {  80, 120,  4,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_YUY2, 2.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_YVYU, 2.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_YVYU, 2.0},

    {  80, 120,  2,  NVCV_IMAGE_FORMAT_UYVY,     NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_YUV2GRAY_UYVY,  0.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_YUV2GRAY_YUY2,  0.0},

    // Code 148 is not implemented
});

// clang-format on

//--------------------------------------------------------------------------------------------------------------------//
TEST_P(OpCvtColor, correct_output)
{
    int wdth = GetParamValue<0>();
    int hght = GetParamValue<1>();
    int imgs = GetParamValue<2>();

    nvcv::ImageFormat srcFrmt{GetParamValue<3>()};
    nvcv::ImageFormat dstFrmt{GetParamValue<4>()};

    NVCVColorConversionCode code{GetParamValue<5>()};

    double maxDiff{GetParamValue<6>()};

    // Create input and output tensors.
    nvcv::Tensor srcTensor = util::CreateTensor(imgs, wdth, hght, srcFrmt);
    nvcv::Tensor dstTensor = util::CreateTensor(imgs, wdth, hght, dstFrmt);

    NVCVDataType dataType;
    ASSERT_EQ(nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(srcFrmt), 0, &dataType), NVCV_SUCCESS);

    switch (dataType) // NOSONAR: typed test dispatch covers all supported source channel layouts.
    {
    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_2U8:
    case NVCV_DATA_TYPE_3U8:
    case NVCV_DATA_TYPE_4U8:
    case NVCV_DATA_TYPE_S8:
    case NVCV_DATA_TYPE_2S8:
    case NVCV_DATA_TYPE_3S8:
    case NVCV_DATA_TYPE_4S8:
        verifyOutput<uint8_t>(srcTensor, srcFrmt, dstTensor, dstFrmt, code, wdth, hght, imgs, maxDiff);
        break;

    case NVCV_DATA_TYPE_U16:
    case NVCV_DATA_TYPE_2U16:
    case NVCV_DATA_TYPE_3U16:
    case NVCV_DATA_TYPE_4U16:
    case NVCV_DATA_TYPE_S16:
    case NVCV_DATA_TYPE_2S16:
    case NVCV_DATA_TYPE_3S16:
    case NVCV_DATA_TYPE_4S16:
    case NVCV_DATA_TYPE_F16:  // Data type float16 is only allowed in conversions that treat it as 16-bit integer
    case NVCV_DATA_TYPE_2F16: //   (e.g., RGB2BGR or Gray2RGB).
    case NVCV_DATA_TYPE_3F16:
    case NVCV_DATA_TYPE_4F16:
        verifyOutput<uint16_t>(srcTensor, srcFrmt, dstTensor, dstFrmt, code, wdth, hght, imgs, maxDiff);
        break;

    case NVCV_DATA_TYPE_S32:
    case NVCV_DATA_TYPE_2S32:
    case NVCV_DATA_TYPE_3S32:
    case NVCV_DATA_TYPE_4S32:
        verifyOutput<int32_t>(srcTensor, srcFrmt, dstTensor, dstFrmt, code, wdth, hght, imgs, maxDiff);
        break;

    case NVCV_DATA_TYPE_F32:
    case NVCV_DATA_TYPE_2F32:
    case NVCV_DATA_TYPE_3F32:
    case NVCV_DATA_TYPE_4F32:
        verifyOutput<float>(srcTensor, srcFrmt, dstTensor, dstFrmt, code, wdth, hght, imgs, maxDiff);
        break;

    case NVCV_DATA_TYPE_F64:
    case NVCV_DATA_TYPE_2F64:
    case NVCV_DATA_TYPE_3F64:
    case NVCV_DATA_TYPE_4F64:
        verifyOutput<double>(srcTensor, srcFrmt, dstTensor, dstFrmt, code, wdth, hght, imgs, maxDiff);
        break;
    default:
        FAIL() << "Unsupported tensor data type.";
        break;
    }
}

//--------------------------------------------------------------------------------------------------------------------//

NVCV_TEST_SUITE_P(OpCvtColorVarShapeReference,
                  test::ValueList<NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode, int>{
  //  Input Format,            Output Format,           Conversion Code,       Max Diff
                      { NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_BGR8,  NVCV_COLOR_RGB2BGR, 0},
                      { NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBA8, NVCV_COLOR_RGB2RGBA, 0},
                      {NVCV_IMAGE_FORMAT_RGBA8,  NVCV_IMAGE_FORMAT_RGB8, NVCV_COLOR_RGBA2RGB, 0},
                      { NVCV_IMAGE_FORMAT_RGB8,    NVCV_IMAGE_FORMAT_Y8, NVCV_COLOR_RGB2GRAY, 1},
                      { NVCV_IMAGE_FORMAT_HSV8,  NVCV_IMAGE_FORMAT_RGB8,  NVCV_COLOR_HSV2RGB, 1},
});

TEST_P(OpCvtColorVarShapeReference, benchmark_kernels_match_independent_reference)
{
    nvcv::ImageFormat       srcFormat{GetParamValue<0>()};
    nvcv::ImageFormat       dstFormat{GetParamValue<1>()};
    NVCVColorConversionCode code{GetParamValue<2>()};
    int                     maxDiff = GetParamValue<3>();

    // Odd, nonuniform dimensions cover scalar tails and per-image var-shape addressing.
    const std::vector<nvcv::Size2D> sizes{
        {31, 23},
        {37, 19},
        {65, 17}
    };
    const int srcChannels = srcFormat.numChannels();
    const int dstChannels = dstFormat.numChannels();

    std::vector<nvcv::Image>          srcImages;
    std::vector<nvcv::Image>          dstImages;
    std::vector<std::vector<uint8_t>> srcVectors;
    std::vector<std::vector<uint8_t>> refVectors;
    srcImages.reserve(sizes.size());
    dstImages.reserve(sizes.size());
    srcVectors.reserve(sizes.size());
    refVectors.reserve(sizes.size());

    RandEng randEng(0);
    for (const nvcv::Size2D &size : sizes)
    {
        const size_t numPixels = static_cast<size_t>(size.w) * size.h;
        srcVectors.emplace_back(numPixels * srcChannels);
        refVectors.emplace_back(numPixels * dstChannels);

        const bool srcRGBA = srcChannels == 4;
        const bool dstRGBA = dstChannels == 4;
        const bool srcBGR  = IsBGR(srcFormat.swizzle());
        const bool dstBGR  = IsBGR(dstFormat.swizzle());
        PopulateSource(srcVectors.back(), size.w, size.h, 1, srcChannels, numPixels, srcRGBA, srcBGR, code, randEng);
        ASSERT_TRUE(BuildCvtColorReference(refVectors.back(), srcVectors.back(), code, numPixels, size.w, size.h, 1,
                                           srcRGBA, srcBGR, dstRGBA, dstBGR));

        srcImages.emplace_back(size, srcFormat);
        dstImages.emplace_back(size, dstFormat);
        auto srcData = srcImages.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(srcData);
        const size_t srcRowBytes = static_cast<size_t>(size.w) * srcFormat.planePixelStrideBytes(0);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVectors.back().data(),
                               srcRowBytes, srcRowBytes, size.h, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape srcBatch(static_cast<int32_t>(sizes.size()));
    nvcv::ImageBatchVarShape dstBatch(static_cast<int32_t>(sizes.size()));
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::CvtColor op;
    EXPECT_NO_THROW(op(stream, srcBatch, dstBatch, code));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (size_t i = 0; i < sizes.size(); ++i)
    {
        SCOPED_TRACE(i);
        auto dstData = dstImages[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(dstData);
        const size_t         dstRowBytes = static_cast<size_t>(sizes[i].w) * dstFormat.planePixelStrideBytes(0);
        std::vector<uint8_t> dstVector(static_cast<size_t>(sizes[i].h) * dstRowBytes);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(dstVector.data(), dstRowBytes, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowBytes, sizes[i].h, cudaMemcpyDeviceToHost));

        if (maxDiff == 0)
        {
            EXPECT_EQ(refVectors[i], dstVector);
        }
        else
        {
            ASSERT_EQ(refVectors[i].size(), dstVector.size());
            for (size_t j = 0; j < refVectors[i].size(); ++j)
            {
                // This one-level tolerance covers fixed-point luma/HSV rounding against the scalar reference.
                EXPECT_NEAR(refVectors[i][j], dstVector[j], maxDiff) << "At index " << j;
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

//--------------------------------------------------------------------------------------------------------------------//

template<typename T>
static std::vector<uint8_t> GenerateCvtColorPlanarParitySource(int width, int height, int numImages, int srcChannels,
                                                               NVCVColorConversionCode code, bool srcRGBA, bool srcBGR)
{
    const size_t   numPixels = static_cast<size_t>(numImages) * height * width;
    std::vector<T> srcVec(numPixels * srcChannels);
    RandEng        randEng(0);

    PopulateSource(srcVec, width, height, numImages, srcChannels, numPixels, srcRGBA, srcBGR, code, randEng);

    std::vector<uint8_t> srcBytes(srcVec.size() * sizeof(T));
    std::memcpy(srcBytes.data(), srcVec.data(), srcBytes.size());
    return srcBytes;
}

static void RunCvtColorTensorPlanarParity(int width, int height, int numImages, int srcChannels, int dstChannels,
                                          NVCVColorConversionCode code, bool srcRGBA, bool srcBGR)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int elemSize     = 1;
    const int     srcRowStride = width * srcChannels * elemSize;
    const int     dstRowStride = width * dstChannels * elemSize;

    nvcv::Tensor srcI(
        {
            {numImages, height, width, srcChannels},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dstI(
        {
            {numImages, height, width, dstChannels},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor srcP(
        {
            {numImages, srcChannels, height, width},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dstP(
        {
            {numImages, dstChannels, height, width},
            "NCHW"
    },
        nvcv::TYPE_U8);

    auto srcIData = srcI.exportData<nvcv::TensorDataStridedCuda>();
    auto dstIData = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPData = srcP.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData = dstP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcIData && dstIData && srcPData && dstPData);

    auto srcIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcIData);
    auto dstIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstIData);
    auto srcPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPData);
    auto dstPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPData);
    ASSERT_TRUE(srcIAcc && dstIAcc && srcPAcc && dstPAcc);

    const auto srcHwcBytes
        = GenerateCvtColorPlanarParitySource<uint8_t>(width, height, numImages, srcChannels, code, srcRGBA, srcBGR);
    const size_t sampleBytes = static_cast<size_t>(height) * srcRowStride;
    for (int i = 0; i < numImages; ++i)
    {
        const auto           sampleStart = srcHwcBytes.begin() + static_cast<ptrdiff_t>(i * sampleBytes);
        std::vector<uint8_t> hwc(sampleStart, sampleStart + static_cast<ptrdiff_t>(sampleBytes));

        test::planar::UploadInterleavedSample(*srcIAcc, i, hwc, width, height, srcRowStride);
        test::planar::UploadPlanarSample(*srcPAcc, i,
                                         test::planar::DeinterleaveToPlanes(hwc, width, height, srcChannels, elemSize),
                                         width, height, srcChannels, elemSize);
    }

    cvcuda::CvtColor op;
    EXPECT_NO_THROW(op(stream, srcI, dstI, code));
    EXPECT_NO_THROW(op(stream, srcP, dstP, code));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        auto gpuInter    = test::planar::DownloadInterleavedSample(*dstIAcc, i, width, height, dstRowStride);
        auto planesOut   = test::planar::DownloadPlanarSample(*dstPAcc, i, width, height, dstChannels, elemSize);
        auto planarInter = test::planar::InterleaveFromPlanes(planesOut, width, height, dstChannels, elemSize);

        EXPECT_EQ(gpuInter, planarInter);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename T>
static void RunCvtColorVarShapePlanarParity(int width, int height, int numImages, nvcv::ImageFormat planarSrcFmt,
                                            nvcv::ImageFormat interleavedSrcFmt, nvcv::ImageFormat planarDstFmt,
                                            nvcv::ImageFormat interleavedDstFmt, NVCVColorConversionCode code,
                                            bool srcRGBA, bool srcBGR)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int srcChannels  = planarSrcFmt.numChannels();
    const int dstChannels  = planarDstFmt.numChannels();
    const int elemSize     = sizeof(T);
    const int srcRowStride = width * srcChannels * elemSize;
    const int dstRowStride = width * dstChannels * elemSize;

    std::vector<nvcv::Image> srcI;
    std::vector<nvcv::Image> dstI;
    std::vector<nvcv::Image> srcP;
    std::vector<nvcv::Image> dstP;
    for (int i = 0; i < numImages; ++i)
    {
        srcI.emplace_back(nvcv::Size2D{width, height}, interleavedSrcFmt);
        dstI.emplace_back(nvcv::Size2D{width, height}, interleavedDstFmt);
        srcP.emplace_back(nvcv::Size2D{width, height}, planarSrcFmt);
        dstP.emplace_back(nvcv::Size2D{width, height}, planarDstFmt);
    }

    nvcv::ImageBatchVarShape batchSrcI(numImages);
    nvcv::ImageBatchVarShape batchDstI(numImages);
    nvcv::ImageBatchVarShape batchSrcP(numImages);
    nvcv::ImageBatchVarShape batchDstP(numImages);
    batchSrcI.pushBack(srcI.begin(), srcI.end());
    batchDstI.pushBack(dstI.begin(), dstI.end());
    batchSrcP.pushBack(srcP.begin(), srcP.end());
    batchDstP.pushBack(dstP.begin(), dstP.end());

    const auto srcHwcBytes
        = GenerateCvtColorPlanarParitySource<T>(width, height, numImages, srcChannels, code, srcRGBA, srcBGR);
    const size_t sampleBytes = static_cast<size_t>(height) * srcRowStride;
    for (int i = 0; i < numImages; ++i)
    {
        const auto           sampleStart = srcHwcBytes.begin() + static_cast<ptrdiff_t>(i * sampleBytes);
        std::vector<uint8_t> hwc(sampleStart, sampleStart + static_cast<ptrdiff_t>(sampleBytes));

        auto idata = srcI[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(idata->plane(0).basePtr, idata->plane(0).rowStride, hwc.data(),
                                            srcRowStride, srcRowStride, height, cudaMemcpyHostToDevice));

        auto      planes     = test::planar::DeinterleaveToPlanes(hwc, width, height, srcChannels, elemSize);
        auto      pdata      = srcP[i].exportData<nvcv::ImageDataStridedCuda>();
        const int planeBytes = width * height * elemSize;
        ASSERT_EQ(pdata->numPlanes(), srcChannels);
        for (int c = 0; c < srcChannels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(pdata->plane(c).basePtr, pdata->plane(c).rowStride, planes.data() + c * planeBytes,
                                   width * elemSize, width * elemSize, height, cudaMemcpyHostToDevice));
        }
    }

    cvcuda::CvtColor op;
    EXPECT_NO_THROW(op(stream, batchSrcI, batchDstI, code));
    EXPECT_NO_THROW(op(stream, batchSrcP, batchDstP, code));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const int dstPlaneBytes = width * height * elemSize;
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> gpuInter(height * dstRowStride);
        auto                 idata = dstI[i].exportData<nvcv::ImageDataStridedCuda>();
        EXPECT_EQ(cudaSuccess, cudaMemcpy2D(gpuInter.data(), dstRowStride, idata->plane(0).basePtr,
                                            idata->plane(0).rowStride, dstRowStride, height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> planesOut(width * height * dstChannels * elemSize);
        auto                 pdata = dstP[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(pdata->numPlanes(), dstChannels);
        for (int c = 0; c < dstChannels; ++c)
        {
            EXPECT_EQ(cudaSuccess,
                      cudaMemcpy2D(planesOut.data() + c * dstPlaneBytes, width * elemSize, pdata->plane(c).basePtr,
                                   pdata->plane(c).rowStride, width * elemSize, height, cudaMemcpyDeviceToHost));
        }
        auto planarInter = test::planar::InterleaveFromPlanes(planesOut, width, height, dstChannels, elemSize);

        EXPECT_EQ(gpuInter, planarInter);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off

NVCV_TEST_SUITE_P(OpCvtColorPlanarTensor,
test::ValueList<int, int, int, int, int, NVCVColorConversionCode, bool, bool>
{
    //  W,  H,  N, SrcC, DstC, Conversion Code,     SrcRGBA, SrcBGR
    { 31, 23, 2,    3,    3,  NVCV_COLOR_RGB2BGR,    false,  false},
    { 29, 17, 1,    3,    4, NVCV_COLOR_RGB2RGBA,    false,  false},
    { 33, 19, 2,    4,    3, NVCV_COLOR_RGBA2RGB,     true,  false},
    { 27, 25, 2,    3,    1, NVCV_COLOR_RGB2GRAY,    false,  false},
    { 35, 21, 1,    1,    3, NVCV_COLOR_GRAY2RGB,    false,  false},
    { 23, 31, 2,    3,    3,  NVCV_COLOR_RGB2HSV,    false,  false},
    { 25, 29, 1,    3,    3,  NVCV_COLOR_HSV2RGB,    false,  false},
    { 37, 27, 2,    3,    3,  NVCV_COLOR_RGB2YUV,    false,  false},
    { 39, 23, 1,    3,    3,  NVCV_COLOR_YUV2RGB,    false,  false},
});

NVCV_TEST_SUITE_P(OpCvtColorPlanarVarShape,
test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVColorConversionCode, bool, bool>
{
    //  W,  H,  N, Planar Src,       Interleaved Src, Planar Dst,        Interleaved Dst, Conversion Code,   SrcRGBA, SrcBGR
    { 31, 23, 2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,  nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8, NVCV_COLOR_RGB2RGBA,  false,  false},
    { 29, 17, 1, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8,   nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, NVCV_COLOR_RGBA2RGB,   true,  false},
    { 33, 21, 2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,    nvcv::FMT_BGR8p,  nvcv::FMT_BGR8, NVCV_COLOR_RGB2BGR,   false, false},
    { 35, 19, 2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,    nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, NVCV_COLOR_RGB2HSV,   false, false},
    { 37, 23, 1,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,    nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, NVCV_COLOR_HSV2RGB,   false, false},
    { 39, 25, 2,  nvcv::FMT_BGR8p,  nvcv::FMT_BGR8,    nvcv::FMT_BGR8p,  nvcv::FMT_BGR8, NVCV_COLOR_BGR2HSV_FULL, false, true},
    { 41, 27, 1,  nvcv::FMT_BGR8p,  nvcv::FMT_BGR8,    nvcv::FMT_BGR8p,  nvcv::FMT_BGR8, NVCV_COLOR_HSV2BGR_FULL, false, true},
    { 43, 29, 2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,    nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, NVCV_COLOR_RGB2YUV,   false, false},
    { 45, 31, 1,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,    nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, NVCV_COLOR_YUV2RGB,   false, false},
    { 47, 33, 2,  nvcv::FMT_BGR8p,  nvcv::FMT_BGR8,    nvcv::FMT_BGR8p,  nvcv::FMT_BGR8, NVCV_COLOR_BGR2YUV,   false, true},
    { 49, 35, 1,  nvcv::FMT_BGR8p,  nvcv::FMT_BGR8,    nvcv::FMT_BGR8p,  nvcv::FMT_BGR8, NVCV_COLOR_YUV2BGR,   false, true},
});

NVCV_TEST_SUITE_P(OpCvtColorPlanarVarShapeFloat,
test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVColorConversionCode, bool, bool>
{
    //  W,  H,  N, Planar Src,          Interleaved Src,    Planar Dst,          Interleaved Dst, Conversion Code,      SrcRGBA, SrcBGR
    { 23, 17, 2, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, NVCV_COLOR_RGB2HSV,      false, false},
    { 25, 19, 1, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, NVCV_COLOR_HSV2RGB,      false, false},
    { 27, 21, 2, nvcv::FMT_BGRf32p,  nvcv::FMT_BGRf32, nvcv::FMT_BGRf32p,  nvcv::FMT_BGRf32, NVCV_COLOR_BGR2HSV_FULL, false, true },
    { 29, 23, 1, nvcv::FMT_BGRf32p,  nvcv::FMT_BGRf32, nvcv::FMT_BGRf32p,  nvcv::FMT_BGRf32, NVCV_COLOR_HSV2BGR_FULL, false, true },
    { 31, 25, 2, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, NVCV_COLOR_RGB2YUV,      false, false},
    { 33, 27, 1, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, NVCV_COLOR_YUV2RGB,      false, false},
});

// clang-format on

TEST_P(OpCvtColorPlanarTensor, tensor_matches_interleaved)
{
    RunCvtColorTensorPlanarParity(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                                  GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(), GetParamValue<7>());
}

TEST_P(OpCvtColorPlanarVarShape, varshape_matches_interleaved)
{
    RunCvtColorVarShapePlanarParity<uint8_t>(
        GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(),
        GetParamValue<5>(), GetParamValue<6>(), GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>());
}

TEST_P(OpCvtColorPlanarVarShapeFloat, varshape_matches_interleaved)
{
    RunCvtColorVarShapePlanarParity<float>(
        GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(),
        GetParamValue<5>(), GetParamValue<6>(), GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>());
}

//--------------------------------------------------------------------------------------------------------------------//

#define VEC_EXPECT_NEAR(vec1, vec2, delta, dtype)                                                                    \
    ASSERT_EQ(vec1.size(), vec2.size());                                                                             \
    for (std::size_t idx = 0; idx < vec1.size() / sizeof(dtype); ++idx)                                              \
    {                                                                                                                \
        EXPECT_NEAR(reinterpret_cast<dtype *>(vec1.data())[idx], reinterpret_cast<dtype *>(vec2.data())[idx], delta) \
            << "At index " << idx;                                                                                   \
    }

// clang-format off

NVCV_TEST_SUITE_P(OpCvtColor_circular,
test::ValueList<int, int, int, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode, NVCVColorConversionCode, double>
{
    //  W,   H,  N,  Input Format,               Output Format,               Convert Code (-->),       Convert Code (<--),   maxDiff
    { 176, 113,  1,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,      0.0},
    { 336, 432,  2,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBA8,    NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,     0.0},
    { 176, 113,  1,  NVCV_IMAGE_FORMAT_BGRS8,    NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,      0.0},
    { 336, 432,  2,  NVCV_IMAGE_FORMAT_RGBS8,    NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRS8,    NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGBS8,    NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,      0.0},
    {  77, 112,  3,  NVCV_IMAGE_FORMAT_RGBS8,    NVCV_IMAGE_FORMAT_BGRS8,    NVCV_COLOR_BGR2RGB,       NVCV_COLOR_RGB2BGR,       0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAS8,   NVCV_IMAGE_FORMAT_BGRAS8,   NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,     0.0},
    { 176, 113,  1,  NVCV_IMAGE_FORMAT_BGR16,    NVCV_IMAGE_FORMAT_BGRA16,   NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,      0.0},
    { 336, 432,  2,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_RGBA16,   NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGR16,    NVCV_IMAGE_FORMAT_RGBA16,   NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_BGRA16,   NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBA16,   NVCV_IMAGE_FORMAT_BGRA16,   NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,     0.0},
    {  77, 110,  3,  NVCV_IMAGE_FORMAT_RGBS16,   NVCV_IMAGE_FORMAT_BGRS16,   NVCV_COLOR_BGR2RGB,       NVCV_COLOR_RGB2BGR,       0.0},
    { 176, 113,  1,  NVCV_IMAGE_FORMAT_BGRS16,   NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,      0.0},
    { 336, 432,  2,  NVCV_IMAGE_FORMAT_RGBS16,   NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRS16,   NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGBS16,   NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAS16,  NVCV_IMAGE_FORMAT_BGRAS16,  NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,     0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRf16,   NVCV_IMAGE_FORMAT_RGBf16,   NVCV_COLOR_BGR2RGB,       NVCV_COLOR_RGB2BGR,       0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAf16,  NVCV_IMAGE_FORMAT_BGRAf16,  NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,     0.0},
    { 176, 113,  1,  NVCV_IMAGE_FORMAT_BGRS32,   NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,      0.0},
    {  88, 110,  3,  NVCV_IMAGE_FORMAT_RGBS32,   NVCV_IMAGE_FORMAT_BGRS32,   NVCV_COLOR_BGR2RGB,       NVCV_COLOR_RGB2BGR,       0.0},
    { 336, 432,  2,  NVCV_IMAGE_FORMAT_RGBS32,   NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRS32,   NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGBS32,   NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAS32,  NVCV_IMAGE_FORMAT_BGRAS32,  NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,     0.0},
    { 176, 113,  1,  NVCV_IMAGE_FORMAT_BGRf64,   NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,      0.0},
    {  77, 177,  3,  NVCV_IMAGE_FORMAT_RGBf64,   NVCV_IMAGE_FORMAT_BGRf64,   NVCV_COLOR_BGR2RGB,       NVCV_COLOR_RGB2BGR,       0.0},
    { 336, 432,  2,  NVCV_IMAGE_FORMAT_RGBf64,   NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,      0.0},
    {  77, 212,  3,  NVCV_IMAGE_FORMAT_BGRf64,   NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,      0.0},
    {  33,  55,  4,  NVCV_IMAGE_FORMAT_RGBf64,   NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,      0.0},
    { 123, 321,  5,  NVCV_IMAGE_FORMAT_RGBAf64,  NVCV_IMAGE_FORMAT_BGRAf64,  NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,     0.0},
    {  23,  21, 63,  NVCV_IMAGE_FORMAT_Y8,       NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_GRAY2BGR,      NVCV_COLOR_BGR2GRAY,      0.0},
    { 402, 202,  5,  NVCV_IMAGE_FORMAT_Y8,       NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_GRAY2RGB,      NVCV_COLOR_RGB2GRAY,      0.0},
    {  32,  21,  4,  NVCV_IMAGE_FORMAT_Y16,      NVCV_IMAGE_FORMAT_BGR16,    NVCV_COLOR_GRAY2BGR,      NVCV_COLOR_BGR2GRAY,      0.0},
    {  54,  66,  5,  NVCV_IMAGE_FORMAT_Y16,      NVCV_IMAGE_FORMAT_RGB16,    NVCV_COLOR_GRAY2RGB,      NVCV_COLOR_RGB2GRAY,      0.0},
    {  64,  21,  3,  NVCV_IMAGE_FORMAT_Yf32,     NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_GRAY2BGR,      NVCV_COLOR_BGR2GRAY,     1E-4},
    {  121, 66,  5,  NVCV_IMAGE_FORMAT_Yf32,     NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_GRAY2RGB,      NVCV_COLOR_RGB2GRAY,     1E-4},
    { 129,  61,  4,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,      0.0},
    {  55, 110,  3,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_BGR2RGB,       NVCV_COLOR_RGB2BGR,       0.0},
    {  63,  31,  3,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,      0.0},
    {  42, 111,  2,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,      0.0},
    {  21,  72,  2,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,      0.0},
    {  23,  31,  3,  NVCV_IMAGE_FORMAT_RGBAf32,  NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,     0.0},
    // Codes 9 to 39 are not implemented
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV,       NVCV_COLOR_HSV2BGR,       5.0},
    { 366,  14,  5,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_RGB2HSV,       NVCV_COLOR_HSV2RGB,       5.0},
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_HSVf32,   NVCV_COLOR_BGR2HSV,       NVCV_COLOR_HSV2BGR,      1E-2},
    { 366,  14,  5,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_HSVf32,   NVCV_COLOR_RGB2HSV,       NVCV_COLOR_HSV2RGB,      1E-2},
    // Codes 42 to 53 and 56 to 65 and 68 to 69 are not implemented
    { 112, 157,  4,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV_FULL,  NVCV_COLOR_HSV2BGR_FULL,  8.0},
    { 333,  13,  3,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_RGB2HSV_FULL,  NVCV_COLOR_HSV2RGB_FULL,  8.0},
    // Codes 72 to 81 are not implemented
    { 133,  22,  2,  NVCV_IMAGE_FORMAT_YUV8,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR,       NVCV_COLOR_BGR2YUV,     128.0},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_YUV8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB,       NVCV_COLOR_RGB2YUV,     128.0},
    { 133,  21,  3,  NVCV_IMAGE_FORMAT_YUV16,    NVCV_IMAGE_FORMAT_BGR16,    NVCV_COLOR_YUV2RGB,       NVCV_COLOR_RGB2YUV,   32768.0},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_YUV16,    NVCV_IMAGE_FORMAT_RGB16,    NVCV_COLOR_YUV2RGB,       NVCV_COLOR_RGB2YUV,   32768.0},
    { 133,  21,  3,  NVCV_IMAGE_FORMAT_YUVf32,   NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_YUV2RGB,       NVCV_COLOR_RGB2YUV,      1E-2},
    { 123,  21,  3,  NVCV_IMAGE_FORMAT_YUVf32,   NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_YUV2RGB,       NVCV_COLOR_RGB2YUV,      1E-2},
    // Codes 86 to 89 are not implemented
    // Codes 90 to 147 dealing with subsampled planes (NV12, etc. formats) are postponed (see comment below)
    //     Codes 109, 110, 113, 114 dealing with VYUY format are not implemented
    //     Codes 125, 126 dealing alpha premultiplication are not implemented
    //     Codes 135 to 139 dealing edge-aware demosaicing are not implemented

    // NV12, ... makes tensors raise an error:
    // "NVCV_ERROR_NOT_IMPLEMENTED: Batch image format must not have subsampled planes, but it is: X"
    { 120,  20,  2,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_NV12,  NVCV_COLOR_RGB2YUV_NV12,   128.0},
    { 100,  40,  3,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_NV12,  NVCV_COLOR_BGR2YUV_NV12,   128.0},
    {  80, 120,  4,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_NV12, NVCV_COLOR_RGBA2YUV_NV12,  128.0},
    {  60,  60,  5,  NVCV_IMAGE_FORMAT_NV12,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_NV12, NVCV_COLOR_BGRA2YUV_NV12,  128.0},
    { 140,  80,  6,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_NV21,  NVCV_COLOR_RGB2YUV_NV21,   128.0},
    { 160,  60,  5,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_NV21,  NVCV_COLOR_BGR2YUV_NV21,   128.0},
    {  60, 100,  4,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_RGBA8,    NVCV_COLOR_YUV2RGBA_NV21, NVCV_COLOR_RGBA2YUV_NV21,  128.0},
    {  80,  80,  3,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGRA_NV21, NVCV_COLOR_BGRA2YUV_NV21,  128.0},
/*
    { 120,  40,  2,  NVCV_IMAGE_FORMAT_UYVY,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_UYVY,  NVCV_COLOR_RGB2YUV,        128.0},
    { 120,  40,  2,  NVCV_IMAGE_FORMAT_YUYV,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_YUV2RGB_YUYV,  NVCV_COLOR_RGB2YUV,        128.0},
*/

    // Code 148 is not implemented
});

// clang-format on

TEST_P(OpCvtColor_circular, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat srcFormat{GetParamValue<3>()};
    nvcv::ImageFormat dstFormat{GetParamValue<4>()};

    // clang-format off
    // Waive the formats that have subsampled planes.
    if (srcFormat.chromaSubsampling() != nvcv::ChromaSubsampling::CSS_444 ||
        dstFormat.chromaSubsampling() != nvcv::ChromaSubsampling::CSS_444)
    {
        GTEST_SKIP() << "Waived the formats that have subsampled planes for OpCvtColor varshape test";
    }
    // clang-format on

    NVCVDataType nvcvDataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(srcFormat), 0, &nvcvDataType));

    NVCVColorConversionCode src2dstCode{GetParamValue<5>()};
    NVCVColorConversionCode dst2srcCode{GetParamValue<6>()};

    double maxDiff{GetParamValue<7>()};

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, srcFormat);

        int srcRowStride   = imgSrc[i].size().w * srcFormat.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        switch (nvcvDataType)
        {
        case NVCV_DATA_TYPE_F32:
        case NVCV_DATA_TYPE_2F32:
        case NVCV_DATA_TYPE_3F32:
        case NVCV_DATA_TYPE_4F32:
            generateRandVec(reinterpret_cast<float *>(srcVec[i].data()), srcVec[i].size() / sizeof(float), rng);
            break;
        default:
            generateRandVec(srcVec[i].data(), srcVec[i].size(), rng);
            break;
        }

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;

    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), dstFormat);
    }

    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // run operator
    cvcuda::CvtColor cvtColorOp;

    EXPECT_NO_THROW(cvtColorOp(stream, batchSrc, batchDst, src2dstCode));

    EXPECT_NO_THROW(cvtColorOp(stream, batchDst, batchSrc, dst2srcCode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);
        ASSERT_EQ(imgData->numPlanes(), 1);

        std::vector<uint8_t> testVec(imgSrc[i].size().h * srcVecRowStride[i]);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), srcVecRowStride[i], imgData->plane(0).basePtr,
                                            imgData->plane(0).rowStride, srcVecRowStride[i], imgSrc[i].size().h,
                                            cudaMemcpyDeviceToHost));

        switch (nvcvDataType)
        {
        case NVCV_DATA_TYPE_F32:
        case NVCV_DATA_TYPE_2F32:
        case NVCV_DATA_TYPE_3F32:
        case NVCV_DATA_TYPE_4F32:
            VEC_EXPECT_NEAR(testVec, srcVec[i], maxDiff, float);
            break;
        default:
            VEC_EXPECT_NEAR(testVec, srcVec[i], maxDiff, uint8_t);
            break;
        }
    }
}

TEST(OpCvtColor_negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaCvtColorCreate(nullptr));
}

TEST(OpCvtColor_Negative, planar_tensor_layout_mismatch)
{
    nvcv::Tensor srcTensor(
        {
            {1, 3, 8, 8},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dstTensor(
        {
            {1, 8, 8, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);

    cvcuda::CvtColor cvtColorOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_RGB2BGR); }));
}

TEST(OpCvtColor_Negative, planar_varshape_layout_mismatch)
{
    std::vector<nvcv::Image> imgSrc;
    imgSrc.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_RGB8p);

    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    std::vector<nvcv::Image> imgDst;
    imgDst.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_RGBA8);

    nvcv::ImageBatchVarShape batchDst(1);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::CvtColor cvtColorOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvtColorOp(nullptr, batchSrc, batchDst, NVCV_COLOR_RGB2RGBA); }));
}

TEST(OpCvtColor_negative, mismatch_shape)
{
    nvcv::Tensor tensorY8   = util::CreateTensor(2, 224, 224, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_Y8});
    nvcv::Tensor tensorHSV8 = util::CreateTensor(2, 224, 224, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_HSV8});
    nvcv::Tensor tensorBGR8 = util::CreateTensor(5, 224, 224, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8});
    nvcv::Tensor tensorRGB8 = util::CreateTensor(2, 224, 224, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB8});

    // run operator
    cvcuda::CvtColor cvtColorOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&cvtColorOp, &tensorY8, &tensorBGR8]
                                { cvtColorOp(nullptr, tensorY8, tensorBGR8, NVCV_COLOR_GRAY2BGR); }));

    // reserved conversion invalid too
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&cvtColorOp, &tensorBGR8, &tensorY8]
                                { cvtColorOp(nullptr, tensorBGR8, tensorY8, NVCV_COLOR_BGR2GRAY); }));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&cvtColorOp, &tensorHSV8, &tensorBGR8]
                                { cvtColorOp(nullptr, tensorHSV8, tensorBGR8, NVCV_COLOR_HSV2BGR); }));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&cvtColorOp, &tensorBGR8, &tensorRGB8]
                                { cvtColorOp(nullptr, tensorBGR8, tensorRGB8, NVCV_COLOR_BGR2RGB); }));
}

TEST(OpCvtColor_negative, invalid_shape_BGR_to_YUV420xp)
{
    std::vector<nvcv::Tensor> srcTensors{util::CreateTensor(1, 7, 8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8}),
                                         util::CreateTensor(1, 8, 6, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8}),
                                         util::CreateTensor(1, 8, 8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGRf16}),
                                         util::CreateTensor(1, 16, 16, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8})};

    auto dstTensor = nvcv::Tensor(
        {
            {8, 8, 1},
            "HWC"
    },
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV21}.planeDataType(0).channelType(0));

    // run operator
    cvcuda::CvtColor cvtColorOp;
    for (const auto &srcTensor : srcTensors)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&cvtColorOp, &srcTensor, &dstTensor]
                                    { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_BGR2YUV_YV12); }));
    }
}

TEST(OpCvtColor_negative, invalid_shape_YUV420xp_toBGR)
{
    std::vector<nvcv::Tensor> srcTensors{
        nvcv::Tensor({             {9, 8, 2}, "HWC"},
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV21      }
        .planeDataType(0).channelType(0)),
        nvcv::Tensor({             {7, 8, 1}, "HWC"},
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV21      }
        .planeDataType(0).channelType(0)),
        nvcv::Tensor({             {9, 8, 1}, "HWC"},
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV21      }
        .planeDataType(0).channelType(0)),
        nvcv::Tensor({            {12, 8, 1}, "HWC"},
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV21      }
        .planeDataType(0).channelType(0)),
        nvcv::Tensor({             {9, 8, 1}, "HWC"},
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV21      }
        .planeDataType(0).channelType(0)),
    };
    nvcv::Tensor srcTensor_1 = util::CreateTensor(1, 8, 8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV21});

    // height: 8 --> 6
    nvcv::Tensor dstTensor   = util::CreateTensor(1, 8, 6, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8});
    nvcv::Tensor dstTensor_1 = util::CreateTensor(1, 8, 6, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_Y8});

    // run operator
    cvcuda::CvtColor cvtColorOp;
    for (const auto &srcTensor : srcTensors)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&cvtColorOp, &srcTensor, &dstTensor]
                                    { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_YUV2BGR_YV12); }));
    }

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&cvtColorOp, &srcTensor_1, &dstTensor_1] {
                                                   cvtColorOp(nullptr, srcTensor_1, dstTensor_1,
                                                              NVCV_COLOR_YUV2BGR_YV12);
                                               })); // incalid output channel
}

TEST(OpCvtColor_negative, invalid_shape_YUV422_to_BGR)
{
    std::vector<nvcv::Tensor> srcTensors = {
        nvcv::Tensor({          {120, 21, 1}, "HWC"},
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_UYVY      }
        .planeDataType(0).channelType(0)),
        nvcv::Tensor({          {120, 20, 2}, "HWC"},
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_UYVY      }
        .planeDataType(0).channelType(0)),
        nvcv::Tensor({          {120, 24, 1}, "HWC"},
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_UYVY      }
        .planeDataType(0).channelType(0))
    };

    auto dstTensor = nvcv::Tensor(
        {
            {120, 40, 3},
            "HWC"
    },
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8}.planeDataType(0).channelType(0));

    // run operator
    cvcuda::CvtColor cvtColorOp;
    for (const auto &srcTensor : srcTensors)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&cvtColorOp, &srcTensor, &dstTensor]
                                    { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_YUV2BGR_UYVY); }));
    }
}

TEST(OpCvtColor_negative, invalid_shape_YUV422_to_BGR_invalid_out)
{
    auto srcTensor = nvcv::Tensor(
        {
            {120, 20, 1},
            "HWC"
    },
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_UYVY}.planeDataType(0).channelType(0));

    std::vector<nvcv::Tensor> dstTensors = {
        nvcv::Tensor({            {120, 20, 1}, "HWC"},
        nvcv::ImageFormat{  NVCV_IMAGE_FORMAT_BGR8      }
        .planeDataType(0).channelType(0)),
        nvcv::Tensor({            {120, 20, 3}, "HWC"},
                     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGRf16      }
        .planeDataType(0).channelType(0))
    };

    // run operator
    cvcuda::CvtColor cvtColorOp;
    for (const auto &dstTensor : dstTensors)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&cvtColorOp, &srcTensor, &dstTensor]
                                    { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_YUV2BGR_UYVY); }));
    }
}

TEST(OpCvtColor_negative, invalid_conversion_code)
{
    nvcv::Tensor srcTensor = util::CreateTensor(1, 8, 8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8});
    nvcv::Tensor dstTensor = util::CreateTensor(1, 8, 8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB8});

    cvcuda::CvtColor cvtColorOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&] { cvtColorOp(nullptr, srcTensor, dstTensor, static_cast<NVCVColorConversionCode>(1000000)); }));
}

TEST(OpCvtColor_negative, varshape_invalid_conversion_code)
{
    std::vector<nvcv::Image> imgSrc;
    imgSrc.emplace_back(nvcv::Size2D{8, 8}, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8});

    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    std::vector<nvcv::Image> imgDst;
    imgDst.emplace_back(nvcv::Size2D{8, 8}, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB8});

    nvcv::ImageBatchVarShape batchDst(1);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::CvtColor cvtColorOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&] { cvtColorOp(nullptr, batchSrc, batchDst, static_cast<NVCVColorConversionCode>(1000000)); }));
}

// clang-format off

NVCV_TEST_SUITE_P(OpCvtColor_negative,
test::ValueList<int, int, int, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode>
{
    //  W,   H,  N,  Input Format,              Output Format,              Conversion Code
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_Y8,      NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_BGR2BGRA}, // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_BGR2BGRA}, // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGRA8p,   NVCV_COLOR_BGR2BGRA}, // mismatch format
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_BGR2BGRA}, // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_GRAY2BGR}, // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_Y8,      NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_GRAY2BGR}, // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_Y8,      NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_GRAY2BGR}, // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_Yf16,    NVCV_IMAGE_FORMAT_BGRAf16,  NVCV_COLOR_GRAY2BGRA}, // invalid f16 + adding alpha
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRA8,   NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_BGR2GRAY}, // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf32,  NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_BGR2GRAY}, // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf16,  NVCV_IMAGE_FORMAT_BGRAf16,  NVCV_COLOR_BGR2BGRA}, // f16 type not allowed to add alpha
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_BGR2GRAY}, // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf64,  NVCV_IMAGE_FORMAT_Yf64,     NVCV_COLOR_BGR2GRAY}, // unsupported data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRA8,   NVCV_IMAGE_FORMAT_YUV8,     NVCV_COLOR_BGR2YUV},  // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf32,  NVCV_IMAGE_FORMAT_YUV8,     NVCV_COLOR_BGR2YUV},  // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_BGR2YUV},  // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf16,  NVCV_IMAGE_FORMAT_YUVf16,   NVCV_COLOR_BGR2YUV},  // unsupported data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRA8,   NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR},  // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_YUV8,    NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_YUV2BGR},  // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_YUV8,    NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_YUV2BGR},  // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_YUVf16,  NVCV_IMAGE_FORMAT_BGRf16,   NVCV_COLOR_YUV2BGR},  // unsupported data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRA8,   NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV},  // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf32,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV},  // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf64,  NVCV_IMAGE_FORMAT_HSVf64,   NVCV_COLOR_BGR2HSV},  // unsupported data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_BGR2HSV},  // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRA8,   NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_HSV2BGR},  // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_HSV8,    NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_HSV2BGR},  // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_HSV8,    NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_HSV2BGR},  // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_HSVf64,  NVCV_IMAGE_FORMAT_BGRf64,   NVCV_COLOR_HSV2BGR},  // unsupported data type
    {  16,   8,  1,  NVCV_IMAGE_FORMAT_Y8,      NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_BGR2YUV_YV12}, // invalid channel
});

// clang-format on

TEST_P(OpCvtColor_negative, invalid_input)
{
    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat srcFormat{GetParamValue<3>()};
    nvcv::ImageFormat dstFormat{GetParamValue<4>()};

    NVCVColorConversionCode src2dstCode{GetParamValue<5>()};

    nvcv::Tensor srcTensor = util::CreateTensor(batches, width, height, srcFormat);
    nvcv::Tensor dstTensor = util::CreateTensor(batches, width, height, dstFormat);

    // run operator
    cvcuda::CvtColor cvtColorOp;
    EXPECT_ANY_THROW(cvtColorOp(nullptr, srcTensor, dstTensor, src2dstCode));
}

TEST_P(OpCvtColor_negative, varshape_invalid_input)
{
    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat srcFormat{GetParamValue<3>()};
    nvcv::ImageFormat dstFormat{GetParamValue<4>()};

    NVCVColorConversionCode src2dstCode{GetParamValue<5>()};

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, srcFormat);
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;

    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), dstFormat);
    }

    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // run operator
    cvcuda::CvtColor cvtColorOp;
    EXPECT_ANY_THROW(cvtColorOp(nullptr, batchSrc, batchDst, src2dstCode));
}

// clang-format off

NVCV_TEST_SUITE_P(OpCvtColor_negative_diff_format, test::ValueList<NVCVImageFormat, NVCVImageFormat, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode>
{
    {NVCV_IMAGE_FORMAT_Y8,       NVCV_IMAGE_FORMAT_Y16,       NVCV_IMAGE_FORMAT_BGR8,       NVCV_IMAGE_FORMAT_BGR8,      NVCV_COLOR_GRAY2BGR},
    {NVCV_IMAGE_FORMAT_Y8,       NVCV_IMAGE_FORMAT_Y8,        NVCV_IMAGE_FORMAT_BGR8,       NVCV_IMAGE_FORMAT_BGRf32,    NVCV_COLOR_GRAY2BGR},
    {NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_RGBf32,    NVCV_IMAGE_FORMAT_HSV8,       NVCV_IMAGE_FORMAT_BGR8,      NVCV_COLOR_BGR2HSV},
    {NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_BGR8,      NVCV_IMAGE_FORMAT_HSV8,       NVCV_IMAGE_FORMAT_BGRf32,    NVCV_COLOR_BGR2HSV},
});
// clang-format on

#undef NVCV_IMAGE_FORMAT_RGBS8
#undef NVCV_IMAGE_FORMAT_BGRS8
#undef NVCV_IMAGE_FORMAT_RGBAS8
#undef NVCV_IMAGE_FORMAT_BGRAS8
#undef NVCV_IMAGE_FORMAT_YS8_ER
#undef NVCV_IMAGE_FORMAT_YS8

#undef NVCV_IMAGE_FORMAT_BGR16
#undef NVCV_IMAGE_FORMAT_RGB16
#undef NVCV_IMAGE_FORMAT_BGRA16
#undef NVCV_IMAGE_FORMAT_RGBA16
#undef NVCV_IMAGE_FORMAT_YUV16
#undef NVCV_IMAGE_FORMAT_YS16
#undef NVCV_IMAGE_FORMAT_BGRS16
#undef NVCV_IMAGE_FORMAT_RGBS16
#undef NVCV_IMAGE_FORMAT_BGRAS16
#undef NVCV_IMAGE_FORMAT_RGBAS16
#undef NVCV_IMAGE_FORMAT_YUVf16

#undef NVCV_IMAGE_FORMAT_YS32
#undef NVCV_IMAGE_FORMAT_BGRS32
#undef NVCV_IMAGE_FORMAT_RGBS32
#undef NVCV_IMAGE_FORMAT_BGRAS32
#undef NVCV_IMAGE_FORMAT_RGBAS32
#undef NVCV_IMAGE_FORMAT_YUVf32
#undef NVCV_IMAGE_FORMAT_Yf32
#undef NVCV_IMAGE_FORMAT_HSVf32

#undef NVCV_IMAGE_FORMAT_HSVf64
#undef NVCV_IMAGE_FORMAT_Yf64

TEST_P(OpCvtColor_negative_diff_format, varshape_hasDifferentFormat)
{
    nvcv::ImageFormat       srcFormat{GetParamValue<0>()};
    nvcv::ImageFormat       srcExtraFormat{GetParamValue<1>()};
    nvcv::ImageFormat       dstFormat{GetParamValue<2>()};
    nvcv::ImageFormat       dstExtraFormat{GetParamValue<3>()};
    NVCVColorConversionCode src2dstCode{GetParamValue<4>()};

    int batches = 4;
    int width   = 224;
    int height  = 224;

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;
    for (int i = 0; i < batches - 1; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, srcFormat);
    }
    imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, srcExtraFormat);

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;

    for (int i = 0; i < batches - 1; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), dstFormat);
    }
    imgDst.emplace_back(imgSrc[batches - 1].size(), dstExtraFormat);

    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::CvtColor cvtColorOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&cvtColorOp, &batchSrc, &batchDst, &src2dstCode]
                                { cvtColorOp(nullptr, batchSrc, batchDst, src2dstCode); }));
}

#undef VEC_EXPECT_NEAR
