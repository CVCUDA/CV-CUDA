/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "TestUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

namespace test = nvcv::test;
namespace util = nvcv::util;
namespace cuda = nvcv::cuda;

using std::vector;

#define NVCV_IMAGE_FORMAT_RGBS8  NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X8_Y8_Z8)
#define NVCV_IMAGE_FORMAT_BGRS8  NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, ZYX1, ASSOCIATED, X8_Y8_Z8)
#define NVCV_IMAGE_FORMAT_RGBAS8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8)
#define NVCV_IMAGE_FORMAT_BGRAS8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, ZYXW, ASSOCIATED, X8_Y8_Z8_W8)
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

    int srcWdth = wdth,
        srcHght = hght;
    int dstWdth = wdth,
        dstHght = hght;

    if (srcFrmt == NVCV_IMAGE_FORMAT_UYVY || srcFrmt == NVCV_IMAGE_FORMAT_UYVY_ER ||
        srcFrmt == NVCV_IMAGE_FORMAT_YUYV || srcFrmt == NVCV_IMAGE_FORMAT_YUYV_ER)
        srcWdth = srcWdth << 1;
    if (srcFrmt == NVCV_IMAGE_FORMAT_NV12 || srcFrmt == NVCV_IMAGE_FORMAT_NV12_ER ||
        srcFrmt == NVCV_IMAGE_FORMAT_NV21 || srcFrmt == NVCV_IMAGE_FORMAT_NV21_ER)
        srcHght = (srcHght * 3) >> 1;
    ASSERT_EQ(srcWdth, srcAccess->numCols());
    ASSERT_EQ(srcHght, srcAccess->numRows());

    if (dstFrmt == NVCV_IMAGE_FORMAT_UYVY || dstFrmt == NVCV_IMAGE_FORMAT_UYVY_ER ||
        dstFrmt == NVCV_IMAGE_FORMAT_YUYV || dstFrmt == NVCV_IMAGE_FORMAT_YUYV_ER)
        dstWdth = dstWdth << 1;
    if (dstFrmt == NVCV_IMAGE_FORMAT_NV12 || dstFrmt == NVCV_IMAGE_FORMAT_NV12_ER ||
        dstFrmt == NVCV_IMAGE_FORMAT_NV21 || dstFrmt == NVCV_IMAGE_FORMAT_NV21_ER)
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

    bool srcBGR  = (srcSwizzle == nvcv::Swizzle::S_ZYXW ||
                    srcSwizzle == nvcv::Swizzle::S_ZYX1 ||
                    srcSwizzle == nvcv::Swizzle::S_ZYX0);
    bool dstBGR  = (dstSwizzle == nvcv::Swizzle::S_ZYXW ||
                    dstSwizzle == nvcv::Swizzle::S_ZYX1 ||
                    dstSwizzle == nvcv::Swizzle::S_ZYX0);
    bool srcRGBA = (srcChannels == 4),
         dstRGBA = (dstChannels == 4);
    bool success = true;

    RandEng randEng(0);

    constexpr size_t minCntAllRGB = 128 * 256 * 256; // Minimum # of pixels to call generateAllRGB.
    constexpr size_t minCntAllHSV =  90 * 256 * 256; // Minimum # of pixels to call generateAllHSV.
    constexpr double minMultHSV   = -0.5;            // Set hue range multiplier to be outside normal range
    constexpr double maxMultHSV   =  1.5;            //   to test robustness to wrapped hue values.

    // Populate source tensor.
    if (srcChannels > 2)
    {
        if (code == NVCV_COLOR_HSV2BGR || code == NVCV_COLOR_HSV2BGR_FULL ||
            code == NVCV_COLOR_HSV2RGB || code == NVCV_COLOR_HSV2RGB_FULL)
        {
            bool full = (code == NVCV_COLOR_HSV2BGR_FULL || code == NVCV_COLOR_HSV2RGB_FULL);

            if (numPixels >= minCntAllHSV)
            {
                if (full) generateAllHSV<T, true >(srcVec, srcWdth, srcHght, imgs);
                else      generateAllHSV<T, false>(srcVec, srcWdth, srcHght, imgs);
            }
            else
            {
                if (full) generateRandHSV<T, true >(srcVec, randEng, minMultHSV, maxMultHSV);
                else      generateRandHSV<T, false>(srcVec, randEng, minMultHSV, maxMultHSV);
            }
        }
        else
        {
            if (numPixels >= minCntAllRGB)
                generateAllRGB(srcVec, srcWdth, srcHght, imgs, srcRGBA, srcBGR);
            else
                generateRandTestRGB(srcVec, randEng, srcRGBA, srcBGR);
        }
    }
    else
        generateRandVec(srcVec, randEng);

    // Copy source from image vector to device tensor.
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->basePtr(), srcAccess->rowStride(), srcVec.data(), srcPitchCPU,
                                        srcPitchCPU, (size_t)imgs * (size_t)srcHght, cudaMemcpyHostToDevice));

    switch (code)
    {
    // Add/remove alpha channel to RGB/BGR image.
    case NVCV_COLOR_BGR2BGRA     :  // NVCV_COLOR_BGR2BGRA      =   0 (NVCV_COLOR_RGB2RGBA)
    case NVCV_COLOR_BGRA2BGR     :  // NVCV_COLOR_BGRA2BGR      =   1 (NVCV_COLOR_RGBA2RGB)
        changeAlpha<T>(refVec, srcVec, numPixels, srcRGBA, dstRGBA);
        break;

    // Convert between RGB and BGR (with or without alpha channel).
    case NVCV_COLOR_BGR2RGBA     :  // NVCV_COLOR_BGR2RGBA      =   2 (NVCV_COLOR_RGB2BGRA)
    case NVCV_COLOR_RGBA2BGR     :  // NVCV_COLOR_RGBA2BGR      =   3 (NVCV_COLOR_BGRA2RGB)
    case NVCV_COLOR_BGR2RGB      :  // NVCV_COLOR_BGR2RGB       =   4 (NVCV_COLOR_BGR2RGB)
    case NVCV_COLOR_BGRA2RGBA    :  // NVCV_COLOR_BGRA2RGBA     =   5 (NVCV_COLOR_RGBA2BGRA)
        convertRGBtoBGR<T>(refVec, srcVec, numPixels, srcRGBA, dstRGBA);
        break;

    // Convert from RGB/BGR to grayscale.
    case NVCV_COLOR_BGR2GRAY     :  // NVCV_COLOR_BGR2GRAY      =   6
    case NVCV_COLOR_RGB2GRAY     :  // NVCV_COLOR_RGB2GRAY      =   7
    case NVCV_COLOR_BGRA2GRAY    :  // NVCV_COLOR_BGRA2GRAY     =  10
    case NVCV_COLOR_RGBA2GRAY    :  // NVCV_COLOR_RGBA2GRAY     =  11
        convertRGBtoGray<T>(refVec, srcVec, numPixels, srcRGBA, srcBGR);
        break;

    // Convert from grayscale to RGB/BGR.
    case NVCV_COLOR_GRAY2BGR     :  // NVCV_COLOR_GRAY2BGR      =   8 (NVCV_COLOR_GRAY2RGB)
    case NVCV_COLOR_GRAY2BGRA    :  // NVCV_COLOR_GRAY2BGRA     =   9 (NVCV_COLOR_GRAY2RGBA)
        convertGrayToRGB<T>(refVec, srcVec, numPixels, dstRGBA);
        break;

    // Convert between RGB/BGR   and BGR565 (16-bit images) --> Conversion codes 12-19 not implemented.
    // Convert between grayscale and BGR565 (16-bit images) --> Conversion codes 20-21 not implemented.
    // Convert between RGB/BGR   and BGR555 (16-bit images) --> Conversion codes 22-29 not implemented.
    // Convert between grayscale and BGR555 (16-bit images) --> Conversion codes 30-31 not implemented.
    // Convert between RGB/BGR   and CIE XYZ                --> Conversion codes 32-35 not implemented.
    // Convert between RGB/BGR   and YCrCb (aka YCC)        --> Conversion codes 36-39 not implemented.

    // Convert from RGB/BGR to HSV (hue, saturation, value).
    case NVCV_COLOR_BGR2HSV      :  // NVCV_COLOR_BGR2HSV       =  40
    case NVCV_COLOR_RGB2HSV      :  // NVCV_COLOR_RGB2HSV       =  41
        convertRGBtoHSV<T, false>(refVec, srcVec, numPixels, srcRGBA, srcBGR);
        break;

    // Conversion codes 42 and 43 not specified.
    // Convert from RGB/BGR to CIE Lab                          --> Conversion codes 44-45 not implemented.
    // Bayer demosaicing to RGB/BGR                             --> Conversion codes 46-49 not implemented.
    // Convert from RGB/BGR to CIE Luv                          --> Conversion codes 50-51 not implemented.
    // Convert from RGB/BGR to HLS (hue, lightness, saturation) --> Conversion codes 52-53 not implemented.

    // Convert from HSV (hue, saturation, value) to RGB/BGR.
    case NVCV_COLOR_HSV2BGR      :  // NVCV_COLOR_HSV2BGR       =  54
    case NVCV_COLOR_HSV2RGB      :  // NVCV_COLOR_HSV2RGB       =  55
        convertHSVtoRGB<T, false>(refVec, srcVec, numPixels, dstRGBA, dstBGR);
        break;

    // Convert to RGB/BGR from CIE Lab                           --> Conversion codes 56-57 not implemented.
    // Convert to RGB/BGR from CIE Luv                           --> Conversion codes 58-59 not implemented.
    // Convert to RGB/BGR from HLS (hue, lightness, saturation)  --> Conversion codes 60-61 not implemented.
    // VNG (Variable Number of Gradients) demosaicing to RGB/BGR --> Conversion codes 62-65 not implemented.

    // Convert from RGB/BGR to full-range HSV (hue, saturation, value).
    case NVCV_COLOR_BGR2HSV_FULL :  // NVCV_COLOR_BGR2HSV_FULL  =  66
    case NVCV_COLOR_RGB2HSV_FULL :  // NVCV_COLOR_RGB2HSV_FULL  =  67
        convertRGBtoHSV<T, true>(refVec, srcVec, numPixels, srcRGBA, srcBGR);
        break;

    // Convert from RGB/BGR to full-range HLS (hue, lightness, saturation) --> Conversion codes 68-69 not implemented.

    // Convert from full-range HSV (hue, saturation, value) to RGB/BGR.
    case NVCV_COLOR_HSV2BGR_FULL :  // NVCV_COLOR_HSV2BGR_FULL  =  70
    case NVCV_COLOR_HSV2RGB_FULL :  // NVCV_COLOR_HSV2RGB_FULL  =  71
        convertHSVtoRGB<T, true>(refVec, srcVec, numPixels, dstRGBA, dstBGR);
        break;

    // Convert from full-range HLS (hue, lightness, saturation) to RGB/BGR --> Conversion codes 72-73 not implemented.
    // Convert from LRGB/LBGR (luminance, red, green, blue) to   CIE Lab   --> Conversion codes 74-75 not implemented.
    // Convert from LRGB/LBGR (luminance, red, green, blue) to   CIE Luv   --> Conversion codes 76-77 not implemented.
    // Convert to   LRGB/LBGR (luminance, red, green, blue) from CIE Lab   --> Conversion codes 78-79 not implemented.
    // Convert to   LRGB/LBGR (luminance, red, green, blue) from CIE Luv   --> Conversion codes 80-81 not implemented.

    // Convert from RGB/BGR to YUV.
    case NVCV_COLOR_BGR2YUV      :  // NVCV_COLOR_BGR2YUV       =  82
    case NVCV_COLOR_RGB2YUV      :  // NVCV_COLOR_RGB2YUV       =  83
        convertRGBtoYUV_PAL<T>(refVec, srcVec, numPixels, srcRGBA, srcBGR);
        break;

    // Convert from YUV to RGB/BGR.
    case NVCV_COLOR_YUV2BGR      :  // NVCV_COLOR_YUV2BGR       =  84
    case NVCV_COLOR_YUV2RGB      :  // NVCV_COLOR_YUV2RGB       =  85
        convertYUVtoRGB_PAL<T>(refVec, srcVec, numPixels, dstRGBA, dstBGR);
        break;

    // Bayer demosaicing to grayscale --> Conversion codes 86-89 not implemented.

    // Convert from YUV 4:2:0 family to RGB/BGR.
    case NVCV_COLOR_YUV2RGB_NV12 :  // NVCV_COLOR_YUV2RGB_NV12  =  90
    case NVCV_COLOR_YUV2BGR_NV12 :  // NVCV_COLOR_YUV2BGR_NV12  =  91
    case NVCV_COLOR_YUV2RGBA_NV12:  // NVCV_COLOR_YUV2RGBA_NV12 =  94
    case NVCV_COLOR_YUV2BGRA_NV12:  // NVCV_COLOR_YUV2BGRA_NV12 =  95
        convertNV12toRGB<T>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, false);
        break;

    case NVCV_COLOR_YUV2RGB_NV21 :  // NVCV_COLOR_YUV2RGB_NV21  =  92 (NVCV_COLOR_YUV420sp2RGB)
    case NVCV_COLOR_YUV2BGR_NV21 :  // NVCV_COLOR_YUV2BGR_NV21  =  93 (NVCV_COLOR_YUV420sp2BGR)
    case NVCV_COLOR_YUV2RGBA_NV21:  // NVCV_COLOR_YUV2RGBA_NV21 =  96 (NVCV_COLOR_YUV420sp2RGBA)
    case NVCV_COLOR_YUV2BGRA_NV21:  // NVCV_COLOR_YUV2BGRA_NV21 =  97 (NVCV_COLOR_YUV420sp2BGRA)
        convertNV12toRGB<T>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, true);
        break;

    case NVCV_COLOR_YUV2RGB_YV12 :  // NVCV_COLOR_YUV2RGB_YV12  =  98 (NVCV_COLOR_YUV420p2RGB)
    case NVCV_COLOR_YUV2BGR_YV12 :  // NVCV_COLOR_YUV2BGR_YV12  =  99 (NVCV_COLOR_YUV420p2BGR)
    case NVCV_COLOR_YUV2RGBA_YV12:  // NVCV_COLOR_YUV2RGBA_YV12 = 102 (NVCV_COLOR_YUV420p2RGBA)
    case NVCV_COLOR_YUV2BGRA_YV12:  // NVCV_COLOR_YUV2BGRA_YV12 = 103 (NVCV_COLOR_YUV420p2BGRA)
        convertYUVtoRGB_420<T>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, true);
        break;

    case NVCV_COLOR_YUV2RGB_IYUV :  // NVCV_COLOR_YUV2RGB_IYUV  = 100 (NVCV_COLOR_YUV2RGB_I420)
    case NVCV_COLOR_YUV2BGR_IYUV :  // NVCV_COLOR_YUV2BGR_IYUV  = 101 (NVCV_COLOR_YUV2BGR_I420)
    case NVCV_COLOR_YUV2RGBA_IYUV:  // NVCV_COLOR_YUV2RGBA_IYUV = 104 (NVCV_COLOR_YUV2RGBA_I420)
    case NVCV_COLOR_YUV2BGRA_IYUV:  // NVCV_COLOR_YUV2BGRA_IYUV = 105 (NVCV_COLOR_YUV2BGRA_I420)
        convertYUVtoRGB_420<T>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, false);
        break;

    // Convert from YUV 4:2:0 family to grayscale.
    case NVCV_COLOR_YUV2GRAY_420 :  // NVCV_COLOR_YUV2GRAY_420  = 106 (NVCV_COLOR_YUV2GRAY_NV21, NVCV_COLOR_YUV2GRAY_NV12,
                                    //                                 NVCV_COLOR_YUV2GRAY_YV12, NVCV_COLOR_YUV2GRAY_IYUV,
                                    //                                 NVCV_COLOR_YUV2GRAY_I420, NVCV_COLOR_YUV420sp2GRAY,
                                    //                                 NVCV_COLOR_YUV420p2GRAY)
        convertYUVtoGray_420<T>(refVec, srcVec, wdth, hght, imgs);
        break;

    // Convert from YUV 4:2:2 family to RGB/BGR.
    case NVCV_COLOR_YUV2RGB_UYVY :  // NVCV_COLOR_YUV2RGB_UYVY  = 107 ( NVCV_COLOR_YUV2RGB_Y422, NVCV_COLOR_YUV2RGB_UYNV)
    case NVCV_COLOR_YUV2BGR_UYVY :  // NVCV_COLOR_YUV2BGR_UYVY  = 108 ( NVCV_COLOR_YUV2RGB_Y422, NVCV_COLOR_YUV2RGB_UYNV)
    // Conversion codes 109 (NVCV_COLOR_YUV2RGB_VYUY) and 110 (NVCV_COLOR_YUV2BGR_VYUY) not available.
    case NVCV_COLOR_YUV2RGBA_UYVY:  // NVCV_COLOR_YUV2RGBA_UYVY = 111 ( NVCV_COLOR_YUV2RGBA_Y422, NVCV_COLOR_YUV2RGBA_UYNV)
    case NVCV_COLOR_YUV2BGRA_UYVY:  // NVCV_COLOR_YUV2BGRA_UYVY = 112 ( NVCV_COLOR_YUV2BGRA_Y422, NVCV_COLOR_YUV2BGRA_UYNV)
        convertYUVtoRGB_422<T, false>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, false);
        break;

    // Conversion codes 113 (NVCV_COLOR_YUV2RGBA_VYUY) and 114 (NVCV_COLOR_YUV2BGRA_VYUY) not available.
    case NVCV_COLOR_YUV2RGB_YUY2 :  // NVCV_COLOR_YUV2RGB_YUY2  = 115 (NVCV_COLOR_YUV2RGB_YUYV, NVCV_COLOR_YUV2RGB_YUNV)
    case NVCV_COLOR_YUV2BGR_YUY2 :  // NVCV_COLOR_YUV2BGR_YUY2  = 116 (NVCV_COLOR_YUV2BGR_YUYV, NVCV_COLOR_YUV2BGR_YUNV)
    case NVCV_COLOR_YUV2RGBA_YUY2:  // NVCV_COLOR_YUV2RGBA_YUY2 = 119 (NVCV_COLOR_YUV2RGBA_YUYV, NVCV_COLOR_YUV2RGBA_YUNV)
    case NVCV_COLOR_YUV2BGRA_YUY2:  // NVCV_COLOR_YUV2BGRA_YUY2 = 120 (NVCV_COLOR_YUV2BGRA_YUYV, NVCV_COLOR_YUV2BGRA_YUNV)
        convertYUVtoRGB_422<T, true>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, false);
        break;

    case NVCV_COLOR_YUV2RGB_YVYU :  // NVCV_COLOR_YUV2RGB_YVYU  = 117
    case NVCV_COLOR_YUV2BGR_YVYU :  // NVCV_COLOR_YUV2BGR_YVYU  = 118
    case NVCV_COLOR_YUV2RGBA_YVYU:  // NVCV_COLOR_YUV2RGBA_YVYU = 121
    case NVCV_COLOR_YUV2BGRA_YVYU:  // NVCV_COLOR_YUV2BGRA_YVYU = 122
        convertYUVtoRGB_422<T, true>(refVec, srcVec, wdth, hght, imgs, dstRGBA, dstBGR, true);
        break;

    // Convert from YUV 4:2:2 family to grayscale.
    case NVCV_COLOR_YUV2GRAY_UYVY:  // NVCV_COLOR_YUV2GRAY_UYVY = 123 (NVCV_COLOR_YUV2GRAY_Y422, NVCV_COLOR_YUV2GRAY_UYNV)
        convertYUVtoGray_422<T, false>(refVec, srcVec, numPixels);
        break;

    case NVCV_COLOR_YUV2GRAY_YUY2:  // NVCV_COLOR_YUV2GRAY_YUY2 = 124 (NVCV_COLOR_YUV2GRAY_YVYU, NVCV_COLOR_YUV2GRAY_YUYV,
                                    //                                 NVCV_COLOR_YUV2GRAY_YUNV)
        convertYUVtoGray_422<T, true>(refVec, srcVec, numPixels);
        break;

    // RGB/BGA alpha premultiplication --> Conversion codes 125-126 not implemented.

    // Convert from RGB/BGR to YUV 4:2:0 family.
    case NVCV_COLOR_RGB2YUV_I420 :  // NVCV_COLOR_RGB2YUV_I420  = 127 (NVCV_COLOR_RGB2YUV_IYUV)
    case NVCV_COLOR_BGR2YUV_I420 :  // NVCV_COLOR_BGR2YUV_I420  = 128 (NVCV_COLOR_BGR2YUV_IYUV)
    case NVCV_COLOR_RGBA2YUV_I420:  // NVCV_COLOR_RGBA2YUV_I420 = 129 (NVCV_COLOR_RGBA2YUV_IYUV)
    case NVCV_COLOR_BGRA2YUV_I420:  // NVCV_COLOR_BGRA2YUV_I420 = 130 (NVCV_COLOR_BGRA2YUV_IYUV)
        convertRGBtoYUV_420<T>(refVec, srcVec, wdth, hght, imgs, srcRGBA, srcBGR, false);
        break;

    case NVCV_COLOR_RGB2YUV_YV12 :  // NVCV_COLOR_RGB2YUV_YV12  = 131
    case NVCV_COLOR_BGR2YUV_YV12 :  // NVCV_COLOR_BGR2YUV_YV12  = 132
    case NVCV_COLOR_RGBA2YUV_YV12:  // NVCV_COLOR_RGBA2YUV_YV12 = 133
    case NVCV_COLOR_BGRA2YUV_YV12:  // NVCV_COLOR_BGRA2YUV_YV12 = 134
        convertRGBtoYUV_420<T>(refVec, srcVec, wdth, hght, imgs, srcRGBA, srcBGR, true);
        break;

    // Edge-aware demosaicing to RGB/BGR --> Conversion codes 135-138 not implemented.
    // OpenCV COLORCVT_MAX               --> Conversion code  139     not implemented.

    // Convert RGB/BGR to YUV 4:2:0 family (two plane YUV; not in OpenCV).
    case NVCV_COLOR_RGB2YUV_NV12 :  // NVCV_COLOR_RGB2YUV_NV12  = 140
    case NVCV_COLOR_BGR2YUV_NV12 :  // NVCV_COLOR_BGR2YUV_NV12  = 141
    case NVCV_COLOR_RGBA2YUV_NV12:  // NVCV_COLOR_RGBA2YUV_NV12 = 144
    case NVCV_COLOR_BGRA2YUV_NV12:  // NVCV_COLOR_BGRA2YUV_NV12 = 145
        convertRGBtoNV12<T>(refVec, srcVec, wdth, hght, imgs, srcRGBA, srcBGR, false);
        break;

    case NVCV_COLOR_RGB2YUV_NV21 :  // NVCV_COLOR_RGB2YUV_NV21  = 142 (NVCV_COLOR_RGB2YUV420sp)
    case NVCV_COLOR_BGR2YUV_NV21 :  // NVCV_COLOR_BGR2YUV_NV21  = 143 (NVCV_COLOR_BGR2YUV420sp)
    case NVCV_COLOR_RGBA2YUV_NV21:  // NVCV_COLOR_RGBA2YUV_NV21 = 146 (NVCV_COLOR_RGBA2YUV420sp)
    case NVCV_COLOR_BGRA2YUV_NV21:  // NVCV_COLOR_BGRA2YUV_NV21 = 147 (NVCV_COLOR_BGRA2YUV420sp)
        convertRGBtoNV12<T>(refVec, srcVec, wdth, hght, imgs, srcRGBA, srcBGR, true);
        break;

    default:
        std::cerr << "**** ERROR: Color conversion not implemented for conversion code " << code << ". ****\n\n";
        success = false;
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

        constexpr uint maxErrCnt = 16;

        // Compare "gold" reference to computed output.
        if (dstFrmt == NVCV_IMAGE_FORMAT_HSV8 || dstFrmt == NVCV_IMAGE_FORMAT_HSVf32)
        {
            const bool   full  = (code == NVCV_COLOR_BGR2HSV_FULL || code == NVCV_COLOR_RGB2HSV_FULL);
            const double range = (sizeof(T) > 1) ? 360.0 : (full ? 256.0 : 180.0);

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

    {  23,  21, 63,  NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_GRAY2BGR,      0.0},
    {  21,  22, 63,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_COLOR_BGR2GRAY,      1.0},
    { 401, 202,  5,  NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_GRAY2RGB,      0.0},
    { 201, 402,  5,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_COLOR_RGB2GRAY,      1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_Y8_ER,    NVCV_COLOR_RGB2GRAY,      1.0},

    {  32,  21,  4,  NVCV_IMAGE_FORMAT_Y16,      NVCV_IMAGE_FORMAT_BGR16,    NVCV_COLOR_GRAY2BGR,      0.0},
    {  32,  21,  4,  NVCV_IMAGE_FORMAT_BGR16,    NVCV_IMAGE_FORMAT_Y16,      NVCV_COLOR_BGR2GRAY,      2.0},
    {  54,  66,  5,  NVCV_IMAGE_FORMAT_Y16,      NVCV_IMAGE_FORMAT_RGB16,    NVCV_COLOR_GRAY2RGB,      0.0},
    {  54,  66,  5,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_Y16,      NVCV_COLOR_RGB2GRAY,      2.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB16,    NVCV_IMAGE_FORMAT_Y16,      NVCV_COLOR_RGB2GRAY,      2.0},

    {  64,  21,  3,  NVCV_IMAGE_FORMAT_Yf32,     NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_GRAY2BGR,   ERR1_4},
    {  64,  21,  3,  NVCV_IMAGE_FORMAT_BGRf32,   NVCV_IMAGE_FORMAT_Yf32,     NVCV_COLOR_BGR2GRAY,   ERR1_4},
    { 121,  66,  5,  NVCV_IMAGE_FORMAT_Yf32,     NVCV_IMAGE_FORMAT_RGBf32,   NVCV_COLOR_GRAY2RGB,   ERR1_4},
    { 121,  66,  5,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_Yf32,     NVCV_COLOR_RGB2GRAY,   ERR1_4},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGBf32,   NVCV_IMAGE_FORMAT_Yf32,     NVCV_COLOR_RGB2GRAY,   ERR1_4},

    // Codes 9 to 39 are not implemented
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV,       1.0},
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_HSV2BGR,       1.0},
    {  55, 257,  4,  NVCV_IMAGE_FORMAT_HSV8,    NVCV_IMAGE_FORMAT_BGRA8,     NVCV_COLOR_HSV2BGR,       1.0},
    { 366,  14,  5,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_RGB2HSV,       1.0},
    { 366,  14,  5,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_HSV2RGB,       1.0},
    {4096,4096,  1,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_RGB2HSV,       1.0},
    {2880,4096,  1,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_HSV2RGB,       1.0},
    {4096,4096, 91,  NVCV_IMAGE_FORMAT_BGR8,     NVCV_IMAGE_FORMAT_HSV8,     NVCV_COLOR_BGR2HSV,       1.0},
    {4096,4096, 92,  NVCV_IMAGE_FORMAT_HSV8,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_HSV2BGR,       1.0},

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
    {4096,4096, 93,  NVCV_IMAGE_FORMAT_RGB8,     NVCV_IMAGE_FORMAT_NV21,     NVCV_COLOR_RGB2YUV_NV21,  1.0},
    {4096,4096, 94,  NVCV_IMAGE_FORMAT_NV21,     NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_YUV2BGR_NV21,  2.0},

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
    ASSERT_EQ(nvcvImageFormatGetPlaneDataType(srcFrmt, 0, &dataType), NVCV_SUCCESS);

    switch (dataType)
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
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(srcFormat, 0, &nvcvDataType));

    NVCVColorConversionCode src2dstCode{GetParamValue<5>()};
    NVCVColorConversionCode dst2srcCode{GetParamValue<6>()};

    double maxDiff{GetParamValue<7>()};

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

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
            generateRandVec(reinterpret_cast<uint8_t *>(srcVec[i].data()), srcVec[i].size(), rng);
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

TEST(OpCvtColor_negative, mismatch_shape)
{
    nvcv::Tensor tensorY8   = util::CreateTensor(2, 224, 224, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_Y8});
    nvcv::Tensor tensorHSV8 = util::CreateTensor(2, 224, 224, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_HSV8});
    nvcv::Tensor tensorBGR8 = util::CreateTensor(5, 224, 224, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8});
    nvcv::Tensor tensorRGB8 = util::CreateTensor(2, 224, 224, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB8});

    // run operator
    cvcuda::CvtColor cvtColorOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvtColorOp(nullptr, tensorY8, tensorBGR8, NVCV_COLOR_GRAY2BGR); }));

    // reserved conversion invalid too
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvtColorOp(nullptr, tensorBGR8, tensorY8, NVCV_COLOR_BGR2GRAY); }));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvtColorOp(nullptr, tensorHSV8, tensorBGR8, NVCV_COLOR_HSV2BGR); }));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvtColorOp(nullptr, tensorBGR8, tensorRGB8, NVCV_COLOR_BGR2RGB); }));
}

TEST(OpCvtColor_negative, invalid_shape_BGR_to_YUV420xp)
{
    std::vector<nvcv::Tensor> srcTensors{util::CreateTensor(1, 7, 8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8}),
                                         util::CreateTensor(1, 8, 6, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8}),
                                         util::CreateTensor(1, 8, 8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGRf16}),
                                         util::CreateTensor(1, 16, 16, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8})};

    nvcv::Tensor dstTensor = nvcv::Tensor(
        {
            {8, 8, 1},
            "HWC"
    },
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV21}.planeDataType(0).channelType(0));

    // run operator
    cvcuda::CvtColor cvtColorOp;
    for (auto &srcTensor : srcTensors)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_BGR2YUV_YV12); }));
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
    for (auto &srcTensor : srcTensors)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_YUV2BGR_YV12); }));
    }

    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall(
            [&] { cvtColorOp(nullptr, srcTensor_1, dstTensor_1, NVCV_COLOR_YUV2BGR_YV12); })); // incalid output channel
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

    nvcv::Tensor dstTensor = nvcv::Tensor(
        {
            {120, 40, 3},
            "HWC"
    },
        nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8}.planeDataType(0).channelType(0));

    // run operator
    cvcuda::CvtColor cvtColorOp;
    for (auto &srcTensor : srcTensors)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_YUV2BGR_UYVY); }));
    }
}

TEST(OpCvtColor_negative, invalid_shape_YUV422_to_BGR_invalid_out)
{
    nvcv::Tensor srcTensor = nvcv::Tensor(
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
    for (auto &dstTensor : dstTensors)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { cvtColorOp(nullptr, srcTensor, dstTensor, NVCV_COLOR_YUV2BGR_UYVY); }));
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpCvtColor_negative,
test::ValueList<int, int, int, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode>
{
    //  W,   H,  N,  Input Format,              Output Format,              Conversion Code
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_Y8,      NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_BGR2BGRA}, // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGRAf32,  NVCV_COLOR_BGR2BGRA}, // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGRA8p,   NVCV_COLOR_BGR2BGRA}, // mismatch format
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8p,   NVCV_IMAGE_FORMAT_BGRA8p,   NVCV_COLOR_BGR2BGRA}, // invalid format
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_BGR2BGRA}, // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_GRAY2BGR}, // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_Y8,      NVCV_IMAGE_FORMAT_BGRf32,   NVCV_COLOR_GRAY2BGR}, // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_Y8,      NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_GRAY2BGR}, // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRA8,   NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_BGR2GRAY}, // invalid input channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf32,  NVCV_IMAGE_FORMAT_Y8,       NVCV_COLOR_BGR2GRAY}, // mismatch data type
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf16,  NVCV_IMAGE_FORMAT_BGRAf16,  NVCV_COLOR_BGR2BGRA}, // f16 type not allowed to add alpha
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGR8,    NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_BGR2GRAY}, // invalid output channel
    {   8,   8,  3,  NVCV_IMAGE_FORMAT_BGRf64,  NVCV_IMAGE_FORMAT_Yf64,    NVCV_COLOR_BGR2GRAY}, // unsupported data type
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
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

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

#undef NVCV_IMAGE_FORMAT_BGRS32
#undef NVCV_IMAGE_FORMAT_RGBS32
#undef NVCV_IMAGE_FORMAT_BGRAS32
#undef NVCV_IMAGE_FORMAT_RGBAS32
#undef NVCV_IMAGE_FORMAT_YUVf32
#undef NVCV_IMAGE_FORMAT_Yf32
#undef NVCV_IMAGE_FORMAT_HSVf32

#undef NVCV_IMAGE_FORMAT_BGRS64
#undef NVCV_IMAGE_FORMAT_RGBS64
#undef NVCV_IMAGE_FORMAT_BGRAS64
#undef NVCV_IMAGE_FORMAT_RGBAS64
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
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

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
              nvcv::ProtectCall([&] { cvtColorOp(nullptr, batchSrc, batchDst, src2dstCode); }));
}

#undef VEC_EXPECT_NEAR
