/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Definitions.hpp"

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <cvcuda/Types.h> // for NVCVInterpolationType, NVCVChannelManip, etc.
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace gt    = ::testing;
namespace test  = nvcv::test;
namespace cuda  = nvcv::cuda;
namespace ttype = test::type;

static std::default_random_engine randEng(std::random_device{}());

template<typename T>
using uniform_dist
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

inline NVCVChannelManip ChannelManip(nvcv::ImageFormat srcFormat, nvcv::ImageFormat dstFormat)
{
    const int           srcChannels = srcFormat.numChannels();
    const nvcv::Swizzle srcSwizzle  = srcFormat.swizzle();
    const nvcv::Swizzle dstSwizzle  = dstFormat.swizzle();

    NVCVChannelManip manip = NVCV_CHANNEL_NO_OP;

    if (srcChannels > 2 && srcSwizzle != dstSwizzle)
    {
        int  srcSwap = static_cast<int>(srcSwizzle), dstSwap = static_cast<int>(dstSwizzle);
        bool srcRGB = (srcSwap == NVCV_SWIZZLE_XYZ0 || srcSwap == NVCV_SWIZZLE_XYZW || srcSwap == NVCV_SWIZZLE_XYZ1),
             srcBGR = (srcSwap == NVCV_SWIZZLE_ZYX0 || srcSwap == NVCV_SWIZZLE_ZYXW || srcSwap == NVCV_SWIZZLE_ZYX1);
        bool dstRGB = (dstSwap == NVCV_SWIZZLE_XYZ0 || dstSwap == NVCV_SWIZZLE_XYZW || dstSwap == NVCV_SWIZZLE_XYZ1),
             dstBGR = (dstSwap == NVCV_SWIZZLE_ZYX0 || dstSwap == NVCV_SWIZZLE_ZYXW || dstSwap == NVCV_SWIZZLE_ZYX1);
        bool swapRB = ((srcRGB && dstBGR) || (srcBGR && dstRGB));

        if (swapRB && srcChannels == 3)
        {
            manip = NVCV_CHANNEL_REVERSE;
        }
    }
    return manip;
}

// clang-format off

template<typename DstT, typename SrcT>
void ResizeCropConvert(      DstT *dst, NVCVSize2D dstSize, nvcv::ImageFormat dstFrmt,
                       const SrcT *src, NVCVSize2D srcSize, nvcv::ImageFormat srcFrmt,
                       int numImages, NVCVSize2D newSize, int2 crop, NVCVInterpolationType interp,
                       const NVCVChannelManip manip, float scale, float offset, bool srcCast = true)
{
    int channels  = dstFrmt.numChannels();
    int srcPlanes = srcFrmt.numPlanes();
    int dstPlanes = dstFrmt.numPlanes();

    size_t srcIncrX = channels / srcPlanes; // 1 if planar; channels if not.
    size_t dstIncrX = channels / dstPlanes; // 1 if planar; channels if not.
    size_t srcIncrY = srcIncrX * srcSize.w;
    size_t dstIncrY = dstIncrX * dstSize.w;
    size_t srcIncrC = (srcPlanes > 1 ? srcSize.w * srcSize.h : 1);
    size_t dstIncrC = (dstPlanes > 1 ? dstSize.w * dstSize.h : 1);
    size_t srcIncrN = srcSize.w * srcSize.h * channels;
    size_t dstIncrN = dstSize.w * dstSize.h * channels;

    int mapC[4] = {0, 1, 2, 3};

    if (manip == NVCV_CHANNEL_REVERSE)
    {
        for (int c = 0; c < channels; ++c) mapC[c] = channels - c - 1;
    }

    float scaleW = static_cast<float>(srcSize.w) / newSize.w;
    float scaleH = static_cast<float>(srcSize.h) / newSize.h;

    for (int i = 0; i < numImages; i++)
    {
        const SrcT *srcBase = src + i * srcIncrN;
        DstT       *dstBase = dst + i * dstIncrN;

        for (int dy = 0; dy < dstSize.h; dy++)
        {
            DstT *dstRow = dstBase + dy * dstIncrY;

            for (int dx = 0; dx < dstSize.w; dx++)
            {
                DstT *dstPtr = dstRow + dx * dstIncrX;

                if (interp == NVCV_INTERP_NEAREST)
                {
                    int sx = std::floor(scaleW * (dx + crop.x + 0.5f));
                    int sy = std::floor(scaleH * (dy + crop.y + 0.5f));

                    const SrcT *src0 = srcBase + sy * srcIncrY + sx * srcIncrX;

                    for (int c = 0; c < channels; c++)
                    {
                        dstPtr[mapC[c] * dstIncrC] = cuda::SaturateCast<DstT>(scale * src0[c * srcIncrC] + offset);
                    }
                }
                else if (interp == NVCV_INTERP_LINEAR)
                {
                    float fx = scaleW * (dx + crop.x + 0.5f) - 0.5f;
                    float fy = scaleH * (dy + crop.y + 0.5f) - 0.5f;

                    int sx0 = std::floor(fx);
                    int sy0 = std::floor(fy);
                    int sx1 = std::min(sx0 + 1, srcSize.w - 1);
                    int sy1 = std::min(sy0 + 1, srcSize.h - 1);

                    fx -= sx0;
                    fy -= sy0;

                    sx0 = std::max(0, sx0);
                    sy0 = std::max(0, sy0);

                    float wghtX[2] = {1 - fx, fx};
                    float wghtY[2] = {1 - fy, fy};

                    const size_t x0 = sx0 * srcIncrX;
                    const size_t x1 = sx1 * srcIncrX;

                    const SrcT *src0 = srcBase + sy0 * srcIncrY;
                    const SrcT *src1 = srcBase + sy1 * srcIncrY;

                    for (int c = 0; c < channels; c++)
                    {
                        const size_t xc = c * srcIncrC;

                        float val = src0[x0 + xc] * wghtY[0] * wghtX[0]
                                  + src0[x1 + xc] * wghtY[0] * wghtX[1]
                                  + src1[x0 + xc] * wghtY[1] * wghtX[0]
                                  + src1[x1 + xc] * wghtY[1] * wghtX[1];

                        val = scale * (srcCast ? cuda::SaturateCast<SrcT>(val) : val) + offset;

                        dstPtr[mapC[c] * dstIncrC] = cuda::SaturateCast<DstT>(val);
                    }
                }
            }
        }
    }
}

template<typename DstT, typename SrcT>
void ResizeCropConvert(      std::vector<DstT> &dst, NVCVSize2D dstSize, nvcv::ImageFormat dstFrmt,
                       const std::vector<SrcT> &src, NVCVSize2D srcSize, nvcv::ImageFormat srcFrmt,
                       int numImages, NVCVSize2D newSize, int2 crop, NVCVInterpolationType interp,
                       const NVCVChannelManip manip, float scale, float offset, bool srcCast = true)
{
    ResizeCropConvert(dst.data(), dstSize, dstFrmt, src.data(), srcSize, srcFrmt,
                      numImages, newSize, crop, interp, manip, scale, offset, srcCast);

}

template<typename DstT, typename SrcT>
void ResizeCropConvert(                  DstT  *dst, NVCVSize2D dstSize, nvcv::ImageFormat dstFrmt,
                       const std::vector<SrcT> &src, NVCVSize2D srcSize, nvcv::ImageFormat srcFrmt,
                       int numImages, NVCVSize2D newSize, int2 crop, NVCVInterpolationType interp,
                       const NVCVChannelManip manip, float scale, float offset, bool srcCast = true)
{
    ResizeCropConvert(dst, dstSize, dstFrmt, src.data(), srcSize, srcFrmt,
                      numImages, newSize, crop, interp, manip, scale, offset, srcCast);

}

// clang-format on

template<typename T>
void fillVec(std::vector<T> &vec, const NVCVSize2D size, const nvcv::ImageFormat frmt, size_t offst = 0)
{
    int    planes   = frmt.numPlanes();
    int    channels = frmt.numChannels();
    size_t incrX    = channels / planes; // 1 if planar; dstChannels if not.
    size_t incrY    = incrX * size.w;
    size_t incrC    = (planes > 1 ? size.w * size.h : 1);

    for (int y = 0; y < size.h; y++)
    {
        size_t yIncr = offst + y * incrY;

        for (int x = 0; x < size.w; x++)
        {
            size_t xIncr = yIncr + x * incrX;

            for (int c = 0; c < channels; c++)
            {
                vec[xIncr + c * incrC] = static_cast<T>((x + y + c) & 255);
            }
        }
    }
}

#define _SHAPE(w, h, n) (int3{w, h, n})

// clang-format off

#define _TEST_ROW(SrcShape, ResizeDim, Interp, DstSize, CropPos, Scale, Offset, SrcFrmt, DstFrmt, SrcType, DstType, SrcCast) \
    ttype::Types<ttype::Value<SrcShape>, ttype::Value<ResizeDim>, ttype::Value<Interp>, ttype::Value<DstSize>, \
                 ttype::Value<CropPos>, ttype::Value<Scale>, ttype::Value<Offset>, \
                 ttype::Value<SrcFrmt>, ttype::Value<DstFrmt>, SrcType, DstType, ttype::Value<SrcCast> >

NVCV_TYPED_TEST_SUITE(
    OpResizeCropConvertReformat, ttype::Types<
    // Test cases: RGB (interleaved) -> BGR (planar); linear interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE(   8,    8,  1), int2(  8,   8), NVCV_INTERP_LINEAR, int2(   6,   6), int2(  1,   1), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  0
    _TEST_ROW(_SHAPE(   8,    8,  1), int2( 16,  16), NVCV_INTERP_LINEAR, int2(  12,  12), int2(  2,   2), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  1
    _TEST_ROW(_SHAPE(  42,   48,  1), int2( 23,  24), NVCV_INTERP_LINEAR, int2(  15,  13), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  2
    _TEST_ROW(_SHAPE(  42,   40,  3), int2( 21,  20), NVCV_INTERP_LINEAR, int2(  17,  13), int2(  1,   1), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  3
    _TEST_ROW(_SHAPE(  21,   21,  5), int2( 42,  42), NVCV_INTERP_LINEAR, int2(  32,  32), int2( 10,  10), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  4
    _TEST_ROW(_SHAPE( 113,   12,  7), int2( 12,  36), NVCV_INTERP_LINEAR, int2(   7,  13), int2(  3,  11), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  5
    _TEST_ROW(_SHAPE(  17,  151,  7), int2( 48,  16), NVCV_INTERP_LINEAR, int2(  32,  16), int2(  4,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  6
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  7
    _TEST_ROW(_SHAPE(1080, 1920, 13), int2(800, 600), NVCV_INTERP_LINEAR, int2( 640, 480), int2(101,  64), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  8
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), //  9
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8p,    uchar3, uint8_t, false), // 10
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8p,    uchar3, uint8_t, false), // 11

    // Test cases: RGB (interleaved) -> RGB (planar); linear interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32p,  uchar3, float  , false), // 12
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32p,  uchar3, float  , false), // 13
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p,    uchar3, uint8_t, false), // 14
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p,    uchar3, uint8_t, false), // 15

    // Test cases: BGR (interleaved) -> RGB (planar); linear interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p,  uchar3, float  , false), // 16
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p,  uchar3, float  , false), // 17
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,    uchar3, uint8_t, false), // 18
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,    uchar3, uint8_t, false), // 19

    // Test cases: BGR (interleaved) -> BGR (planar); linear interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), // 20
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float  , false), // 21
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGR8p,    uchar3, uint8_t, false), // 22
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGR8p,    uchar3, uint8_t, false), // 23

    // Test cases: RGB (interleaved) -> BGR (interleaved); linear interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE(   8,    8,  1), int2(  8,   8), NVCV_INTERP_LINEAR, int2(   6,   6), int2(  1,   1), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3 , false), // 24
    _TEST_ROW(_SHAPE(   8,    8,  1), int2( 16,  16), NVCV_INTERP_LINEAR, int2(  12,  12), int2(  2,   2), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3 , false), // 25
    _TEST_ROW(_SHAPE( 113,   12,  7), int2( 12,  36), NVCV_INTERP_LINEAR, int2(   7,  13), int2(  3,  11), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3 , false), // 26
    _TEST_ROW(_SHAPE(  17,  151,  7), int2( 48,  16), NVCV_INTERP_LINEAR, int2(  32,  16), int2(  4,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3 , false), // 27
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3 , false), // 28
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3 , false), // 29
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8,     uchar3, uchar3 , false), // 30
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8,     uchar3, uchar3 , false), // 31

    // Test cases: RGB (interleaved) -> RGB (interleaved); linear interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32,   uchar3, float3 , false), // 32
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32,   uchar3, float3 , false), // 33
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8,     uchar3, uchar3 , false), // 34
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8,     uchar3, uchar3 , false), // 35

    // Test cases: BGR (interleaved) -> RGB (interleaved); linear interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32,   uchar3, float3 , false), // 36
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32,   uchar3, float3 , false), // 37
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8,     uchar3, uchar3 , false), // 38
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8,     uchar3, uchar3 , false), // 39

    // Test cases: RGB (interleaved) -> BGR (planar); nearest-neighbor interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation   ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE(   8,    8,  1), int2(  8,   8), NVCV_INTERP_NEAREST, int2(   6,   6), int2(  1,   1), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float  , false), // 40
    _TEST_ROW(_SHAPE(   8,    8,  5), int2( 16,  16), NVCV_INTERP_NEAREST, int2(  12,  12), int2(  2,   2), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float  , false), // 41
    _TEST_ROW(_SHAPE(  42,   48,  1), int2( 23,  24), NVCV_INTERP_NEAREST, int2(  15,  13), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float  , false), // 42
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_NEAREST, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float  , false), // 43
    _TEST_ROW(_SHAPE(1080, 1920, 13), int2(800, 600), NVCV_INTERP_NEAREST, int2( 640, 480), int2(101,  64), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float  , false), // 44
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float  , false), // 45
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_NEAREST, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8p,   uchar3, uint8_t, false), // 46
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8p,   uchar3, uint8_t, false), // 47

    // Test cases: BGR (interleaved) -> RGB (planar); nearest-neighbor interpolation; float and uchar output.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation   ,   dest.(w, h)  , crop(x, y), scale, offst,   source format   ,     destination format  , src type, dst type, src cast
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_NEAREST, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float  , false), // 48
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float  , false), // 49
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_NEAREST, int2( 412, 336), int2(  0,   0), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,   uchar3, uint8_t, false), // 50
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16), 1, 0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,   uchar3, uint8_t, false), // 51

    // Test cases: Rescaling.
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  ,   crop(x, y)  , scale, offst,     source format     ,     destination format , src type, dst type, src cast
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR,  int2( 412, 336), int2(  0,   0), 1/127.5, -1, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float  , false), // 52
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16),   2,   -255, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float  , false), // 53
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16),  -1,    255, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8,    uchar3, uchar3 , false), // 54
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR,  int2( 412, 336), int2(  0,   0),   0.5,    0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,   uchar3, uint8_t, false), // 55
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16),   2,      0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,   uchar3, uint8_t, false), // 56

    // Test cases: Source cast true (with and w/o rescaling).
    //             source(w, h, n)  ,  resize(w, h) ,    interpolation  ,   dest.(w, h)  ,   crop(x, y)  , scale, offst,     source format     ,     destination format , src type, dst type, src cast
    _TEST_ROW(_SHAPE( 353,  450,  3), int2(256, 256), NVCV_INTERP_LINEAR,  int2( 224, 224), int2( 16,  16),   1,      0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float  , true),  // 57
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR,  int2( 412, 336), int2(  0,   0), 1/127.5, -1, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float  , true),  // 58
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16),   2,   -255, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float  , true),  // 59
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16),  -1,    255, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8,    uchar3, uchar3 , true),  // 60
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR,  int2( 412, 336), int2(  0,   0),   1,      0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,   uchar3, uint8_t, true),  // 61
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16),   1,      0, NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,   uchar3, uint8_t, true)   // 62
>);
#undef _TEST_ROW

// clang-format on

TYPED_TEST(OpResizeCropConvertReformat, tensor_correct_output)
{
    int3 srcShape = ttype::GetValue<TypeParam, 0>;
    int2 resize   = ttype::GetValue<TypeParam, 1>;

    NVCVInterpolationType interp = ttype::GetValue<TypeParam, 2>;

    int2 cropDim = ttype::GetValue<TypeParam, 3>;
    int2 cropPos = ttype::GetValue<TypeParam, 4>;

    float scale  = ttype::GetValue<TypeParam, 5>;
    float offset = ttype::GetValue<TypeParam, 6>;

    nvcv::ImageFormat srcFormat{ttype::GetValue<TypeParam, 7>};
    nvcv::ImageFormat dstFormat{ttype::GetValue<TypeParam, 8>};

    using SrcVT = typename ttype::GetType<TypeParam, 9>;
    using DstVT = typename ttype::GetType<TypeParam, 10>;
    using SrcBT = typename cuda::BaseType<SrcVT>;
    using DstBT = typename cuda::BaseType<DstVT>;

    bool srcCast = ttype::GetValue<TypeParam, 11>;

    int srcW = srcShape.x;
    int srcH = srcShape.y;
    int dstW = cropDim.x;
    int dstH = cropDim.y;

    int numImages   = srcShape.z;
    int srcChannels = srcFormat.numChannels();
    int dstChannels = dstFormat.numChannels();
    int srcPlanes   = srcFormat.numPlanes();
    int dstPlanes   = dstFormat.numPlanes();
    int srcPixElems = srcChannels / srcPlanes;
    int dstPixElems = dstChannels / dstPlanes;

    ASSERT_LE(srcChannels, 4);
    ASSERT_EQ(srcChannels, dstChannels);

    NVCVChannelManip manip = ChannelManip(srcFormat, dstFormat);

    // Create input and output tensors.
    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(numImages, srcW, srcH, srcFormat);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(numImages, dstW, dstH, dstFormat);

    auto src = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dst = dstTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*src);
    ASSERT_TRUE(srcAccess);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dst);
    ASSERT_TRUE(dstAccess);

    int srcRowElems = srcPixElems * srcW;
    int dstRowElems = dstPixElems * dstW;

    size_t srcElems = (size_t)srcRowElems * (size_t)srcH * (size_t)srcPlanes * (size_t)numImages;
    size_t dstElems = (size_t)dstRowElems * (size_t)dstH * (size_t)dstPlanes * (size_t)numImages;

    NVCVSize2D srcSize{srcW, srcH};
    NVCVSize2D newSize{resize.x, resize.y};
    NVCVSize2D dstSize{dstW, dstH};

    size_t srcPitch = srcW * sizeof(SrcVT);
    size_t dstPitch = dstW * sizeof(DstVT);

    std::vector<SrcBT> srcVec(srcElems);
    std::vector<DstBT> refVec(dstElems);

    // Populate source tensor.
    for (int n = 0; n < numImages; n++)
    {
        fillVec(srcVec, srcSize, srcFormat, n * (size_t)srcRowElems * (size_t)srcH * (size_t)srcPlanes);
    }

    // Copy source tensor to device.
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(src->basePtr(), srcAccess->rowStride(), srcVec.data(), srcPitch, srcPitch,
                                        srcH * srcPlanes * numImages, cudaMemcpyHostToDevice));

    // Generate "gold" result for image and place in reference vector.
    ResizeCropConvert(refVec, dstSize, dstFormat, srcVec, srcSize, srcFormat, numImages, newSize, cropPos, interp,
                      manip, scale, offset, srcCast);

    // Run fused ResizeCropConvertReformat operator.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::ResizeCropConvertReformat resizeCrop;
    EXPECT_NO_THROW(resizeCrop(stream, srcTensor, dstTensor, newSize, interp, cropPos, manip, scale, offset, srcCast));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Copy destination tensor back to host.
    std::vector<DstBT> dstVec(dstElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dstVec.data(), dstPitch, dst->basePtr(), dstAccess->rowStride(), dstPitch,
                                        dstH * dstPlanes * numImages, cudaMemcpyDeviceToHost));

    // Compare "gold" reference to computed output.
    VEC_EXPECT_NEAR(refVec, dstVec, 1);
}

TYPED_TEST(OpResizeCropConvertReformat, varshape_correct_output)
{
    int3 srcShape = ttype::GetValue<TypeParam, 0>;
    int2 resize   = ttype::GetValue<TypeParam, 1>;

    NVCVInterpolationType interp = ttype::GetValue<TypeParam, 2>;

    int2 cropDim = ttype::GetValue<TypeParam, 3>;
    int2 cropPos = ttype::GetValue<TypeParam, 4>;

    float scale  = ttype::GetValue<TypeParam, 5>;
    float offset = ttype::GetValue<TypeParam, 6>;

    nvcv::ImageFormat srcFormat{ttype::GetValue<TypeParam, 7>};
    nvcv::ImageFormat dstFormat{ttype::GetValue<TypeParam, 8>};

    using SrcVT = typename ttype::GetType<TypeParam, 9>;
    using DstVT = typename ttype::GetType<TypeParam, 10>;
    using SrcBT = typename cuda::BaseType<SrcVT>;
    using DstBT = typename cuda::BaseType<DstVT>;

    bool srcCast = ttype::GetValue<TypeParam, 11>;

    int srcW = srcShape.x;
    int srcH = srcShape.y;
    int dstW = cropDim.x;
    int dstH = cropDim.y;

    int numImages   = srcShape.z;
    int srcChannels = srcFormat.numChannels();
    int dstChannels = dstFormat.numChannels();
    int srcPlanes   = srcFormat.numPlanes();
    int dstPlanes   = dstFormat.numPlanes();
    int srcPixElems = srcChannels / srcPlanes;
    int dstPixElems = dstChannels / dstPlanes;

    ASSERT_LE(srcChannels, 4);
    ASSERT_EQ(srcChannels, dstChannels);

    NVCVChannelManip manip = ChannelManip(srcFormat, dstFormat);

    std::vector<nvcv::Image> srcImg;

    uniform_dist<SrcBT> randVal(std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::min : SrcBT{0},
                                std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::max : SrcBT{1});

    std::uniform_int_distribution<int> randW(srcW * 0.8, srcW * 1.2);
    std::uniform_int_distribution<int> randH(srcH * 0.8, srcH * 1.2);

    int dstRowElems = dstPixElems * dstW;

    size_t refIncr  = (size_t)dstRowElems * (size_t)dstH * (size_t)dstPlanes;
    size_t dstElems = refIncr * (size_t)numImages;

    NVCVSize2D newSize{resize.x, resize.y};
    NVCVSize2D dstSize{dstW, dstH};

    std::vector<DstBT> refVec(dstElems);

    size_t dstPitch = dstW * sizeof(DstVT);

    for (int i = 0; i < numImages; ++i)
    {
        int imgW = (interp ? randW(randEng) : srcW);
        int imgH = (interp ? randH(randEng) : srcH);

        srcImg.emplace_back(NVCVSize2D{imgW, imgH}, srcFormat);

        auto srcData = srcImg[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(srcData);

        int imgRowElems = srcPixElems * imgW;

        size_t imgPitch = imgW * sizeof(SrcVT);
        size_t imgElems = (size_t)imgRowElems * (size_t)imgH * (size_t)srcPlanes;

        NVCVSize2D imgSize{imgW, imgH};

        std::vector<SrcBT> imgVec(imgElems);

        // Populate image tensor .
        fillVec(imgVec, imgSize, srcFormat);

        // Generate "gold" result for image and place in reference image plane.
        DstBT *refPlane = refVec.data() + i * refIncr;

        ResizeCropConvert(refPlane, dstSize, dstFormat, imgVec, imgSize, srcFormat, 1, newSize, cropPos, interp, manip,
                          scale, offset, srcCast);

        // Copy source tensor to device.
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, imgVec.data(),
                                            imgPitch, imgPitch, imgH * srcPlanes, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape src(numImages);

    src.pushBack(srcImg.begin(), srcImg.end());

    // Create output tensor.
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(numImages, dstW, dstH, dstFormat);

    auto dst = dstTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(dst, nullptr);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dst);
    ASSERT_TRUE(dstAccess);

    // Run fused ResizeCropConvertReformat operator.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::ResizeCropConvertReformat resizeCrop;
    EXPECT_NO_THROW(resizeCrop(stream, src, dstTensor, newSize, interp, cropPos, manip, scale, offset, srcCast));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Copy destination tensor back to host.
    std::vector<DstBT> dstVec(dstElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dstVec.data(), dstPitch, dst->basePtr(), dstAccess->rowStride(), dstPitch,
                                        dstH * dstPlanes * numImages, cudaMemcpyDeviceToHost));

    // Compare "gold" reference to computed output.
    VEC_EXPECT_NEAR(refVec, dstVec, 1);
}

#define _TEST_ROW(Interp, inputBatch, outputBatch, srcFmt, dstFmt, DstSize, CropPos, ResizeDim, SrcType, DstType,      \
                  returnCode)                                                                                          \
    ttype::Types<ttype::Value<Interp>, ttype::Value<inputBatch>, ttype::Value<outputBatch>, ttype::Value<srcFmt>,      \
                 ttype::Value<dstFmt>, ttype::Value<DstSize>, ttype::Value<CropPos>, ttype::Value<ResizeDim>, SrcType, \
                 DstType, ttype::Value<returnCode>>

// clang-format off
NVCV_TYPED_TEST_SUITE(OpResizeCropConvertReformat_Negative,
ttype::Types<
    // Interpolation, input batch size, output batch size, src fmt, dst fmt, crop dim, crop pos
    // invalid Interpolation
    _TEST_ROW(NVCV_INTERP_CUBIC, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    _TEST_ROW(NVCV_INTERP_AREA, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    _TEST_ROW(NVCV_INTERP_LANCZOS, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    // different input/output batch size
    _TEST_ROW(NVCV_INTERP_LINEAR, 1, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 1, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    // different channels
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRA8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_NOT_COMPATIBLE),
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRA8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_NOT_COMPATIBLE),
    // not equal to 3 channels
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_BGRA8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_NOT_COMPATIBLE),
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_BGRA8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_NOT_COMPATIBLE),
    // input is not uchar
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGBf32, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_NOT_COMPATIBLE),
    // output is not uchar/float
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf16, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_NOT_COMPATIBLE),
    // invalid Crop Range
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(-1, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, -1), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(32, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 32), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT),
    // invalid input layout
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, 0), NVCVSize2D(16, 16), uchar3, uint8_t, NVCV_ERROR_NOT_COMPATIBLE),
    // invalid resize dim
    _TEST_ROW(NVCV_INTERP_LINEAR, 2, 2, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, int2(4, 4), int2(0, 0), NVCVSize2D(0, 0), uchar3, uint8_t, NVCV_ERROR_INVALID_ARGUMENT)
>);
// clang-format on

#undef _TEST_ROW

TEST(OpResizeCropConvertReformat_Negative, createWillNullPtr)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaResizeCropConvertReformatCreate(nullptr));
}

TYPED_TEST(OpResizeCropConvertReformat_Negative, infer_negative_parameter)
{
    NVCVInterpolationType interp = ttype::GetValue<TypeParam, 0>;

    int inputBatchSize  = ttype::GetValue<TypeParam, 1>;
    int outputBatchSize = ttype::GetValue<TypeParam, 2>;

    nvcv::ImageFormat srcFormat{ttype::GetValue<TypeParam, 3>};
    nvcv::ImageFormat dstFormat{ttype::GetValue<TypeParam, 4>};

    int2       cropDim   = ttype::GetValue<TypeParam, 5>;
    int2       cropPos   = ttype::GetValue<TypeParam, 6>;
    NVCVSize2D resizeDim = ttype::GetValue<TypeParam, 7>;

    using SrcVT = typename ttype::GetType<TypeParam, 8>;
    using DstVT = typename ttype::GetType<TypeParam, 9>;
    using SrcBT = typename cuda::BaseType<SrcVT>;
    using DstBT = typename cuda::BaseType<DstVT>;

    NVCVStatus expectedReturnCode = ttype::GetValue<TypeParam, 10>;

    // Resize to 16 * 16 then crop
    int srcW = 32;
    int srcH = 32;
    int dstW = cropDim.x;
    int dstH = cropDim.y;

    NVCVChannelManip manip = ChannelManip(srcFormat, dstFormat);

    // Create input and output tensors.
    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(inputBatchSize, srcW, srcH, srcFormat);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(outputBatchSize, dstW, dstH, dstFormat);

    cvcuda::ResizeCropConvertReformat resizeCrop;
    EXPECT_EQ(expectedReturnCode,
              nvcv::ProtectCall([&] { resizeCrop(nullptr, srcTensor, dstTensor, resizeDim, interp, cropPos, manip); }));
}

TYPED_TEST(OpResizeCropConvertReformat_Negative, varshape_infer_negative_parameter)
{
    NVCVInterpolationType interp = ttype::GetValue<TypeParam, 0>;

    int inputBatchSize  = ttype::GetValue<TypeParam, 1>;
    int outputBatchSize = ttype::GetValue<TypeParam, 2>;

    nvcv::ImageFormat srcFormat{ttype::GetValue<TypeParam, 3>};
    nvcv::ImageFormat dstFormat{ttype::GetValue<TypeParam, 4>};

    int2       cropDim   = ttype::GetValue<TypeParam, 5>;
    int2       cropPos   = ttype::GetValue<TypeParam, 6>;
    NVCVSize2D resizeDim = ttype::GetValue<TypeParam, 7>;

    using SrcVT = typename ttype::GetType<TypeParam, 8>;
    using DstVT = typename ttype::GetType<TypeParam, 9>;
    using SrcBT = typename cuda::BaseType<SrcVT>;
    using DstBT = typename cuda::BaseType<DstVT>;

    NVCVStatus expectedReturnCode = ttype::GetValue<TypeParam, 10>;

    std::vector<nvcv::Image> srcImg;

    int srcW = 32;
    int srcH = 32;
    int dstW = cropDim.x;
    int dstH = cropDim.y;

    NVCVChannelManip manip = ChannelManip(srcFormat, dstFormat);

    uniform_dist<SrcBT> randVal(std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::min : SrcBT{0},
                                std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::max : SrcBT{1});

    std::uniform_int_distribution<int> randW(srcW * 0.8, srcW * 1.2);
    std::uniform_int_distribution<int> randH(srcH * 0.8, srcH * 1.2);

    for (int i = 0; i < inputBatchSize; ++i)
    {
        int imgW = (interp ? randW(randEng) : srcW);
        int imgH = (interp ? randH(randEng) : srcH);

        srcImg.emplace_back(nvcv::Size2D{imgW, imgH}, srcFormat);
    }

    nvcv::ImageBatchVarShape src(inputBatchSize);
    src.pushBack(srcImg.begin(), srcImg.end());

    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(outputBatchSize, dstW, dstH, dstFormat);

    cvcuda::ResizeCropConvertReformat resizeCrop;
    EXPECT_EQ(expectedReturnCode,
              nvcv::ProtectCall([&] { resizeCrop(nullptr, src, dstTensor, resizeDim, interp, cropPos, manip); }));
}
