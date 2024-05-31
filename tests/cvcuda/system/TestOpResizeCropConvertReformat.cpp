/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ResizeUtils.hpp"

#include <common/InterpUtils.hpp>
#include <common/TypedTests.hpp>
// #include <common/ValueTests.hpp>
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
#include <util/TensorDataUtils.hpp>

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

template<typename SrcT, typename DstT>
void CropConvert(DstT *dst, const nvcv::Size2D dstSize, const nvcv::ImageFormat dstFormat, const SrcT *src,
                 const nvcv::Size2D srcSize, const nvcv::ImageFormat srcFormat, const int numImages, const int2 cropPos,
                 const NVCVChannelManip manip, const double scale = 1.0, const double offst = 0.0)
{
    int srcPlanes   = srcFormat.numPlanes();
    int dstPlanes   = dstFormat.numPlanes();
    int srcChannels = srcFormat.numChannels();
    int dstChannels = dstFormat.numChannels();

    size_t srcIncrX = srcChannels / srcPlanes; // 1 if planar; srcChannels if not.
    size_t dstIncrX = dstChannels / dstPlanes; // 1 if planar; dstChannels if not.
    size_t srcIncrY = srcIncrX * srcSize.w;
    size_t dstIncrY = dstIncrX * dstSize.w;
    size_t srcIncrC = (srcPlanes > 1 ? srcSize.w * srcSize.h : 1);
    size_t dstIncrC = (dstPlanes > 1 ? dstSize.w * dstSize.h : 1);
    size_t srcIncrN = srcSize.w * srcSize.h * srcChannels;
    size_t dstIncrN = dstSize.w * dstSize.h * dstChannels;
    size_t srcOffst = cropPos.y * srcIncrY + cropPos.x * srcIncrX;

    int channelMap[4] = {0, 1, 2, 3};

    int channels = (srcChannels < dstChannels ? srcChannels : dstChannels);

    if (manip == NVCV_CHANNEL_REVERSE)
    {
        for (int c = 0; c < channels; ++c) channelMap[c] = channels - c - 1;
    }

    for (int i = 0; i < numImages; i++)
    {
        const SrcT *srcBase = src + i * srcIncrN + srcOffst;
        DstT       *dstBase = dst + i * dstIncrN;

        for (int y = 0; y < dstSize.h; y++)
        {
            const SrcT *srcRow = srcBase + y * srcIncrY;
            DstT       *dstRow = dstBase + y * dstIncrY;

            for (int x = 0; x < dstSize.w; x++)
            {
                const SrcT *srcPtr = srcRow + x * srcIncrX;
                DstT       *dstPtr = dstRow + x * dstIncrX;

                for (int c = 0; c < channels; c++)
                {
                    dstPtr[channelMap[c] * dstIncrC] = static_cast<DstT>(srcPtr[c * srcIncrC] * scale + offst);
                }
            }
        }
    }
}

template<typename SrcT, typename DstT>
void CropConvert(std::vector<DstT> &dst, const nvcv::Size2D dstSize, const nvcv::ImageFormat dstFormat,
                 const std::vector<SrcT> src, const nvcv::Size2D srcSize, const nvcv::ImageFormat srcFormat,
                 const int numImages, const int2 cropPos, const NVCVChannelManip manip, const double scale = 1.0,
                 const double offst = 0.0)
{
    CropConvert(dst.data(), dstSize, dstFormat, src.data(), srcSize, srcFormat, numImages, cropPos, manip, scale,
                offst);
}

template<typename SrcT, typename DstT>
void CropConvert(DstT *dst, const nvcv::Size2D dstSize, const nvcv::ImageFormat dstFormat, const std::vector<SrcT> src,
                 const nvcv::Size2D srcSize, const nvcv::ImageFormat srcFormat, const int numImages, const int2 cropPos,
                 const NVCVChannelManip manip, const double scale = 1.0, const double offst = 0.0)
{
    CropConvert(dst, dstSize, dstFormat, src.data(), srcSize, srcFormat, numImages, cropPos, manip, scale, offst);
}

template<typename T>
void fillVec(std::vector<T> &vec, const nvcv::Size2D size, const nvcv::ImageFormat frmt, size_t offst = 0)
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

#define _TEST_ROW(SrcShape, ResizeDim, Interp, DstSize, CropPos, SrcFrmt, DstFrmt, SrcType, DstType)           \
    ttype::Types<ttype::Value<SrcShape>, ttype::Value<ResizeDim>, ttype::Value<Interp>, ttype::Value<DstSize>, \
                 ttype::Value<CropPos>, ttype::Value<SrcFrmt>, ttype::Value<DstFrmt>, SrcType, DstType>

// clang-format off

NVCV_TYPED_TEST_SUITE(
    OpResizeCropConvertReformat, ttype::Types<
    // Test cases: RGB (interleaved) -> BGR (planar); linear interpolation; float and uchar output.
    _TEST_ROW(_SHAPE(   8,    8,  1), int2(  8,   8), NVCV_INTERP_LINEAR, int2(   6,   6), int2(  1,   1), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  0
    _TEST_ROW(_SHAPE(   8,    8,  1), int2( 16,  16), NVCV_INTERP_LINEAR, int2(  12,  12), int2(  2,   2), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  1
    _TEST_ROW(_SHAPE(  42,   48,  1), int2( 23,  24), NVCV_INTERP_LINEAR, int2(  15,  13), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  2
    _TEST_ROW(_SHAPE(  42,   40,  3), int2( 21,  20), NVCV_INTERP_LINEAR, int2(  17,  13), int2(  1,   1), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  3
    _TEST_ROW(_SHAPE(  21,   21,  5), int2( 42,  42), NVCV_INTERP_LINEAR, int2(  32,  32), int2( 10,  10), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  4
    _TEST_ROW(_SHAPE( 113,   12,  7), int2( 12,  36), NVCV_INTERP_LINEAR, int2(   7,  13), int2(  3,  11), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  5
    _TEST_ROW(_SHAPE(  17,  151,  7), int2( 48,  16), NVCV_INTERP_LINEAR, int2(  32,  16), int2(  4,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  6
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  7
    _TEST_ROW(_SHAPE(1080, 1920, 13), int2(800, 600), NVCV_INTERP_LINEAR, int2( 640, 480), int2(101,  64), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  8
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   //  9
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8p,    uchar3, uint8_t), // 10
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8p,    uchar3, uint8_t), // 11

    // Test cases: RGB (interleaved) -> RGB (planar); linear interpolation; float and uchar output.
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32p,  uchar3, float),   // 12
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32p,  uchar3, float),   // 13
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p,    uchar3, uint8_t), // 14
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p,    uchar3, uint8_t), // 15

    // Test cases: BGR (interleaved) -> RGB (planar); linear interpolation; float and uchar output.
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p,  uchar3, float),   // 16
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p,  uchar3, float),   // 17
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,    uchar3, uint8_t), // 18
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,    uchar3, uint8_t), // 19

    // Test cases: BGR (interleaved) -> BGR (planar); linear interpolation; float and uchar output.
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   // 20
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGRf32p,  uchar3, float),   // 21
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGR8p,    uchar3, uint8_t), // 22
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_BGR8p,    uchar3, uint8_t), // 23

    // Test cases: RGB (interleaved) -> BGR (interleaved); linear interpolation; float and uchar output.
    _TEST_ROW(_SHAPE(   8,    8,  1), int2(  8,   8), NVCV_INTERP_LINEAR, int2(   6,   6), int2(  1,   1), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3),  // 24
    _TEST_ROW(_SHAPE(   8,    8,  1), int2( 16,  16), NVCV_INTERP_LINEAR, int2(  12,  12), int2(  2,   2), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3),  // 25
    _TEST_ROW(_SHAPE( 113,   12,  7), int2( 12,  36), NVCV_INTERP_LINEAR, int2(   7,  13), int2(  3,  11), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3),  // 26
    _TEST_ROW(_SHAPE(  17,  151,  7), int2( 48,  16), NVCV_INTERP_LINEAR, int2(  32,  16), int2(  4,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3),  // 27
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3),  // 28
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32,   uchar3, float3),  // 29
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8,     uchar3, uchar3),  // 30
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8,     uchar3, uchar3),  // 31

    // Test cases: RGB (interleaved) -> RGB (interleaved); linear interpolation; float and uchar output.
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32,   uchar3, float3),  // 32
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32,   uchar3, float3),  // 33
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8,     uchar3, uchar3),  // 34
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8,     uchar3, uchar3),  // 35

    // Test cases: BGR (interleaved) -> RGB (interleaved); linear interpolation; float and uchar output.
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32,   uchar3, float3),  // 36
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBf32,   uchar3, float3),  // 37
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_LINEAR, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8,     uchar3, uchar3),  // 38
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_LINEAR, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8,     uchar3, uchar3),  // 39

    // Test cases: RGB (interleaved) -> BGR (planar); nearest-neighbor interpolation; float and uchar output.
    _TEST_ROW(_SHAPE(   8,    8,  1), int2(  8,   8), NVCV_INTERP_NEAREST, int2(   6,   6), int2(  1,   1), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float),   // 40
    _TEST_ROW(_SHAPE(   8,    8,  5), int2( 16,  16), NVCV_INTERP_NEAREST, int2(  12,  12), int2(  2,   2), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float),   // 41
    _TEST_ROW(_SHAPE(  42,   48,  1), int2( 23,  24), NVCV_INTERP_NEAREST, int2(  15,  13), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float),   // 42
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_NEAREST, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float),   // 43
    _TEST_ROW(_SHAPE(1080, 1920, 13), int2(800, 600), NVCV_INTERP_NEAREST, int2( 640, 480), int2(101,  64), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float),   // 44
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGRf32p, uchar3, float),   // 45
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_NEAREST, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8p,   uchar3, uint8_t), // 46
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8p,   uchar3, uint8_t), // 47

    // Test cases: BGR (interleaved) -> RGB (planar); nearest-neighbor interpolation; float and uchar output.
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_NEAREST, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float),   // 48
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGBf32p, uchar3, float),   // 49
    _TEST_ROW(_SHAPE( 313,  212,  4), int2(412, 336), NVCV_INTERP_NEAREST, int2( 412, 336), int2(  0,   0), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,   uchar3, uint8_t), // 50
    _TEST_ROW(_SHAPE(1280,  960,  3), int2(300, 225), NVCV_INTERP_NEAREST, int2( 250, 200), int2( 15,  16), NVCV_IMAGE_FORMAT_BGR8, NVCV_IMAGE_FORMAT_RGB8p,   uchar3, uint8_t)  // 51

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

    nvcv::ImageFormat srcFormat{ttype::GetValue<TypeParam, 5>};
    nvcv::ImageFormat dstFormat{ttype::GetValue<TypeParam, 6>};

    using SrcVT = typename ttype::GetType<TypeParam, 7>;
    using DstVT = typename ttype::GetType<TypeParam, 8>;
    using SrcBT = typename cuda::BaseType<SrcVT>;
    using DstBT = typename cuda::BaseType<DstVT>;

    int srcW = srcShape.x;
    int srcH = srcShape.y;
    int dstW = cropDim.x;
    int dstH = cropDim.y;
    int tmpW = resize.x;
    int tmpH = resize.y;

    int numImages   = srcShape.z;
    int srcChannels = srcFormat.numChannels();
    int dstChannels = dstFormat.numChannels();
    int srcPlanes   = srcFormat.numPlanes();
    int dstPlanes   = dstFormat.numPlanes();
    int srcPixElems = srcChannels / srcPlanes;
    int dstPixElems = dstChannels / dstPlanes;

    ASSERT_LE(srcChannels, 4);
    ASSERT_EQ(srcChannels, dstChannels);

    NVCVSize2D resizeDim{resize.x, resize.y};

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
    int tmpRowElems = srcPixElems * tmpW;
    int dstRowElems = dstPixElems * dstW;

    size_t srcElems = (size_t)srcRowElems * (size_t)srcH * (size_t)srcPlanes * (size_t)numImages;
    size_t tmpElems = (size_t)tmpRowElems * (size_t)tmpH * (size_t)srcPlanes * (size_t)numImages;
    size_t dstElems = (size_t)dstRowElems * (size_t)dstH * (size_t)dstPlanes * (size_t)numImages;

    nvcv::Size2D srcSize{srcW, srcH};
    nvcv::Size2D tmpSize{tmpW, tmpH};
    nvcv::Size2D dstSize{dstW, dstH};

    size_t srcPitch = srcW * sizeof(SrcVT);
    size_t tmpPitch = tmpW * sizeof(SrcVT);
    size_t dstPitch = dstW * sizeof(DstVT);

    std::vector<SrcBT> srcVec(srcElems);
    std::vector<SrcBT> tmpVec(tmpElems);
    std::vector<DstBT> refVec(dstElems);

    // Populate source tensor.
    fillVec(srcVec, srcSize, srcFormat);

    // Generate "gold" result for image and place in reference vector.
    test::Resize(tmpVec, tmpPitch, tmpSize, srcVec, srcPitch, srcSize, srcFormat, interp, false);
    CropConvert(refVec, dstSize, dstFormat, tmpVec, tmpSize, srcFormat, numImages, cropPos, manip);

    // Copy source tensor to device.
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(src->basePtr(), srcAccess->rowStride(), srcVec.data(), srcPitch, srcPitch,
                                        srcH * srcPlanes, cudaMemcpyHostToDevice));

    // Run fused ResizeCropConvertReformat operator.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::ResizeCropConvertReformat resizeCrop;
    EXPECT_NO_THROW(resizeCrop(stream, srcTensor, dstTensor, resizeDim, interp, cropPos, manip));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Copy destination tensor back to host.
    std::vector<DstBT> dstVec(dstElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dstVec.data(), dstPitch, dst->basePtr(), dstAccess->rowStride(), dstPitch,
                                        dstH * dstPlanes, cudaMemcpyDeviceToHost));

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

    nvcv::ImageFormat srcFormat{ttype::GetValue<TypeParam, 5>};
    nvcv::ImageFormat dstFormat{ttype::GetValue<TypeParam, 6>};

    using SrcVT = typename ttype::GetType<TypeParam, 7>;
    using DstVT = typename ttype::GetType<TypeParam, 8>;
    using SrcBT = typename cuda::BaseType<SrcVT>;
    using DstBT = typename cuda::BaseType<DstVT>;

    int srcW = srcShape.x;
    int srcH = srcShape.y;
    int dstW = cropDim.x;
    int dstH = cropDim.y;
    int tmpW = resize.x;
    int tmpH = resize.y;

    int numImages   = srcShape.z;
    int srcChannels = srcFormat.numChannels();
    int dstChannels = dstFormat.numChannels();
    int srcPlanes   = srcFormat.numPlanes();
    int dstPlanes   = dstFormat.numPlanes();
    int srcPixElems = srcChannels / srcPlanes;
    int dstPixElems = dstChannels / dstPlanes;

    ASSERT_LE(srcChannels, 4);
    ASSERT_EQ(srcChannels, dstChannels);

    NVCVSize2D resizeDim{resize.x, resize.y};

    NVCVChannelManip manip = ChannelManip(srcFormat, dstFormat);

    std::vector<nvcv::Image> srcImg;

    uniform_dist<SrcBT> randVal(std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::min : SrcBT{0},
                                std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::max : SrcBT{1});

    std::uniform_int_distribution<int> randW(srcW * 0.8, srcW * 1.2);
    std::uniform_int_distribution<int> randH(srcH * 0.8, srcH * 1.2);

    int tmpRowElems = srcPixElems * tmpW;
    int dstRowElems = dstPixElems * dstW;

    size_t tmpElems = (size_t)tmpRowElems * (size_t)tmpH * (size_t)srcPlanes;
    size_t refIncr  = (size_t)dstRowElems * (size_t)dstH * (size_t)dstPlanes;
    size_t dstElems = refIncr * (size_t)numImages;

    nvcv::Size2D tmpSize{tmpW, tmpH};
    nvcv::Size2D dstSize{dstW, dstH};

    std::vector<SrcBT> tmpVec(tmpElems);
    std::vector<DstBT> refVec(dstElems);

    size_t tmpPitch = tmpW * sizeof(SrcVT);
    size_t dstPitch = dstW * sizeof(DstVT);

    for (int i = 0; i < numImages; ++i)
    {
        int imgW = (interp ? randW(randEng) : srcW);
        int imgH = (interp ? randH(randEng) : srcH);

        srcImg.emplace_back(nvcv::Size2D{imgW, imgH}, srcFormat);

        auto srcData = srcImg[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(srcData);

        int imgRowElems = srcPixElems * imgW;

        size_t imgPitch = imgW * sizeof(SrcVT);
        size_t imgElems = (size_t)imgRowElems * (size_t)imgH * (size_t)srcPlanes;

        nvcv::Size2D imgSize{imgW, imgH};

        std::vector<SrcBT> imgVec(imgElems);

        // Populate image tensor .
        fillVec(imgVec, imgSize, srcFormat);

        // Generate "gold" result for image and place in reference image plane.
        DstBT *refPlane = refVec.data() + i * refIncr;

        test::Resize(tmpVec, tmpPitch, tmpSize, imgVec, imgPitch, imgSize, srcFormat, interp, true);
        CropConvert(refPlane, dstSize, dstFormat, tmpVec, tmpSize, srcFormat, 1, cropPos, manip);

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
    EXPECT_NO_THROW(resizeCrop(stream, src, dstTensor, resizeDim, interp, cropPos, manip));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Copy destination tensor back to host.
    std::vector<DstBT> dstVec(dstElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dstVec.data(), dstPitch, dst->basePtr(), dstAccess->rowStride(), dstPitch,
                                        dstH * dstPlanes * numImages, cudaMemcpyDeviceToHost));

    // Compare "gold" reference to computed output.
    VEC_EXPECT_NEAR(refVec, dstVec, 1);
}
