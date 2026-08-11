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

#include "Definitions.hpp"
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpAdvCvtColor.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstring>
#include <iostream>
#include <random>

namespace gt    = ::testing;
namespace test  = nvcv::test;
namespace util  = nvcv::util;
namespace mmath = nvcv::cuda::math;

template<class T>
using Matrix3x3 = mmath::Matrix<T, 3, 3>;

static Matrix3x3<float> getRGB2YUVMatrix(nvcv::ColorSpec spec)
{
    Matrix3x3<float> matrix;
    switch (static_cast<NVCVColorSpec>(spec))
    {
    case NVCV_COLOR_SPEC_BT601:
    {
        matrix.load({0.299f, 0.587f, 0.114f, -0.168736f, -0.331264f, 0.5f, 0.5f, -0.418688f, -0.0813124f});
        return matrix;
    }
    case NVCV_COLOR_SPEC_BT709:
    {
        matrix.load({0.2126f, 0.7152f, 0.0722f, -0.114572f, -0.385428f, 0.5f, 0.5f, -0.454153f, -0.0458471f});
        return matrix;
    }
    case NVCV_COLOR_SPEC_BT2020:
    {
        matrix.load({0.2627f, 0.678f, 0.0593f, -0.13963f, -0.36037f, 0.5f, 0.5f, -0.459786f, -0.0402143f});
        return matrix;
    }
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unknown color spec");
    }
}

static Matrix3x3<float> getYUV2RGBMatrix(nvcv::ColorSpec spec)
{
    Matrix3x3<float> matrix = getRGB2YUVMatrix(spec);
    mmath::inv_inplace<float>(matrix);
    return matrix;
}

static float ToFloat(int value)
{
    return static_cast<float>(value);
}

static int TruncateToInt(float value)
{
    return static_cast<int>(value);
}

struct RGBPixel
{
    int r;
    int g;
    int b;
};

static RGBPixel GetRGBPixel(const std::vector<uint8_t> &rgbData, int pixelIndex, bool bgr)
{
    RGBPixel pixel{rgbData[pixelIndex], rgbData[pixelIndex + 1], rgbData[pixelIndex + 2]};
    if (bgr)
    {
        std::swap(pixel.r, pixel.b);
    }
    return pixel;
}

struct NV12ChromaBlockData
{
    const Matrix3x3<float>     &conversionMatrix;
    const std::vector<uint8_t> &rgbData;
    int                         width;
    bool                        bgr;
};

static void AccumulateNV12ChromaBlock(const NV12ChromaBlockData &block, int row, int col, int &uSum, int &vSum)
{
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            int      idx   = ((row + y) * block.width + (col + x)) * 3;
            RGBPixel pixel = GetRGBPixel(block.rgbData, idx, block.bgr);

            uSum += TruncateToInt(block.conversionMatrix[1][0] * ToFloat(pixel.r)
                                  + block.conversionMatrix[1][1] * ToFloat(pixel.g)
                                  + block.conversionMatrix[1][2] * ToFloat(pixel.b));
            vSum += TruncateToInt(block.conversionMatrix[2][0] * ToFloat(pixel.r)
                                  + block.conversionMatrix[2][1] * ToFloat(pixel.g)
                                  + block.conversionMatrix[2][2] * ToFloat(pixel.b));
        }
    }
}

static bool isBGR(NVCVColorConversionCode code)
{
    switch (code)
    {
    case NVCV_COLOR_BGR2YUV: //444 types
    case NVCV_COLOR_YUV2BGR:
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2BGR_NV21:
        return true;
    default:
        return false;
    }
}

static bool isYVU(NVCVColorConversionCode code)
{
    switch (code)
    {
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_YUV2BGR_NV21:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_YUV2RGB_NV21:
        return true;
    default:
        return false;
    }
}

static std::vector<uint8_t> convertYUVtoRGB(const Matrix3x3<float>     &conversionMatrix,
                                            const std::vector<uint8_t> &yuvData, bool bgr)
{
    assert(yuvData.size() % 3 == 0); // Ensure the input data has sets of 3 (Y, U, V)
    std::vector<uint8_t> rgbData;
    rgbData.reserve(yuvData.size()); // Reserve space for RGB data

    for (size_t i = 0; i < yuvData.size(); i += 3)
    {
        int Y = yuvData[i];
        int U = yuvData[i + 1];
        int V = yuvData[i + 2];

        int R = TruncateToInt(conversionMatrix[0][0] * ToFloat(Y) + conversionMatrix[0][1] * ToFloat(U - 128)
                              + conversionMatrix[0][2] * ToFloat(V - 128));
        int G = TruncateToInt(conversionMatrix[1][0] * ToFloat(Y) + conversionMatrix[1][1] * ToFloat(U - 128)
                              + conversionMatrix[1][2] * ToFloat(V - 128));
        int B = TruncateToInt(conversionMatrix[2][0] * ToFloat(Y) + conversionMatrix[2][1] * ToFloat(U - 128)
                              + conversionMatrix[2][2] * ToFloat(V - 128));

        if (bgr)
        {
            std::swap(R, B);
        }
        rgbData.push_back(nvcv::cuda::SaturateCast<uint8_t>(R));
        rgbData.push_back(nvcv::cuda::SaturateCast<uint8_t>(G));
        rgbData.push_back(nvcv::cuda::SaturateCast<uint8_t>(B));
    }

    return rgbData;
}

static std::vector<uint8_t> convertRGBtoYUV(const Matrix3x3<float>     &conversionMatrix,
                                            const std::vector<uint8_t> &rgbData, bool bgr)
{
    assert(rgbData.size() % 3 == 0); // Ensure the input data has sets of 3 (Y, U, V)
    std::vector<uint8_t> yuvData;
    yuvData.reserve(rgbData.size()); // Reserve space for YUV data

    for (size_t i = 0; i < rgbData.size(); i += 3)
    {
        int R = rgbData[i];
        int G = rgbData[i + 1];
        int B = rgbData[i + 2];
        if (bgr)
        {
            std::swap(R, B);
        }

        int Y = TruncateToInt(conversionMatrix[0][0] * ToFloat(R) + conversionMatrix[0][1] * ToFloat(G)
                              + conversionMatrix[0][2] * ToFloat(B));
        int U = TruncateToInt(conversionMatrix[1][0] * ToFloat(R) + conversionMatrix[1][1] * ToFloat(G)
                              + conversionMatrix[1][2] * ToFloat(B) + 128.0f);
        int V = TruncateToInt(conversionMatrix[2][0] * ToFloat(R) + conversionMatrix[2][1] * ToFloat(G)
                              + conversionMatrix[2][2] * ToFloat(B) + 128.0f);

        yuvData.push_back(nvcv::cuda::SaturateCast<uint8_t>(Y));
        yuvData.push_back(nvcv::cuda::SaturateCast<uint8_t>(U));
        yuvData.push_back(nvcv::cuda::SaturateCast<uint8_t>(V));
    }

    return yuvData;
}

static std::vector<uint8_t> convertRGBtoNV12(const Matrix3x3<float>     &conversionMatrix,
                                             const std::vector<uint8_t> &rgbData, int width, int height, bool bgr,
                                             bool yvu)
{
    assert(rgbData.size()
           == (size_t)(width * height
                       * 3)); // Ensure the input data has sets of 3 (R, G, B) for the given width and height
    assert(width % 2 == 0
           && height % 2 == 0); // Ensure both width and height are even since we're processing 2x2 blocks

    std::vector<uint8_t> nv12Data;
    nv12Data.reserve(width * height * 3 / 2); // NV12 needs 1.5 bytes per RGB pixel

    NV12ChromaBlockData chromaBlock{conversionMatrix, rgbData, width, bgr};

    // Convert all RGB values to Y values and store them.
    for (size_t i = 0; i < rgbData.size(); i += 3)
    {
        int R = rgbData[i];
        int G = rgbData[i + 1];
        int B = rgbData[i + 2];
        if (bgr)
        {
            std::swap(R, B);
        }

        int Y = TruncateToInt(conversionMatrix[0][0] * ToFloat(R) + conversionMatrix[0][1] * ToFloat(G)
                              + conversionMatrix[0][2] * ToFloat(B));
        nv12Data.push_back(nvcv::cuda::SaturateCast<uint8_t>(Y));
    }

    // Calculate U and V values for each 2x2 block and store them interleaved.
    for (int h = 0; h < height; h += 2)
    {
        for (int w = 0; w < width; w += 2)
        {
            int U_sum = 0;
            int V_sum = 0;

            AccumulateNV12ChromaBlock(chromaBlock, h, w, U_sum, V_sum);

            int U = (U_sum / 4) + 128; // Average of 4 U values
            int V = (V_sum / 4) + 128; // Average of 4 V values

            if (yvu)
            {
                std::swap(U, V);
            }
            nv12Data.push_back(nvcv::cuda::SaturateCast<uint8_t>(U));
            nv12Data.push_back(nvcv::cuda::SaturateCast<uint8_t>(V));
        }
    }

    return nv12Data;
}

struct FixedRGB2YUVConstants
{
    int r2y;
    int g2y;
    int b2y;
    int b2u;
    int r2v;
};

static FixedRGB2YUVConstants GetFixedRGB2YUVConstants(nvcv::ColorSpec spec)
{
    switch (static_cast<NVCVColorSpec>(spec))
    {
    case NVCV_COLOR_SPEC_BT601:
        return {4899, 9671, 1868, 9246, 11686};
    case NVCV_COLOR_SPEC_BT709:
        return {3483, 11718, 1265, 8829, 10404};
    case NVCV_COLOR_SPEC_BT2020:
        return {4304, 11108, 972, 8708, 11111};
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unknown color spec");
    }
}

static int FixedDescale(int value)
{
    constexpr int shift = 14;
    return (value + (1 << (shift - 1))) >> shift;
}

static void AccumulateRGBToNVPixel(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int width, int channels,
                                   int x, int y, int bidx, const FixedRGB2YUVConstants &constants, int &uSum, int &vSum)
{
    size_t srcIdx = (static_cast<size_t>(y) * width + x) * channels;
    int    b      = src[srcIdx + bidx];
    int    g      = src[srcIdx + 1];
    int    r      = src[srcIdx + (bidx ^ 2)];
    int    yValue = FixedDescale(r * constants.r2y + g * constants.g2y + b * constants.b2y);
    int    u      = FixedDescale((b - yValue) * constants.b2u);
    int    v      = FixedDescale((r - yValue) * constants.r2v);

    dst[static_cast<size_t>(y) * width + x] = nvcv::cuda::SaturateCast<uint8_t>(yValue);
    uSum += u;
    vSum += v;
}

static std::vector<uint8_t> ConvertRGBToNVExact(const std::vector<uint8_t> &src, int width, int height, int channels,
                                                NVCVColorConversionCode code, nvcv::ColorSpec spec)
{
    const auto constants = GetFixedRGB2YUVConstants(spec);
    const int  bidx      = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_BGR2YUV_NV21) ? 0 : 2;
    const int  uidx      = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_RGB2YUV_NV12) ? 0 : 1;

    std::vector<uint8_t> dst(static_cast<size_t>(width) * height * 3 / 2);
    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            int uSum = 0;
            int vSum = 0;
            AccumulateRGBToNVPixel(src, dst, width, channels, x, y, bidx, constants, uSum, vSum);
            AccumulateRGBToNVPixel(src, dst, width, channels, x + 1, y, bidx, constants, uSum, vSum);
            AccumulateRGBToNVPixel(src, dst, width, channels, x, y + 1, bidx, constants, uSum, vSum);
            AccumulateRGBToNVPixel(src, dst, width, channels, x + 1, y + 1, bidx, constants, uSum, vSum);

            size_t uvIdx          = static_cast<size_t>(width) * height + static_cast<size_t>(y / 2) * width + x;
            dst[uvIdx + uidx]     = nvcv::cuda::SaturateCast<uint8_t>(uSum / 4 + 128);
            dst[uvIdx + 1 - uidx] = nvcv::cuda::SaturateCast<uint8_t>(vSum / 4 + 128);
        }
    }
    return dst;
}

struct FixedYUV2RGBConstants
{
    int u2b;
    int u2g;
    int v2g;
    int v2r;
};

static FixedYUV2RGBConstants GetFixedYUV2RGBConstants(nvcv::ColorSpec spec)
{
    switch (static_cast<NVCVColorSpec>(spec))
    {
    case NVCV_COLOR_SPEC_BT601:
        return {29032, -5636, -11698, 22970};
    case NVCV_COLOR_SPEC_BT709:
        return {30402, -3069, -7670, 25802};
    case NVCV_COLOR_SPEC_BT2020:
        return {30825, -2696, -9361, 24160};
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unknown color spec");
    }
}

static std::vector<uint8_t> ConvertNVToRGBExact(const std::vector<uint8_t> &src, int width, int height, int channels,
                                                NVCVColorConversionCode code, nvcv::ColorSpec spec)
{
    const auto constants = GetFixedYUV2RGBConstants(spec);
    const int  bidx      = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2BGR_NV21) ? 0 : 2;
    const int  uidx      = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2RGB_NV12) ? 0 : 1;
    const auto uvOffset  = static_cast<size_t>(width) * height;

    std::vector<uint8_t> dst(static_cast<size_t>(width) * height * channels);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int    yValue = src[static_cast<size_t>(y) * width + x];
            size_t uvIdx  = uvOffset + static_cast<size_t>(y / 2) * width + (x & ~1);
            int    u      = src[uvIdx + uidx] - 128;
            int    v      = src[uvIdx + 1 - uidx] - 128;
            int    b      = yValue + FixedDescale(u * constants.u2b);
            int    g      = yValue + FixedDescale(u * constants.u2g + v * constants.v2g);
            int    r      = yValue + FixedDescale(v * constants.v2r);

            size_t dstIdx            = (static_cast<size_t>(y) * width + x) * channels;
            dst[dstIdx + bidx]       = nvcv::cuda::SaturateCast<uint8_t>(b);
            dst[dstIdx + 1]          = nvcv::cuda::SaturateCast<uint8_t>(g);
            dst[dstIdx + (bidx ^ 2)] = nvcv::cuda::SaturateCast<uint8_t>(r);
            if (channels == 4)
            {
                dst[dstIdx + 3] = 0xff;
            }
        }
    }
    return dst;
}

static std::vector<uint8_t> Convert444Exact(const std::vector<uint8_t> &src, int width, int height,
                                            NVCVColorConversionCode code, nvcv::ColorSpec spec)
{
    std::vector<uint8_t> dst(src.size());
    const bool           toYUV = code == NVCV_COLOR_BGR2YUV || code == NVCV_COLOR_RGB2YUV;
    const int            bidx  = (code == NVCV_COLOR_BGR2YUV || code == NVCV_COLOR_YUV2BGR) ? 0 : 2;

    if (toYUV)
    {
        const auto constants = GetFixedRGB2YUVConstants(spec);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t idx    = (static_cast<size_t>(y) * width + x) * 3;
                const int    b      = src[idx + bidx];
                const int    g      = src[idx + 1];
                const int    r      = src[idx + (bidx ^ 2)];
                const int    yValue = FixedDescale(r * constants.r2y + g * constants.g2y + b * constants.b2y);
                const int    u      = FixedDescale((b - yValue) * constants.b2u + (128 << 14));
                const int    v      = FixedDescale((r - yValue) * constants.r2v + (128 << 14));

                dst[idx]     = nvcv::cuda::SaturateCast<uint8_t>(yValue);
                dst[idx + 1] = nvcv::cuda::SaturateCast<uint8_t>(u);
                dst[idx + 2] = nvcv::cuda::SaturateCast<uint8_t>(v);
            }
        }
    }
    else
    {
        const auto constants = GetFixedYUV2RGBConstants(spec);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t idx    = (static_cast<size_t>(y) * width + x) * 3;
                const int    yValue = src[idx];
                const int    u      = src[idx + 1] - 128;
                const int    v      = src[idx + 2] - 128;
                const int    b      = yValue + FixedDescale(u * constants.u2b);
                const int    g      = yValue + FixedDescale(u * constants.u2g + v * constants.v2g);
                const int    r      = yValue + FixedDescale(v * constants.v2r);

                dst[idx + bidx]       = nvcv::cuda::SaturateCast<uint8_t>(b);
                dst[idx + 1]          = nvcv::cuda::SaturateCast<uint8_t>(g);
                dst[idx + (bidx ^ 2)] = nvcv::cuda::SaturateCast<uint8_t>(r);
            }
        }
    }
    return dst;
}

static std::vector<uint8_t> convertNV12toRGB(const Matrix3x3<float>     &conversionMatrix,
                                             const std::vector<uint8_t> &nv12Data, int width, int height, bool bgr,
                                             bool yvu)
{
    assert(nv12Data.size()
           == (size_t)(width * height * 3
                       / 2)); // Ensure the input data size is consistent with the provided width and height
    std::vector<uint8_t> rgbData;
    rgbData.reserve(nv12Data.size() * 3 / 2); // Reserve space for RGB data (1.5 times the NV12 data size)

    // Pointer to the beginning of the UV data
    const uint8_t *uvData = &nv12Data[width * height];

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int Y = nv12Data[i * width + j];

            // Calculate UV index: For every two Y values in a row, there's one UV pair.
            int uvIndex = (i / 2) * width + (j & ~1); // 'j & ~1' ensures j is even.
            int U       = uvData[uvIndex];
            int V       = uvData[uvIndex + 1];
            if (yvu)
            {
                std::swap(U, V);
            }

            int R = TruncateToInt(conversionMatrix[0][0] * ToFloat(Y) + conversionMatrix[0][1] * ToFloat(U - 128)
                                  + conversionMatrix[0][2] * ToFloat(V - 128));
            int G = TruncateToInt(conversionMatrix[1][0] * ToFloat(Y) + conversionMatrix[1][1] * ToFloat(U - 128)
                                  + conversionMatrix[1][2] * ToFloat(V - 128));
            int B = TruncateToInt(conversionMatrix[2][0] * ToFloat(Y) + conversionMatrix[2][1] * ToFloat(U - 128)
                                  + conversionMatrix[2][2] * ToFloat(V - 128));

            if (bgr)
            {
                std::swap(R, B);
            }

            rgbData.push_back(nvcv::cuda::SaturateCast<uint8_t>(R));
            rgbData.push_back(nvcv::cuda::SaturateCast<uint8_t>(G));
            rgbData.push_back(nvcv::cuda::SaturateCast<uint8_t>(B));
        }
    }

    return rgbData;
}

#define VEC_EXPECT_NEAR(vec1, vec2, delta)                              \
    ASSERT_EQ(vec1.size(), vec2.size());                                \
    for (std::size_t idx = 0; idx < vec1.size(); ++idx)                 \
    {                                                                   \
        EXPECT_NEAR(vec1[idx], vec2[idx], delta) << "At index " << idx; \
    }

struct VerifyOutputParams
{
    int                     batches;
    NVCVColorConversionCode convCode;
    nvcv::ColorSpec         colorSpec;
    float                   maxDiff;
    int                     width;
    int                     height;
};

static void verifyOutput(const nvcv::Tensor &inTensor, const nvcv::Tensor &outTensor, const VerifyOutputParams &params)
{
    for (int i = 0; i < params.batches; ++i)
    {
        std::vector<uint8_t> outData;
        std::vector<uint8_t> inData;

        // get 0th sample since histogram is just a 2d tensor
        util::GetImageVectorFromTensor(inTensor.exportData(), i, inData);
        util::GetImageVectorFromTensor(outTensor.exportData(), i, outData);

        switch (params.convCode)
        {
        case NVCV_COLOR_BGR2YUV:
        case NVCV_COLOR_RGB2YUV:
        {
            std::vector<uint8_t> goldOut
                = convertRGBtoYUV(getRGB2YUVMatrix(params.colorSpec), inData, isBGR(params.convCode));
            VEC_EXPECT_NEAR(goldOut, outData, params.maxDiff);
            break;
        }
        case NVCV_COLOR_YUV2BGR:
        case NVCV_COLOR_YUV2RGB:
        {
            std::vector<uint8_t> goldOut
                = convertYUVtoRGB(getYUV2RGBMatrix(params.colorSpec), inData, isBGR(params.convCode));
            VEC_EXPECT_NEAR(goldOut, outData, params.maxDiff);
            break;
        }
        case NVCV_COLOR_YUV2RGB_NV12:
        case NVCV_COLOR_YUV2BGR_NV12:
        case NVCV_COLOR_YUV2RGB_NV21:
        case NVCV_COLOR_YUV2BGR_NV21:
        {
            std::vector<uint8_t> goldOut
                = convertNV12toRGB(getYUV2RGBMatrix(params.colorSpec), inData, params.width, params.height,
                                   isBGR(params.convCode), isYVU(params.convCode));
            VEC_EXPECT_NEAR(goldOut, outData, params.maxDiff);
            break;
        }
        case NVCV_COLOR_RGB2YUV_NV21:
        case NVCV_COLOR_BGR2YUV_NV21:
        case NVCV_COLOR_RGB2YUV_NV12:
        case NVCV_COLOR_BGR2YUV_NV12:
        {
            std::vector<uint8_t> goldOut
                = convertRGBtoNV12(getRGB2YUVMatrix(params.colorSpec), inData, params.width, params.height,
                                   isBGR(params.convCode), isYVU(params.convCode));
            VEC_EXPECT_NEAR(goldOut, outData, params.maxDiff);
            break;
        }
        default:
            FAIL() << "Unsupported conversion code";
            break;
        }
    }
}

static bool isFromNV(NVCVColorConversionCode code)
{
    switch (code)
    {
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2BGR_NV21:
        return true;
    default:
        return false;
    }
}

static bool isToNV(NVCVColorConversionCode code)
{
    switch (code)
    {
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_BGR2YUV_NV21:
        return true;
    default:
        return false;
    }
}

static nvcv::Tensor MakeAdvCvtColorTensor(int numImages, int width, int height, int channels, const char *layout)
{
    if (std::strcmp(layout, "HWC") == 0 || std::strcmp(layout, "CHW") == 0)
    {
        if (std::strcmp(layout, "CHW") == 0)
        {
            return nvcv::Tensor(
                {
                    {channels, height, width},
                    layout
            },
                nvcv::TYPE_U8);
        }
        return nvcv::Tensor(
            {
                {height, width, channels},
                layout
        },
            nvcv::TYPE_U8);
    }
    if (std::strcmp(layout, "NCHW") == 0)
    {
        return nvcv::Tensor(
            {
                {numImages, channels, height, width},
                layout
        },
            nvcv::TYPE_U8);
    }
    return nvcv::Tensor(
        {
            {numImages, height, width, channels},
            layout
    },
        nvcv::TYPE_U8);
}

static cudaError_t AllocateMisalignedTensorBuffer(int samples, int sampleStride, int rowStride, int pixelStride,
                                                  NVCVByte *&allocation, nvcv::TensorDataStridedCuda::Buffer &buffer)
{
    if (cudaError_t status
        = cudaMalloc(reinterpret_cast<void **>(&allocation), static_cast<size_t>(sampleStride) * samples + 1);
        status != cudaSuccess)
    {
        return status;
    }

    buffer.basePtr    = allocation + 1;
    buffer.strides[0] = sampleStride;
    buffer.strides[1] = rowStride;
    buffer.strides[2] = pixelStride;
    buffer.strides[3] = 1;
    return cudaSuccess;
}

static void RunAdvCvtColorTensorPlanarParity(int width, int height, int numImages, int srcChannels, int dstChannels,
                                             NVCVColorConversionCode code, nvcv::ColorSpec colorSpec, bool rank3)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int samples = rank3 ? 1 : numImages;
    const int srcH    = isFromNV(code) ? height * 3 / 2 : height;
    const int dstH    = isToNV(code) ? height * 3 / 2 : height;
    const int dstC    = isToNV(code) ? 1 : dstChannels;
    const int srcRow  = width * srcChannels * static_cast<int>(sizeof(uint8_t));
    const int dstRow  = width * dstC * static_cast<int>(sizeof(uint8_t));

    nvcv::Tensor srcI = MakeAdvCvtColorTensor(samples, width, srcH, srcChannels, rank3 ? "HWC" : "NHWC");
    nvcv::Tensor dstI = MakeAdvCvtColorTensor(samples, width, dstH, dstC, rank3 ? "HWC" : "NHWC");
    nvcv::Tensor srcP = MakeAdvCvtColorTensor(samples, width, srcH, srcChannels, rank3 ? "CHW" : "NCHW");
    nvcv::Tensor dstP = MakeAdvCvtColorTensor(samples, width, dstH, dstC, rank3 ? "CHW" : "NCHW");

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

    for (int i = 0; i < samples; ++i)
    {
        std::vector<uint8_t> hwc(static_cast<size_t>(srcH) * srcRow);
        test::planar::FillDeterministicValues(hwc, static_cast<size_t>(i) * 101 + 13, nvcv::TYPE_U8);
        test::planar::UploadInterleavedSample(*srcIAcc, i, hwc, width, srcH, srcRow);
        test::planar::UploadPlanarSample(
            *srcPAcc, i, test::planar::DeinterleaveToPlanes(hwc, width, srcH, srcChannels, sizeof(uint8_t)), width,
            srcH, srcChannels, sizeof(uint8_t));
    }

    cvcuda::AdvCvtColor op;
    EXPECT_NO_THROW(op(stream, srcI, dstI, code, colorSpec));
    EXPECT_NO_THROW(op(stream, srcP, dstP, code, colorSpec));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);
        auto gpuInter    = test::planar::DownloadInterleavedSample(*dstIAcc, i, width, dstH, dstRow);
        auto planesOut   = test::planar::DownloadPlanarSample(*dstPAcc, i, width, dstH, dstC, sizeof(uint8_t));
        auto planarInter = test::planar::InterleaveFromPlanes(planesOut, width, dstH, dstC, sizeof(uint8_t));

        EXPECT_EQ(gpuInter, planarInter);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off

NVCV_TEST_SUITE_P(
    OpAdvCvtColor444PlanarExact,
    test::ValueList<int, int, NVCVColorConversionCode, NVCVColorSpec, bool>{
        // W,  H, Conversion code,      Color spec,             Rank-3 CHW
        { 37, 35, NVCV_COLOR_RGB2YUV,  NVCV_COLOR_SPEC_BT601,  false},
        { 29,  7, NVCV_COLOR_BGR2YUV,  NVCV_COLOR_SPEC_BT709,  true },
        { 41, 33, NVCV_COLOR_RGB2YUV,  NVCV_COLOR_SPEC_BT2020, false},
        { 37, 35, NVCV_COLOR_YUV2RGB,  NVCV_COLOR_SPEC_BT601,  false},
        { 31, 11, NVCV_COLOR_YUV2BGR,  NVCV_COLOR_SPEC_BT709,  true },
        { 41, 33, NVCV_COLOR_YUV2RGB,  NVCV_COLOR_SPEC_BT2020, false},
    });

// clang-format on

TEST_P(OpAdvCvtColor444PlanarExact, matches_fixed_point_reference)
{
    const int                     width  = GetParamValue<0>();
    const int                     height = GetParamValue<1>();
    const NVCVColorConversionCode code   = GetParamValue<2>();
    const nvcv::ColorSpec         colorSpec{GetParamValue<3>()};
    const bool                    rank3   = GetParamValue<4>();
    const int                     samples = rank3 ? 1 : 2;

    nvcv::Tensor src = MakeAdvCvtColorTensor(samples, width, height, 3, rank3 ? "CHW" : "NCHW");
    nvcv::Tensor dst = MakeAdvCvtColorTensor(samples, width, height, 3, rank3 ? "CHW" : "NCHW");

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess && dstAccess);

    std::vector<std::vector<uint8_t>> gold(samples);
    for (int i = 0; i < samples; ++i)
    {
        std::vector<uint8_t> input(static_cast<size_t>(width) * height * 3);
        test::planar::FillDeterministicValues(input, static_cast<size_t>(i) * 211 + 43, nvcv::TYPE_U8);
        test::planar::UploadPlanarSample(*srcAccess, i,
                                         test::planar::DeinterleaveToPlanes(input, width, height, 3, sizeof(uint8_t)),
                                         width, height, 3, sizeof(uint8_t));
        gold[i] = Convert444Exact(input, width, height, code, colorSpec);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AdvCvtColor op;
    EXPECT_NO_THROW(op(stream, src, dst, code, colorSpec));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);
        auto planes = test::planar::DownloadPlanarSample(*dstAccess, i, width, height, 3, sizeof(uint8_t));
        auto output = test::planar::InterleaveFromPlanes(planes, width, height, 3, sizeof(uint8_t));
        EXPECT_EQ(gold[i], output);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off

NVCV_TEST_SUITE_P(
    OpAdvCvtColorRGBToNVExact,
    test::ValueList<int, int, int, NVCVColorConversionCode, NVCVColorSpec, bool>{
        // W,  H, C, Conversion code,             Color spec,             NCHW
        {   2,  2, 3, NVCV_COLOR_RGB2YUV_NV12,  NVCV_COLOR_SPEC_BT601,  false},
        {  66, 18, 3, NVCV_COLOR_BGR2YUV_NV21,  NVCV_COLOR_SPEC_BT709,  false},
        {  70, 22, 4, NVCV_COLOR_BGR2YUV_NV12,  NVCV_COLOR_SPEC_BT2020, false},
        {   2,  2, 4, NVCV_COLOR_RGB2YUV_NV21,  NVCV_COLOR_SPEC_BT601,  false},
        {  66, 18, 3, NVCV_COLOR_BGR2YUV_NV12,  NVCV_COLOR_SPEC_BT709,  true },
        {   2,  2, 3, NVCV_COLOR_RGB2YUV_NV21,  NVCV_COLOR_SPEC_BT2020, true },
        {   2,  2, 4, NVCV_COLOR_RGB2YUV_NV12,  NVCV_COLOR_SPEC_BT601,  true },
        {  70, 22, 4, NVCV_COLOR_BGR2YUV_NV21,  NVCV_COLOR_SPEC_BT709,  true },
    });

// clang-format on

TEST_P(OpAdvCvtColorRGBToNVExact, matches_fixed_point_reference)
{
    const int                     width    = GetParamValue<0>();
    const int                     height   = GetParamValue<1>();
    const int                     channels = GetParamValue<2>();
    const NVCVColorConversionCode code     = GetParamValue<3>();
    const nvcv::ColorSpec         colorSpec{GetParamValue<4>()};
    const bool                    planar    = GetParamValue<5>();
    constexpr int                 samples   = 2;
    const int                     dstHeight = height * 3 / 2;

    nvcv::Tensor src = MakeAdvCvtColorTensor(samples, width, height, channels, planar ? "NCHW" : "NHWC");
    nvcv::Tensor dst = MakeAdvCvtColorTensor(samples, width, dstHeight, 1, planar ? "NCHW" : "NHWC");

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess && dstAccess);

    std::vector<std::vector<uint8_t>> gold(samples);
    for (int i = 0; i < samples; ++i)
    {
        std::vector<uint8_t> input(static_cast<size_t>(width) * height * channels);
        test::planar::FillDeterministicValues(input, static_cast<size_t>(i) * 173 + 29, nvcv::TYPE_U8);
        if (planar)
        {
            test::planar::UploadPlanarSample(
                *srcAccess, i, test::planar::DeinterleaveToPlanes(input, width, height, channels, sizeof(uint8_t)),
                width, height, channels, sizeof(uint8_t));
        }
        else
        {
            test::planar::UploadInterleavedSample(*srcAccess, i, input, width, height, width * channels);
        }
        gold[i] = ConvertRGBToNVExact(input, width, height, channels, code, colorSpec);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AdvCvtColor op;
    EXPECT_NO_THROW(op(stream, src, dst, code, colorSpec));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> output;
        if (planar)
        {
            auto planes = test::planar::DownloadPlanarSample(*dstAccess, i, width, dstHeight, 1, sizeof(uint8_t));
            output      = test::planar::InterleaveFromPlanes(planes, width, dstHeight, 1, sizeof(uint8_t));
        }
        else
        {
            output = test::planar::DownloadInterleavedSample(*dstAccess, i, width, dstHeight, width);
        }
        EXPECT_EQ(gold[i], output);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void RunMisalignedPaddedRGBToNVExact(int channels, NVCVColorConversionCode code, nvcv::ColorSpec colorSpec,
                                            size_t seed)
{
    constexpr int     width     = 66;
    constexpr int     height    = 18;
    constexpr int     samples   = 2;
    constexpr int     dstHeight = height * 3 / 2;
    const int         srcRow    = width * channels + 1;
    const int         srcSample = srcRow * height + 1;
    constexpr int     dstRow    = width + 1;
    constexpr int     dstSample = dstRow * dstHeight + 1;
    constexpr uint8_t padding   = 0xa5;

    NVCVByte                           *srcAllocation{};
    NVCVByte                           *dstAllocation{};
    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    nvcv::TensorDataStridedCuda::Buffer dstBuffer{};
    ASSERT_EQ(cudaSuccess,
              AllocateMisalignedTensorBuffer(samples, srcSample, srcRow, channels, srcAllocation, srcBuffer));
    cudaError_t status = AllocateMisalignedTensorBuffer(samples, dstSample, dstRow, 1, dstAllocation, dstBuffer);
    if (status != cudaSuccess)
    {
        cudaFree(srcAllocation);
    }
    ASSERT_EQ(cudaSuccess, status);

    nvcv::Tensor src = nvcv::TensorWrapData(
        nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{samples, height, width, channels}, "NHWC"},
            nvcv::TYPE_U8, srcBuffer
    },
        nvcv::TensorDataCleanupCallback{[srcAllocation](const nvcv::TensorData &)
                                        {
                                            cudaFree(srcAllocation);
                                        }});
    nvcv::Tensor dst = nvcv::TensorWrapData(
        nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{samples, dstHeight, width, 1}, "NHWC"},
            nvcv::TYPE_U8, dstBuffer
    },
        nvcv::TensorDataCleanupCallback{[dstAllocation](const nvcv::TensorData &)
                                        {
                                            cudaFree(dstAllocation);
                                        }});

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess && dstAccess);

    ASSERT_EQ(cudaSuccess, cudaMemset(srcAllocation, padding, static_cast<size_t>(srcSample) * samples + 1));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocation, padding, static_cast<size_t>(dstSample) * samples + 1));
    std::vector<std::vector<uint8_t>> gold(samples);
    for (int i = 0; i < samples; ++i)
    {
        std::vector<uint8_t> input(static_cast<size_t>(width) * height * channels);
        test::planar::FillDeterministicValues(input, static_cast<size_t>(i) * 173 + seed, nvcv::TYPE_U8);
        test::planar::UploadInterleavedSample(*srcAccess, i, input, width, height, width * channels);
        gold[i] = ConvertRGBToNVExact(input, width, height, channels, code, colorSpec);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AdvCvtColor op;
    EXPECT_NO_THROW(op(stream, src, dst, code, colorSpec));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);
        auto output = test::planar::DownloadInterleavedSample(*dstAccess, i, width, dstHeight, width);
        EXPECT_EQ(gold[i], output);
    }

    std::vector<uint8_t> rawOutput(static_cast<size_t>(dstSample) * samples + 1);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(rawOutput.data(), dstAllocation, rawOutput.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(padding, rawOutput[0]);
    for (int i = 0; i < samples; ++i)
    {
        for (int y = 0; y < dstHeight; ++y)
        {
            EXPECT_EQ(padding,
                      rawOutput[1 + static_cast<size_t>(i) * dstSample + static_cast<size_t>(y) * dstRow + width]);
        }
        EXPECT_EQ(padding, rawOutput[1 + static_cast<size_t>(i) * dstSample + static_cast<size_t>(dstHeight) * dstRow]);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdvCvtColorRGBToNVExact, misaligned_padded_nhwc_matches_fixed_point_reference)
{
    RunMisalignedPaddedRGBToNVExact(3, NVCV_COLOR_BGR2YUV_NV12, nvcv::ColorSpec{NVCV_COLOR_SPEC_BT2020}, 79);
    RunMisalignedPaddedRGBToNVExact(4, NVCV_COLOR_RGB2YUV_NV21, nvcv::ColorSpec{NVCV_COLOR_SPEC_BT709}, 131);
}

// clang-format off

NVCV_TEST_SUITE_P(
    OpAdvCvtColorNVToRGBExact,
    test::ValueList<int, int, int, NVCVColorConversionCode, NVCVColorSpec, bool>{
        // W,  H, C, Conversion code,             Color spec,             NCHW
        {   2,  2, 3, NVCV_COLOR_YUV2RGB_NV12,  NVCV_COLOR_SPEC_BT601,  false},
        {  66, 18, 3, NVCV_COLOR_YUV2BGR_NV21,  NVCV_COLOR_SPEC_BT709,  false},
        {  66, 20, 3, NVCV_COLOR_YUV2BGR_NV12,  NVCV_COLOR_SPEC_BT2020, false},
        {  70, 22, 4, NVCV_COLOR_YUV2BGR_NV12,  NVCV_COLOR_SPEC_BT2020, false},
        {   2,  2, 4, NVCV_COLOR_YUV2RGB_NV21,  NVCV_COLOR_SPEC_BT601,  false},
        {  66, 18, 3, NVCV_COLOR_YUV2BGR_NV12,  NVCV_COLOR_SPEC_BT709,  true },
        {   2,  2, 3, NVCV_COLOR_YUV2RGB_NV21,  NVCV_COLOR_SPEC_BT2020, true },
        {   2,  2, 4, NVCV_COLOR_YUV2RGB_NV12,  NVCV_COLOR_SPEC_BT601,  true },
        {  70, 22, 4, NVCV_COLOR_YUV2BGR_NV21,  NVCV_COLOR_SPEC_BT709,  true },
    });

// clang-format on

TEST_P(OpAdvCvtColorNVToRGBExact, matches_fixed_point_reference)
{
    const int                     width    = GetParamValue<0>();
    const int                     height   = GetParamValue<1>();
    const int                     channels = GetParamValue<2>();
    const NVCVColorConversionCode code     = GetParamValue<3>();
    const nvcv::ColorSpec         colorSpec{GetParamValue<4>()};
    const bool                    planar    = GetParamValue<5>();
    constexpr int                 samples   = 2;
    const int                     srcHeight = height * 3 / 2;

    nvcv::Tensor src = MakeAdvCvtColorTensor(samples, width, srcHeight, 1, planar ? "NCHW" : "NHWC");
    nvcv::Tensor dst = MakeAdvCvtColorTensor(samples, width, height, channels, planar ? "NCHW" : "NHWC");

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess && dstAccess);

    std::vector<std::vector<uint8_t>> gold(samples);
    for (int i = 0; i < samples; ++i)
    {
        std::vector<uint8_t> input(static_cast<size_t>(width) * srcHeight);
        test::planar::FillDeterministicValues(input, static_cast<size_t>(i) * 191 + 37, nvcv::TYPE_U8);
        if (planar)
        {
            test::planar::UploadPlanarSample(*srcAccess, i, {input}, width, srcHeight, 1, sizeof(uint8_t));
        }
        else
        {
            test::planar::UploadInterleavedSample(*srcAccess, i, input, width, srcHeight, width);
        }
        gold[i] = ConvertNVToRGBExact(input, width, height, channels, code, colorSpec);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AdvCvtColor op;
    EXPECT_NO_THROW(op(stream, src, dst, code, colorSpec));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> output;
        if (planar)
        {
            auto planes = test::planar::DownloadPlanarSample(*dstAccess, i, width, height, channels, sizeof(uint8_t));
            output      = test::planar::InterleaveFromPlanes(planes, width, height, channels, sizeof(uint8_t));
        }
        else
        {
            output = test::planar::DownloadInterleavedSample(*dstAccess, i, width, height, width * channels);
        }
        EXPECT_EQ(gold[i], output);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void RunMisalignedPaddedNVToRGBExact(int channels, NVCVColorConversionCode code, nvcv::ColorSpec colorSpec,
                                            size_t seed)
{
    constexpr int     width     = 66;
    constexpr int     height    = 18;
    constexpr int     samples   = 2;
    constexpr int     srcHeight = height * 3 / 2;
    constexpr int     srcRow    = width + 1;
    constexpr int     srcSample = srcRow * srcHeight + 1;
    const int         dstRow    = width * channels + 1;
    const int         dstSample = dstRow * height + 1;
    constexpr uint8_t padding   = 0xa5;

    NVCVByte                           *srcAllocation{};
    NVCVByte                           *dstAllocation{};
    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    nvcv::TensorDataStridedCuda::Buffer dstBuffer{};
    ASSERT_EQ(cudaSuccess, AllocateMisalignedTensorBuffer(samples, srcSample, srcRow, 1, srcAllocation, srcBuffer));
    cudaError_t status = AllocateMisalignedTensorBuffer(samples, dstSample, dstRow, channels, dstAllocation, dstBuffer);
    if (status != cudaSuccess)
    {
        cudaFree(srcAllocation);
    }
    ASSERT_EQ(cudaSuccess, status);

    nvcv::Tensor src = nvcv::TensorWrapData(
        nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{samples, srcHeight, width, 1}, "NHWC"},
            nvcv::TYPE_U8, srcBuffer
    },
        nvcv::TensorDataCleanupCallback{[srcAllocation](const nvcv::TensorData &)
                                        {
                                            cudaFree(srcAllocation);
                                        }});
    nvcv::Tensor dst = nvcv::TensorWrapData(
        nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{samples, height, width, channels}, "NHWC"},
            nvcv::TYPE_U8, dstBuffer
    },
        nvcv::TensorDataCleanupCallback{[dstAllocation](const nvcv::TensorData &)
                                        {
                                            cudaFree(dstAllocation);
                                        }});

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess && dstAccess);

    ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocation, padding, static_cast<size_t>(dstSample) * samples + 1));
    std::vector<std::vector<uint8_t>> gold(samples);
    for (int i = 0; i < samples; ++i)
    {
        std::vector<uint8_t> input(static_cast<size_t>(width) * srcHeight);
        test::planar::FillDeterministicValues(input, static_cast<size_t>(i) * 191 + seed, nvcv::TYPE_U8);
        test::planar::UploadInterleavedSample(*srcAccess, i, input, width, srcHeight, width);
        gold[i] = ConvertNVToRGBExact(input, width, height, channels, code, colorSpec);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AdvCvtColor op;
    EXPECT_NO_THROW(op(stream, src, dst, code, colorSpec));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);
        auto output = test::planar::DownloadInterleavedSample(*dstAccess, i, width, height, width * channels);
        EXPECT_EQ(gold[i], output);
    }

    std::vector<uint8_t> rawOutput(static_cast<size_t>(dstSample) * samples + 1);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(rawOutput.data(), dstAllocation, rawOutput.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(padding, rawOutput[0]);
    for (int i = 0; i < samples; ++i)
    {
        for (int y = 0; y < height; ++y)
        {
            EXPECT_EQ(
                padding,
                rawOutput[1 + static_cast<size_t>(i) * dstSample + static_cast<size_t>(y) * dstRow + width * channels]);
        }
        EXPECT_EQ(padding, rawOutput[1 + static_cast<size_t>(i) * dstSample + static_cast<size_t>(height) * dstRow]);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdvCvtColorNVToRGBExact, misaligned_padded_nhwc_matches_fixed_point_reference)
{
    RunMisalignedPaddedNVToRGBExact(3, NVCV_COLOR_YUV2RGB_NV21, nvcv::ColorSpec{NVCV_COLOR_SPEC_BT2020}, 73);
    RunMisalignedPaddedNVToRGBExact(3, NVCV_COLOR_YUV2BGR_NV12, nvcv::ColorSpec{NVCV_COLOR_SPEC_BT709}, 97);
}

TEST(OpAdvCvtColorNVToRGBExact, misaligned_padded_nhwc_c4_matches_fixed_point_reference)
{
    RunMisalignedPaddedNVToRGBExact(4, NVCV_COLOR_YUV2BGR_NV12, nvcv::ColorSpec{NVCV_COLOR_SPEC_BT709}, 113);
}

// clang-format off
// Max diff is 2.0 for 8-bit images is defined here since the conversion in the test is using floats directly and not integers.

NVCV_TEST_SUITE_P(OpAdvCvtColor, test::ValueList<int, int, int, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode, NVCVColorSpec, float>
{
  //inWidth, inHeight, numberInBatch,              In format,                  Out format,                     in2outCode,             colorSpec,  maxDiff

//Nv12/24 must be even w/h
    {     4,       40,              1, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_COLOR_RGB2YUV_NV12,  NVCV_COLOR_SPEC_BT601,  2.0},
    {   100,      440,              2, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_COLOR_RGB2YUV_NV12,  NVCV_COLOR_SPEC_BT709,  2.0},
    {   346,      672,              4, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_COLOR_RGB2YUV_NV12,  NVCV_COLOR_SPEC_BT2020, 2.0},

    {     2,      300,              2, NVCV_IMAGE_FORMAT_BGR8,       NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_COLOR_BGR2YUV_NV12,  NVCV_COLOR_SPEC_BT601,  2.0},
    {    74,       28,              3, NVCV_IMAGE_FORMAT_BGR8,       NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_COLOR_BGR2YUV_NV12,  NVCV_COLOR_SPEC_BT709,  2.0},
    {   720,      400,              2, NVCV_IMAGE_FORMAT_BGR8,       NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_COLOR_BGR2YUV_NV12,  NVCV_COLOR_SPEC_BT2020, 2.0},

    {    66,       48,              1, NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB_NV12,  NVCV_COLOR_SPEC_BT601,  2.0},
    {   536,      422,              2, NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB_NV12,  NVCV_COLOR_SPEC_BT709,  2.0},
    {   400,        4,              5, NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB_NV12,  NVCV_COLOR_SPEC_BT2020, 2.0},

    {     2,        2,              2, NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR_NV12,  NVCV_COLOR_SPEC_BT601,  2.0},
    {    56,       42,              1, NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR_NV12,  NVCV_COLOR_SPEC_BT709,  2.0},
    {     4,      108,              3, NVCV_IMAGE_FORMAT_NV12_ER,       NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR_NV12,  NVCV_COLOR_SPEC_BT2020, 2.0},

    {     4,       40,              1, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_NV21_ER,       NVCV_COLOR_RGB2YUV_NV21,  NVCV_COLOR_SPEC_BT601,  2.0},
    {   100,      440,              2, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_NV21_ER,       NVCV_COLOR_RGB2YUV_NV21,  NVCV_COLOR_SPEC_BT709,  2.0},
    {   346,      672,              4, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_NV21_ER,       NVCV_COLOR_RGB2YUV_NV21,  NVCV_COLOR_SPEC_BT2020, 2.0},

    {     2,      300,              2, NVCV_IMAGE_FORMAT_BGR8,      NVCV_IMAGE_FORMAT_NV21_ER,       NVCV_COLOR_BGR2YUV_NV21,  NVCV_COLOR_SPEC_BT601,  2.0},
    {    74,       28,              3, NVCV_IMAGE_FORMAT_BGR8,      NVCV_IMAGE_FORMAT_NV21_ER,       NVCV_COLOR_BGR2YUV_NV21,  NVCV_COLOR_SPEC_BT709,  2.0},
    {   720,      400,              2, NVCV_IMAGE_FORMAT_BGR8,      NVCV_IMAGE_FORMAT_NV21_ER,       NVCV_COLOR_BGR2YUV_NV21,  NVCV_COLOR_SPEC_BT2020, 2.0},

    {    66,       48,              1, NVCV_IMAGE_FORMAT_NV21_ER,      NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB_NV21,  NVCV_COLOR_SPEC_BT601,  2.0},
    {   536,      422,              2, NVCV_IMAGE_FORMAT_NV21_ER,      NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB_NV21,  NVCV_COLOR_SPEC_BT709,  2.0},
    {   400,        4,              5, NVCV_IMAGE_FORMAT_NV21_ER,      NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB_NV21,  NVCV_COLOR_SPEC_BT2020, 2.0},

    {     2,        2,              2, NVCV_IMAGE_FORMAT_NV21_ER,      NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR_NV21,  NVCV_COLOR_SPEC_BT601,  2.0},
    {    56,       42,              1, NVCV_IMAGE_FORMAT_NV21_ER,      NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR_NV21,  NVCV_COLOR_SPEC_BT709,  2.0},
    {     4,      108,              3, NVCV_IMAGE_FORMAT_NV21_ER,      NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR_NV21,  NVCV_COLOR_SPEC_BT2020, 2.0},


//YUV can be odd dims
    {    321,       24,             2, NVCV_IMAGE_FORMAT_YUV8,      NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB,    NVCV_COLOR_SPEC_BT601,  2.0},
    {     85,       27,             1, NVCV_IMAGE_FORMAT_YUV8,      NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB,    NVCV_COLOR_SPEC_BT709,  2.0},
    {     21,       22,             4, NVCV_IMAGE_FORMAT_YUV8,      NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB,    NVCV_COLOR_SPEC_BT2020, 2.0},

    {      3,      124,             1, NVCV_IMAGE_FORMAT_YUV8,      NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR,    NVCV_COLOR_SPEC_BT601,  2.0},
    {    131,      239,             2, NVCV_IMAGE_FORMAT_YUV8,      NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR,    NVCV_COLOR_SPEC_BT709,  2.0},
    {     45,       45,             1, NVCV_IMAGE_FORMAT_YUV8,      NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR,    NVCV_COLOR_SPEC_BT2020, 2.0},

    {   1080,        1,             5, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_YUV8,       NVCV_COLOR_RGB2YUV,    NVCV_COLOR_SPEC_BT601,  2.0},
    {     42,        2,             7, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_YUV8,       NVCV_COLOR_RGB2YUV,    NVCV_COLOR_SPEC_BT709,  2.0},
    {    340,      620,            10, NVCV_IMAGE_FORMAT_RGB8,      NVCV_IMAGE_FORMAT_YUV8,       NVCV_COLOR_RGB2YUV,    NVCV_COLOR_SPEC_BT2020, 2.0},

    {      3,        3,             1, NVCV_IMAGE_FORMAT_BGR8,      NVCV_IMAGE_FORMAT_YUV8,       NVCV_COLOR_BGR2YUV,    NVCV_COLOR_SPEC_BT601,  2.0},
    {     43,      208,             2, NVCV_IMAGE_FORMAT_BGR8,      NVCV_IMAGE_FORMAT_YUV8,       NVCV_COLOR_BGR2YUV,    NVCV_COLOR_SPEC_BT709,  2.0},
    {    340,      220,             1, NVCV_IMAGE_FORMAT_BGR8,      NVCV_IMAGE_FORMAT_YUV8,       NVCV_COLOR_BGR2YUV,    NVCV_COLOR_SPEC_BT2020, 2.0},

});

// clang-format on

// clang-format off

NVCV_TEST_SUITE_P(OpAdvCvtColorPlanarTensor,
test::ValueList<int, int, int, int, int, NVCVColorConversionCode, NVCVColorSpec, bool>
{
    //  W,  H,  N, SrcC, DstC, Conversion Code,          Color Spec,             Rank-3 CHW/HWC
    { 31, 23, 2,    3,    3,  NVCV_COLOR_RGB2YUV,       NVCV_COLOR_SPEC_BT601,  false},
    { 29, 17, 1,    3,    3,  NVCV_COLOR_YUV2BGR,       NVCV_COLOR_SPEC_BT709,  true},
    { 32, 24, 2,    3,    1,  NVCV_COLOR_RGB2YUV_NV12,  NVCV_COLOR_SPEC_BT2020, false},
    { 34, 22, 1,    4,    1,  NVCV_COLOR_BGR2YUV_NV21,  NVCV_COLOR_SPEC_BT601,  true},
    { 30, 20, 2,    1,    3,  NVCV_COLOR_YUV2RGB_NV12,  NVCV_COLOR_SPEC_BT709,  false},
    { 28, 18, 1,    1,    4,  NVCV_COLOR_YUV2BGR_NV21,  NVCV_COLOR_SPEC_BT2020, true},
});

// clang-format on

TEST_P(OpAdvCvtColorPlanarTensor, tensor_matches_interleaved)
{
    RunAdvCvtColorTensorPlanarParity(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                                     GetParamValue<4>(), GetParamValue<5>(), nvcv::ColorSpec{GetParamValue<6>()},
                                     GetParamValue<7>());
}

TEST_P(OpAdvCvtColor, AdvCvtColor_sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                     width   = GetParamValue<0>();
    int                     height  = GetParamValue<1>();
    int                     batches = GetParamValue<2>();
    nvcv::ImageFormat       formatIn{GetParamValue<3>()};
    nvcv::ImageFormat       formatOut{GetParamValue<4>()};
    NVCVColorConversionCode convCode = GetParamValue<5>();
    nvcv::ColorSpec         colorSpec{GetParamValue<6>()};
    float                   maxDiff = GetParamValue<7>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, formatIn);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, formatOut);

    // NV12/21/YUV8/ARGB are all nHWC
    auto   colorChannels  = static_cast<int>(inTensor.shape()[inTensor.shape().rank() - 1]);
    auto   tensorHeight   = static_cast<int>(inTensor.shape()[inTensor.shape().rank() - 3]);
    size_t imageSizeBytes = static_cast<size_t>(width) * static_cast<size_t>(tensorHeight)
                          * static_cast<size_t>(colorChannels) * sizeof(uint8_t);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution randomByte(0, 255);

    std::vector<uint8_t> imageVec(imageSizeBytes, 128);
    for (int i = 0; i < batches; ++i)
    {
        // generate random input image
        for (uint8_t &value : imageVec)
        {
            value = static_cast<uint8_t>(randomByte(randEng));
        }
        // copy random input to device tensor
        util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, i);
    }

    // run operator
    cvcuda::AdvCvtColor op;
    EXPECT_NO_THROW(op(stream, inTensor, outTensor, convCode, colorSpec));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    verifyOutput(inTensor, outTensor, {batches, convCode, colorSpec, maxDiff, width, height});
}

// clang-format off
// inputNumSamples, outputNumSamples, inWidth, inHeight, outWidth, outHeight, inFormat, outFormat, convCode, colorSpec
NVCV_TEST_SUITE_P(OpAdvCvtColor_Negative, test::ValueList<int, int, int, int, int, int, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode, NVCVColorSpec>
{
    {2, 2, 400, 400, 400, 400, NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_RGB8, NVCV_COLOR_RGBA2RGB, NVCV_COLOR_SPEC_BT601},
    {2, 2, 400, 400, 400, 400, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_YUV8, NVCV_COLOR_RGB2YUV, NVCV_COLOR_SPEC_sRGB},
    {2, 2, 400, 400, 400, 400, NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_YUV8, NVCV_COLOR_RGB2YUV, NVCV_COLOR_SPEC_BT601},
    {3, 2, 400, 400, 400, 400, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_YUV8, NVCV_COLOR_RGB2YUV, NVCV_COLOR_SPEC_BT601},
    {2, 2, 400, 500, 400, 400, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_YUV8, NVCV_COLOR_RGB2YUV, NVCV_COLOR_SPEC_BT601},
    {2, 2, 500, 400, 400, 400, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_YUV8, NVCV_COLOR_RGB2YUV, NVCV_COLOR_SPEC_BT601},
    {2, 2, 300, 300, 300, 300, NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_YUV8, NVCV_COLOR_RGB2YUV, NVCV_COLOR_SPEC_BT601},
    {2, 2, 400, 400, 400, 400, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_U8, NVCV_COLOR_YUV2BGR_NV12, NVCV_COLOR_SPEC_BT601},
    {2, 2, 500, 400, 400, 400, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_U8, NVCV_COLOR_YUV2BGR_NV12, NVCV_COLOR_SPEC_BT601},
    {2, 2, 300, 300, 300, 300, NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_COLOR_BGR2YUV_NV12, NVCV_COLOR_SPEC_BT601},
    {2, 2, 301, 300, 300, 300, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_COLOR_BGR2YUV_NV12, NVCV_COLOR_SPEC_BT601},
    {2, 2, 300, 301, 300, 300, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_COLOR_BGR2YUV_NV12, NVCV_COLOR_SPEC_BT601},
    {2, 2, 302, 300, 300, 300, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_NV12_ER, NVCV_COLOR_BGR2YUV_NV12, NVCV_COLOR_SPEC_BT601},

});

// clang-format on

TEST_P(OpAdvCvtColor_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                     inputNumSamples  = GetParamValue<0>();
    int                     outputNumSamples = GetParamValue<1>();
    int                     inWidth          = GetParamValue<2>();
    int                     inHeight         = GetParamValue<3>();
    int                     outWidth         = GetParamValue<4>();
    int                     outHeight        = GetParamValue<5>();
    nvcv::ImageFormat       inFormat{GetParamValue<6>()};
    nvcv::ImageFormat       outFormat{GetParamValue<7>()};
    NVCVColorConversionCode convCode = GetParamValue<8>();
    nvcv::ColorSpec         colorSpec{GetParamValue<9>()};

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(inputNumSamples, inWidth, inHeight, inFormat);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(outputNumSamples, outWidth, outHeight, outFormat);

    // run operator
    cvcuda::AdvCvtColor op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &inTensor, &outTensor, &convCode, &colorSpec]
                                { op(stream, inTensor, outTensor, convCode, colorSpec); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdvCvtColor_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaAdvCvtColorCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
