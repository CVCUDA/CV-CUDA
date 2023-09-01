/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/ValueTests.hpp>
#include <cvcuda/OpAdvCvtColor.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/SaturateCast.hpp>
#include <nvcv/cuda/math/LinAlg.hpp>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>

namespace gt    = ::testing;
namespace test  = nvcv::test;
namespace util  = nvcv::util;
namespace mmath = nvcv::cuda::math;

#define NVCV_IMAGE_FORMAT_YUV8 NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8_Z8)

template<class T>
using Matrix3x3 = mmath::Matrix<T, 3, 3>;

static const Matrix3x3<float> getRGB2YUVMatrix(nvcv::ColorSpec spec)
{
    Matrix3x3<float> matrix;
    switch (spec)
    {
    case NVCV_COLOR_SPEC_BT601:
    {
        const float values[] = {0.299, 0.587, 0.114, -0.168736, -0.331264, 0.5, 0.5, -0.418688, -0.0813124};
        matrix.load(values);
        return matrix;
    }
    case NVCV_COLOR_SPEC_BT709:
    {
        const float values[] = {0.2126, 0.7152, 0.0722, -0.114572, -0.385428, 0.5, 0.5, -0.454153, -0.0458471};
        matrix.load(values);
        return matrix;
    }
    case NVCV_COLOR_SPEC_BT2020:
    {
        const float values[] = {0.2627, 0.678, 0.0593, -0.13963, -0.36037, 0.5, 0.5, -0.459786, -0.0402143};
        matrix.load(values);
        return matrix;
    }
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unknown color spec");
    }
}

static const Matrix3x3<float> getYUV2RGBMatrix(nvcv::ColorSpec spec)
{
    Matrix3x3<float> matrix = getRGB2YUVMatrix(spec);
    mmath::inv_inplace<float>(matrix);
    return matrix;
}

static bool isBGR(NVCVColorConversionCode &code)
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

static bool isYVU(NVCVColorConversionCode &code)
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

        int R = conversionMatrix[0][0] * Y + conversionMatrix[0][1] * (U - 128) + conversionMatrix[0][2] * (V - 128);
        int G = conversionMatrix[1][0] * Y + conversionMatrix[1][1] * (U - 128) + conversionMatrix[1][2] * (V - 128);
        int B = conversionMatrix[2][0] * Y + conversionMatrix[2][1] * (U - 128) + conversionMatrix[2][2] * (V - 128);

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
    yuvData.reserve(yuvData.size()); // Reserve space for RGB data

    for (size_t i = 0; i < rgbData.size(); i += 3)
    {
        int R = rgbData[i];
        int G = rgbData[i + 1];
        int B = rgbData[i + 2];
        if (bgr)
        {
            std::swap(R, B);
        }

        int Y = conversionMatrix[0][0] * R + conversionMatrix[0][1] * G + conversionMatrix[0][2] * B;
        int U = conversionMatrix[1][0] * R + conversionMatrix[1][1] * G + conversionMatrix[1][2] * B + 128;
        int V = conversionMatrix[2][0] * R + conversionMatrix[2][1] * G + conversionMatrix[2][2] * B + 128;

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

        int Y = conversionMatrix[0][0] * R + conversionMatrix[0][1] * G + conversionMatrix[0][2] * B;
        nv12Data.push_back(nvcv::cuda::SaturateCast<uint8_t>(Y));
    }

    // Calculate U and V values for each 2x2 block and store them interleaved.
    for (int h = 0; h < height; h += 2)
    {
        for (int w = 0; w < width; w += 2)
        {
            int U_sum = 0;
            int V_sum = 0;

            // Loop through the 2x2 block to compute average U and V
            for (int y = 0; y < 2; ++y)
            {
                for (int x = 0; x < 2; ++x)
                {
                    int idx = ((h + y) * width + (w + x)) * 3;
                    int R   = rgbData[idx];
                    int G   = rgbData[idx + 1];
                    int B   = rgbData[idx + 2];
                    if (bgr)
                    {
                        std::swap(R, B);
                    }

                    U_sum += conversionMatrix[1][0] * R + conversionMatrix[1][1] * G + conversionMatrix[1][2] * B;
                    V_sum += conversionMatrix[2][0] * R + conversionMatrix[2][1] * G + conversionMatrix[2][2] * B;
                }
            }

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

            int R
                = conversionMatrix[0][0] * Y + conversionMatrix[0][1] * (U - 128) + conversionMatrix[0][2] * (V - 128);
            int G
                = conversionMatrix[1][0] * Y + conversionMatrix[1][1] * (U - 128) + conversionMatrix[1][2] * (V - 128);
            int B
                = conversionMatrix[2][0] * Y + conversionMatrix[2][1] * (U - 128) + conversionMatrix[2][2] * (V - 128);

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

static void verifyOutput(int batches, nvcv::Tensor &inTensor, nvcv::Tensor &outTensor, nvcv::ImageFormat format,
                         NVCVColorConversionCode convCode, nvcv::ColorSpec colorSpec, float maxDiff, int width,
                         int height)
{
    for (int i = 0; i < batches; ++i)
    {
        std::vector<uint8_t> outData, inData;

        // get 0th sample since histogram is just a 2d tensor
        EXPECT_NO_THROW(util::GetImageVectorFromTensor(inTensor.exportData(), i, inData));
        EXPECT_NO_THROW(util::GetImageVectorFromTensor(outTensor.exportData(), i, outData));

        switch (convCode)
        {
        case NVCV_COLOR_BGR2YUV:
        case NVCV_COLOR_RGB2YUV:
        {
            std::vector<uint8_t> goldOut = convertRGBtoYUV(getRGB2YUVMatrix(colorSpec), inData, isBGR(convCode));
            VEC_EXPECT_NEAR(goldOut, outData, maxDiff);
            break;
        }
        case NVCV_COLOR_YUV2BGR:
        case NVCV_COLOR_YUV2RGB:
        {
            std::vector<uint8_t> goldOut = convertYUVtoRGB(getYUV2RGBMatrix(colorSpec), inData, isBGR(convCode));
            VEC_EXPECT_NEAR(goldOut, outData, maxDiff);
            break;
        }
        case NVCV_COLOR_YUV2RGB_NV12:
        case NVCV_COLOR_YUV2BGR_NV12:
        case NVCV_COLOR_YUV2RGB_NV21:
        case NVCV_COLOR_YUV2BGR_NV21:
        {
            std::vector<uint8_t> goldOut = convertNV12toRGB(getYUV2RGBMatrix(colorSpec), inData, width, height,
                                                            isBGR(convCode), isYVU(convCode));
            VEC_EXPECT_NEAR(goldOut, outData, maxDiff);
            break;
        }
        case NVCV_COLOR_RGB2YUV_NV21:
        case NVCV_COLOR_BGR2YUV_NV21:
        case NVCV_COLOR_RGB2YUV_NV12:
        case NVCV_COLOR_BGR2YUV_NV12:
        {
            std::vector<uint8_t> goldOut = convertRGBtoNV12(getRGB2YUVMatrix(colorSpec), inData, width, height,
                                                            isBGR(convCode), isYVU(convCode));
            VEC_EXPECT_NEAR(goldOut, outData, maxDiff);
            break;
        }
        default:
            FAIL() << "Unsupported conversion code";
            break;
        }
    }
}

// clang-format off
// Max diff is 2.0 for 8-bit images is defined here since the conversion in the test is using floats directly and not integers.

NVCV_TEST_SUITE_P(OpAdvCvtColor, test::ValueList<int, int, int, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode, nvcv::ColorSpec, int>
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

TEST_P(OpAdvCvtColor, AdvCvtColor_sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                     width   = GetParamValue<0>();
    int                     height  = GetParamValue<1>();
    int                     batches = GetParamValue<2>();
    nvcv::ImageFormat       formatIn{GetParamValue<3>()};
    nvcv::ImageFormat       formatOut{GetParamValue<4>()};
    NVCVColorConversionCode convCode  = GetParamValue<5>();
    nvcv::ColorSpec         colorSpec = GetParamValue<6>();
    int                     maxDiff   = GetParamValue<7>();

    nvcv::Tensor inTensor      = nvcv::util::CreateTensor(batches, width, height, formatIn);
    nvcv::Tensor outTensor     = nvcv::util::CreateTensor(batches, width, height, formatOut);
    nvcv::Tensor outBackTensor = nvcv::util::CreateTensor(batches, width, height, formatIn);

    // NV12/21/YUV8/ARGB are all nHWC
    int    colorChannels  = inTensor.shape()[inTensor.shape().rank() - 1];
    int    tensorHeight   = inTensor.shape()[inTensor.shape().rank() - 3];
    size_t imageSizeBytes = width * tensorHeight * colorChannels * sizeof(uint8_t);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::vector<uint8_t> imageVec(imageSizeBytes, 128);
    for (int i = 0; i < batches; ++i)
    {
        // generate random input image
        std::generate(imageVec.begin(), imageVec.end(), [&]() { return rand(randEng); });
        // copy random input to device tensor
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, i));
    }

    // run operator
    cvcuda::AdvCvtColor op;
    EXPECT_NO_THROW(op(stream, inTensor, outTensor, convCode, colorSpec));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    verifyOutput(batches, inTensor, outTensor, formatOut, convCode, colorSpec, maxDiff, width, height);
}
