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

#include "ConvUtils.hpp"
#include "Definitions.hpp"
#include "PlanarParityUtils.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpGammaContrast.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

constexpr bool DBG_GAMMA_CONTRAST = false;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

static void printChannelRow(const std::vector<uint8_t> &vec, int row, int rowPitch, int bytesPerPixel, int channel)
{
    for (int col = 0; col < rowPitch / bytesPerPixel; col++)
    {
        printf("%4d, ", static_cast<int>(vec[row * rowPitch + col * bytesPerPixel + channel]));
    }
    std::cout << std::endl;
}

static void printChannel(const std::vector<uint8_t> &vec, int height, int rowPitch, int bytesPerPixel, int channel,
                         const std::string &name)
{
    std::cout << "\nPrint " << name << " for channel: " << channel << std::endl;

    for (int row = 0; row < height; row++)
    {
        printChannelRow(vec, row, rowPitch, bytesPerPixel, channel);
    }
}

static void printVec(const std::vector<uint8_t> &vec, int height, int rowPitch, int bytesPerPixel,
                     const std::string &name)
{
    if constexpr (DBG_GAMMA_CONTRAST)
    {
        for (int channel = 0; channel < bytesPerPixel; channel++)
        {
            printChannel(vec, height, rowPitch, bytesPerPixel, channel, name);
        }
        std::cout << std::endl;
    }
}

// The tolerance accounts for the host std::pow/std::rint reference versus device __powf/SaturateCast rounding.
#define VEC_EXPECT_NEAR(vec1, vec2, delta, dtype)                                                                    \
    ASSERT_EQ(vec1.size(), vec2.size());                                                                             \
    for (std::size_t idx = 0; idx < vec1.size() / sizeof(dtype); ++idx)                                              \
    {                                                                                                                \
        EXPECT_NEAR(reinterpret_cast<dtype *>(vec1.data())[idx], reinterpret_cast<dtype *>(vec2.data())[idx], delta) \
            << "At index " << idx;                                                                                   \
    }

namespace {

// uint8 cpu op
template<typename T>
void GammaContrastVarShapeCpuOp(std::vector<T> &hDst, int dstRowStride, nvcv::Size2D dstSize,
                                const std::vector<T> &hSrc, int, nvcv::Size2D, nvcv::ImageFormat fmt,
                                const std::vector<float> &gamma, const int imageIndex, bool perChannel)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            for (int k = 0; k < elementsPerPixel; k++)
            {
                int   index     = dst_y * dstRowStride + dst_x * elementsPerPixel + k;
                float gamma_tmp = perChannel ? gamma[imageIndex * elementsPerPixel + k] : gamma[imageIndex];
                float tmp       = (srcPtr[index] + 0.0f) / 255.0f;
                auto  out       = static_cast<T>(std::rint(pow(tmp, gamma_tmp) * 255.0f));
                dstPtr[index]   = out;
            }
        }
    }
}

// float cpu op
template<>
void GammaContrastVarShapeCpuOp(std::vector<float> &hDst, int dstRowStride, nvcv::Size2D dstSize,
                                const std::vector<float> &hSrc, int, nvcv::Size2D, nvcv::ImageFormat fmt,
                                const std::vector<float> &gamma, const int imageIndex, bool perChannel)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            for (int k = 0; k < elementsPerPixel; k++)
            {
                int   index     = dst_y * dstRowStride + dst_x * elementsPerPixel + k;
                float gamma_tmp = perChannel ? gamma[imageIndex * elementsPerPixel + k] : gamma[imageIndex];
                float out       = nvcv::cuda::clamp(nvcv::cuda::pow(hSrc[index], gamma_tmp), 0.f, 1.f);
                hDst[index]     = out;
            }
        }
    }
}

void GammaContrastVarShapeCpuOpWrapper(std::vector<uint8_t> &hDst, int dstRowStride, nvcv::Size2D dstSize,
                                       const std::vector<uint8_t> &hSrc, int srcRowStride, nvcv::Size2D srcSize,
                                       nvcv::ImageFormat fmt, const std::vector<float> &gamma, const int imageIndex,
                                       bool perChannel, NVCVDataType nvcvDataType)
{
    if (nvcvDataType == NVCV_DATA_TYPE_F32 || nvcvDataType == NVCV_DATA_TYPE_2F32 || nvcvDataType == NVCV_DATA_TYPE_3F32
        || nvcvDataType == NVCV_DATA_TYPE_4F32)
    {
        std::vector<float> src_tmp(hSrc.size() / sizeof(float));
        std::vector<float> dst_tmp(hDst.size() / sizeof(float));
        size_t             copySize = hSrc.size();
        memcpy(static_cast<void *>(src_tmp.data()), static_cast<const void *>(hSrc.data()), copySize);
        memcpy(static_cast<void *>(dst_tmp.data()), static_cast<void *>(hDst.data()), copySize);
        GammaContrastVarShapeCpuOp(dst_tmp, dstRowStride / sizeof(float), dstSize, src_tmp,
                                   srcRowStride / sizeof(float), srcSize, fmt, gamma, imageIndex, perChannel);
        memcpy(static_cast<void *>(hDst.data()), static_cast<void *>(dst_tmp.data()), copySize);
    }
    else
    {
        GammaContrastVarShapeCpuOp(hDst, dstRowStride, dstSize, hSrc, srcRowStride, srcSize, fmt, gamma, imageIndex,
                                   perChannel);
    }
}

} // namespace

// clang-format off

NVCV_TEST_SUITE_P(OpGammaContrast, test::ValueList<int, int, int, NVCVImageFormat, float, bool>
{
    // width, height, batches,                    format,  Gamma,  per channel
    {   5,      5,       1,      NVCV_IMAGE_FORMAT_U8,       0.5,        true},
    {   9,     11,       2,      NVCV_IMAGE_FORMAT_U8,      0.75,        true},
    {  37,      5,       2,      NVCV_IMAGE_FORMAT_U8,      0.65,        true},
    {   12,     7,       3,    NVCV_IMAGE_FORMAT_RGB8,       1.0,        true},
    {   11,    11,       4,   NVCV_IMAGE_FORMAT_RGBA8,       0.4,        true},
    {   7,      8,       3,    NVCV_IMAGE_FORMAT_RGB8,       0.9,        true},
    {   7,      6,       4,   NVCV_IMAGE_FORMAT_RGBA8,       0.8,        true},

    {   5,      5,       1,      NVCV_IMAGE_FORMAT_U8,        0.5,      false},
    {   9,     11,       2,      NVCV_IMAGE_FORMAT_U8,       0.75,      false},
    {   12,     7,       3,    NVCV_IMAGE_FORMAT_RGB8,        1.0,      false},
    {   11,    11,       4,   NVCV_IMAGE_FORMAT_RGBA8,        0.4,      false},
    {   7,      8,       3,    NVCV_IMAGE_FORMAT_RGB8,        0.9,      false},
    {   7,      6,       4,   NVCV_IMAGE_FORMAT_RGBA8,        0.8,      false},

    {   5,      5,       1,     NVCV_IMAGE_FORMAT_F32,       0.5,        true},
    {   9,     11,       2,     NVCV_IMAGE_FORMAT_F32,      0.75,        true},
    {   12,     7,       3,  NVCV_IMAGE_FORMAT_RGBf32,       1.0,        true},
    {   11,    11,       4, NVCV_IMAGE_FORMAT_RGBAf32,       0.4,        true},
    {   7,      8,       3,  NVCV_IMAGE_FORMAT_RGBf32,       0.9,        true},
    {   7,      6,       4, NVCV_IMAGE_FORMAT_RGBAf32,       0.8,        true},

    {   5,      5,       1,     NVCV_IMAGE_FORMAT_F32,        0.5,      false},
    {   9,     11,       2,     NVCV_IMAGE_FORMAT_F32,       0.75,      false},
    {   12,     7,       3,  NVCV_IMAGE_FORMAT_RGBf32,        1.0,      false},
    {   11,    11,       4, NVCV_IMAGE_FORMAT_RGBAf32,        0.4,      false},
    {   7,      8,       3,  NVCV_IMAGE_FORMAT_RGBf32,        0.9,      false},
    {   7,      6,       4, NVCV_IMAGE_FORMAT_RGBAf32,        0.8,      false},
});

// clang-format on

TEST_P(OpGammaContrast, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    NVCVDataType nvcvDataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(format), 0, &nvcvDataType));
    float gamma       = GetParamValue<4>();
    bool  isFloatTest = false;

    bool perChannel = GetParamValue<5>();

    // Create input varshape
    std::default_random_engine            rng;
    std::uniform_int_distribution         udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution         udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));
    std::uniform_real_distribution<float> udistGamma(gamma * 0.8f, 1.f);

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);
        std::uniform_real_distribution<float>  udistf(0.f, 1.f);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        switch (nvcvDataType)
        {
        case NVCV_DATA_TYPE_F32:
        case NVCV_DATA_TYPE_2F32:
        case NVCV_DATA_TYPE_3F32:
        case NVCV_DATA_TYPE_4F32:
            isFloatTest = true;
            for (size_t idx = 0; idx < (srcVec[i].size() / sizeof(float)); ++idx)
            {
                reinterpret_cast<float *>(srcVec[i].data())[idx] = udistf(rng);
            }
            break;
        default:
            std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });
            break;
        }

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        printVec(srcVec[i], imgSrc[i].size().h, srcVecRowStride[i], format.numChannels(), "input");

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
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create gamma tensor
    std::vector<float> gammaVec;
    if (perChannel)
    {
        gammaVec.resize(batches * format.numChannels());
    }
    else
    {
        gammaVec.resize(batches);
    }
    std::ranges::generate(gammaVec, [&udistGamma, &rng]() { return udistGamma(rng); });

    auto         nElements = static_cast<int>(gammaVec.size());
    nvcv::Tensor gammaTensor({{nElements}, "N"}, nvcv::TYPE_F32);
    {
        auto dev = gammaTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), gammaVec.data(), gammaVec.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::GammaContrast gammacontrastOp(batches, format.numChannels());

    EXPECT_NO_THROW(gammacontrastOp(stream, batchSrc, batchDst, gammaTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * format.planePixelStrideBytes(0);
        int srcRowStride = dstWidth * format.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, dstHeight, cudaMemcpyDeviceToHost));

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride);
        std::ranges::generate(goldVec, []() { return 0; });

        // Generate gold result
        GammaContrastVarShapeCpuOpWrapper(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcRowStride,
                                          {srcWidth, srcHeight}, format, gammaVec, i, perChannel, nvcvDataType);

        printVec(goldVec, srcHeight, dstRowStride, format.numChannels(), "golden output");

        printVec(testVec, srcHeight, dstRowStride, format.numChannels(), "operator output");

        if (!isFloatTest)
        {
            EXPECT_EQ(testVec, goldVec);
        }
        else
        {
            VEC_EXPECT_NEAR(testVec, goldVec, 1E-6F, float);
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpGammaContrast_Negative, test::ValueList<int, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // batches, inFmt, outFmt
    {6, nvcv::FMT_U8, nvcv::FMT_U8}, // larger than max batches
    {2, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8}, // larger than max channels
    {2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8}, // input/output format mismatch (planar in, interleaved out)
    {2, nvcv::FMT_RGBf16, nvcv::FMT_RGBf16},
    {2, nvcv::FMT_U8, nvcv::FMT_S8},
});

// clang-format on

TEST_P(OpGammaContrast_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               batches = GetParamValue<0>();
    nvcv::ImageFormat inFmt   = GetParamValue<1>();
    nvcv::ImageFormat outFmt  = GetParamValue<2>();

    int width       = 24;
    int height      = 24;
    int maxBatches  = 5;
    int maxChannels = 3;

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));
    std::vector<nvcv::Image>      imgSrc;

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inFmt);
    }
    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), outFmt);
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor gammaTensor({{batches}, "N"}, nvcv::TYPE_F32); // not per channel

    // Run operator
    cvcuda::GammaContrast gammacontrastOp(maxBatches, maxChannels);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&gammacontrastOp, &stream, &batchSrc, &batchDst, &gammaTensor]
                                { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int batches     = 5;
    int       maxChannels = 4;

    int srcWidthBase  = 24;
    int srcHeightBase = 24;

    nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_RGBA8,  nvcv::FMT_RGB8},
        { nvcv::FMT_RGB8, nvcv::FMT_RGBA8}
    };

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        // Create input and output
        std::default_random_engine    randEng;
        std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
        std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));

        nvcv::Tensor gammaTensor({{batches}, "N"}, nvcv::TYPE_F32); // not per channel

        std::vector<nvcv::Image> imgSrc;

        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < batches - 1; ++i)
        {
            int tmpWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
            int tmpHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);

            imgSrc.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
            imgDst.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(batches);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(batches);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Run operator
        cvcuda::GammaContrast gammacontrastOp(batches, maxChannels);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&gammacontrastOp, &stream, &batchSrc, &batchDst, &gammaTensor]
                                    { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, gamma_length_must_match_batch_or_batch_channels)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int batches     = 3;
    const int channels    = 3;
    const int width       = 8;
    const int height      = 8;
    const int gammaLength = batches + 1;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
        imgDst.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor          gammaTensor({{gammaLength}, "N"}, nvcv::TYPE_F32);
    cvcuda::GammaContrast gammacontrastOp(batches, channels);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, input_output_data_types_and_channels_must_match)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int batches = 2;
    const int width   = 8;
    const int height  = 8;

    const std::array<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat, int, const char *>, 2> testCases{
        {{nvcv::FMT_U16, nvcv::FMT_F32, 1, "dtype"}, {nvcv::FMT_RGB8, nvcv::FMT_RGBA8, 4, "channels"}}
    };

    for (auto [inputFormat, outputFormat, maxChannels, name] : testCases)
    {
        SCOPED_TRACE(name);

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < batches; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{width, height}, inputFormat);
            imgDst.emplace_back(nvcv::Size2D{width, height}, outputFormat);
        }

        nvcv::ImageBatchVarShape batchSrc(batches);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(batches);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        nvcv::Tensor          gammaTensor({{batches}, "N"}, nvcv::TYPE_F32);
        cvcuda::GammaContrast gammacontrastOp(batches, maxChannels);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

namespace {

template<typename Submit>
void TestTensorInputOutputShapeMismatches(Submit &&submit)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int samples = 2;
    constexpr int width   = 8;
    constexpr int height  = 8;

    nvcv::Tensor input = nvcv::util::CreateTensor(samples, width, height, nvcv::FMT_RGB8);

    struct OutputShape
    {
        int         samples;
        int         width;
        int         height;
        const char *name;
    };

    const std::array<OutputShape, 3> outputShapes{
        {{samples - 1, width, height, "samples"},
         {samples, width - 1, height, "width"},
         {samples, width, height - 1, "height"}}
    };

    for (const auto &outputShape : outputShapes)
    {
        SCOPED_TRACE(outputShape.name);
        nvcv::Tensor output
            = nvcv::util::CreateTensor(outputShape.samples, outputShape.width, outputShape.height, nvcv::FMT_RGB8);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { submit(stream, input, output); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

TEST(OpGammaContrast_Negative, tensor_input_output_shapes_must_match)
{
    constexpr int         samples = 2;
    nvcv::Tensor          gammaTensor({{samples}, "N"}, nvcv::TYPE_F32);
    cvcuda::GammaContrast gammacontrastOp(samples, 3);

    TestTensorInputOutputShapeMismatches(
        [&gammacontrastOp, &gammaTensor](cudaStream_t stream, const nvcv::Tensor &input, const nvcv::Tensor &output)
        { gammacontrastOp(stream, input, output, gammaTensor); });
}

// The scalar (host-float) overload must reject sample/width/height mismatches too -- grid sizing is
// derived from the source access, so a smaller output would otherwise drive out-of-bounds writes.
TEST(OpGammaContrast_Negative, scalar_tensor_input_output_shapes_must_match)
{
    constexpr int         samples = 2;
    constexpr float       gamma   = 1.0f;
    constexpr float       gain    = 1.0f;
    cvcuda::GammaContrast gammacontrastOp(samples, 3);

    TestTensorInputOutputShapeMismatches(
        [&gammacontrastOp](cudaStream_t stream, const nvcv::Tensor &input, const nvcv::Tensor &output)
        { gammacontrastOp(stream, input, output, gamma, gain); });
}

TEST(OpGammaContrast_Negative, varshape_input_output_shapes_must_match)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int samples = 2;
    constexpr int width   = 8;
    constexpr int height  = 8;

    nvcv::ImageBatchVarShape input(samples);
    for (int i = 0; i < samples; ++i)
    {
        input.pushBack(nvcv::Image({width, height}, nvcv::FMT_RGB8));
    }

    nvcv::Tensor          gammaTensor({{samples}, "N"}, nvcv::TYPE_F32);
    cvcuda::GammaContrast gammacontrastOp(samples, 3);

    const std::array<std::pair<std::vector<nvcv::Size2D>, const char *>, 3> outputShapes{
        {{{{width, height}}, "samples"},
         {{{width, height}, {width - 1, height}}, "width"},
         {{{width, height}, {width, height - 1}}, "height"}}
    };

    for (const auto &[sizes, name] : outputShapes)
    {
        SCOPED_TRACE(name);
        nvcv::ImageBatchVarShape output(static_cast<int32_t>(sizes.size()));
        for (nvcv::Size2D size : sizes)
        {
            output.pushBack(nvcv::Image(size, nvcv::FMT_RGB8));
        }

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { gammacontrastOp(stream, input, output, gammaTensor); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaGammaContrastCreate(nullptr, 4, 4));
}

TEST(OpGammaContrast_Negative, create_rejects_non_positive_max_batch_or_channels)
{
    const std::vector<std::pair<int32_t, int32_t>> invalidLimits = {
        { 0,  4},
        {-1,  4},
        { 4,  0},
        { 4, -1}
    };

    for (auto [maxBatchSize, maxChannelCount] : invalidLimits)
    {
        NVCVOperatorHandle handle = nullptr;

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaGammaContrastCreate(&handle, maxBatchSize, maxChannelCount))
            << "maxBatchSize=" << maxBatchSize << ", maxChannelCount=" << maxChannelCount;
        EXPECT_EQ(nullptr, handle);

        if (handle != nullptr)
        {
            nvcvOperatorDestroy(handle);
        }
    }
}

#undef VEC_EXPECT_NEAR

// ---------------------------------------------------------------------------
// Planar (NCHW/CHW) layout support
//
// Gamma contrast applies x**gamma (or 255*(x/255)**gamma) per pixel per channel, independently of the
// other channels, so a planar input is processed plane-by-plane and the per-channel gamma maps to the
// same value the interleaved kernel reads. These tests run identical data + the same gamma tensor in
// both layouts through cvcuda::GammaContrast and require the (re-interleaved) planar output to match
// the interleaved output bit-for-bit. Scaffolding lives in PlanarParityUtils.hpp.
namespace {

float DeterministicGammaValue(size_t idx, float minValue, float maxValue, int salt)
{
    const int bucket = (static_cast<int>(idx) * 37 + salt) % 101;
    return minValue + (maxValue - minValue) * static_cast<float>(bucket) / 100.f;
}

uint8_t DeterministicByteValue(size_t idx, int sample)
{
    return static_cast<uint8_t>((static_cast<int>(idx) * 13 + sample * 29 + 17) % 256);
}

float DeterministicUnitValue(size_t idx, int sample)
{
    const int bucket = (static_cast<int>(idx) * 17 + sample * 31 + 11) % 100;
    return static_cast<float>(bucket) / 99.f;
}

struct TensorTestData
{
    NVCVDataType                      dataType;
    bool                              isFloat;
    int                               rowStride;
    size_t                            sampleBytes;
    std::vector<std::vector<uint8_t>> hostSamples;
};

void UploadDeterministicTensorInput(nvcv::Tensor &tensor, int samples, int width, int height, nvcv::ImageFormat format,
                                    TensorTestData &testData)
{
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(tensorData, nullptr);
    auto tensorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*tensorData);
    ASSERT_TRUE(tensorAccess);

    ASSERT_EQ(NVCV_SUCCESS,
              nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(format), 0, &testData.dataType));
    testData.isFloat = testData.dataType == NVCV_DATA_TYPE_F32 || testData.dataType == NVCV_DATA_TYPE_2F32
                    || testData.dataType == NVCV_DATA_TYPE_3F32 || testData.dataType == NVCV_DATA_TYPE_4F32;
    const int bytesPerElement = format.planePixelStrideBytes(0) / format.numChannels();
    testData.rowStride        = width * format.numChannels() * bytesPerElement;
    testData.sampleBytes      = static_cast<size_t>(testData.rowStride) * height;
    testData.hostSamples.assign(samples, std::vector<uint8_t>(testData.sampleBytes));

    for (int sample = 0; sample < samples; ++sample)
    {
        auto &hostSample = testData.hostSamples[sample];
        if (testData.isFloat)
        {
            auto *values = reinterpret_cast<float *>(hostSample.data());
            for (size_t i = 0; i < testData.sampleBytes / sizeof(float); ++i)
            {
                values[i] = DeterministicUnitValue(i, sample);
            }
        }
        else
        {
            for (size_t i = 0; i < hostSample.size(); ++i)
            {
                hostSample[i] = DeterministicByteValue(i, sample);
            }
        }

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(tensorAccess->sampleData(sample), tensorAccess->rowStride(), hostSample.data(),
                               testData.rowStride, testData.rowStride, height, cudaMemcpyHostToDevice));
    }
}

void DownloadTensorSample(const nvcv::Tensor &tensor, int sample, int rowStride, int height,
                          std::vector<uint8_t> &hostSample)
{
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(tensorData, nullptr);
    auto tensorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*tensorData);
    ASSERT_TRUE(tensorAccess);

    hostSample.resize(static_cast<size_t>(rowStride) * height);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(hostSample.data(), rowStride, tensorAccess->sampleData(sample),
                                        tensorAccess->rowStride(), rowStride, height, cudaMemcpyDeviceToHost));
}

void UploadGammaTensor(nvcv::Tensor &tensor, const std::vector<float> &values)
{
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(tensorData, nullptr);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(tensorData->basePtr(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void ExpectTensorNear(const std::vector<uint8_t> &got, const std::vector<uint8_t> &gold, bool isFloat)
{
    ASSERT_EQ(got.size(), gold.size());
    if (isFloat)
    {
        const auto *actual   = reinterpret_cast<const float *>(got.data());
        const auto *expected = reinterpret_cast<const float *>(gold.data());
        // Host std::pow and device __powf have different precision; this bounds the expected approximation error.
        for (size_t i = 0; i < got.size() / sizeof(float); ++i)
        {
            EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "at " << i;
        }
    }
    else
    {
        // Integer results can differ by one when host and device power approximations reach the cast boundary.
        for (size_t i = 0; i < got.size(); ++i)
        {
            EXPECT_NEAR(static_cast<int>(got[i]), static_cast<int>(gold[i]), 1) << "at " << i;
        }
    }
}

void RunGammaContrastPlanarParityCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                      int numImages, bool perChannel)
{
    const int channels  = interleavedFmt.numChannels();
    const int nElements = perChannel ? numImages * channels : numImages;

    // Build the gamma tensor once (same instance feeds both layouts).
    std::vector<float> gammaVec(nElements);
    for (int i = 0; i < nElements; ++i)
    {
        gammaVec[i] = DeterministicGammaValue(i, 0.5f, 2.0f, 23);
    }

    nvcv::Tensor gammaTensor({{nElements}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, gammaVec));

    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, w, h, w, h, numImages,
        [numImages, channels, &gammaTensor](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                            const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::GammaContrast op(numImages, channels);
            EXPECT_NO_THROW(op(s, src, dst, gammaTensor));
        });
}

} // namespace

// Parameters: width, height, numImages, perChannelGamma, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastPlanar,
                  test::ValueList<int, int, int, bool, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2, false,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, per-image gamma
    {123,  66, 2,  true,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, per-channel gamma
    { 64,  48, 1,  true,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8, per-channel gamma
    { 50,  40, 2, false,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32}, // RGBf32, per-image gamma
    { 72,  54, 2,  true, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}, // RGBAf32, per-channel gamma
});

// clang-format on

TEST_P(OpGammaContrastPlanar, varshape_matches_interleaved)
{
    RunGammaContrastPlanarParityCase(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(), GetParamValue<1>(),
                                     GetParamValue<2>(), GetParamValue<3>());
}

// ---------------------------------------------------------------------------
// Tensor variant (interleaved + planar)
namespace {

// Tensor planar (NCHW/CHW) parity: planar tensor output must match interleaved tensor output bit-for-
// bit for the same data + gamma. Validates the planar tensor kernel against the interleaved one.
void RunGammaContrastTensorPlanarParityCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                            int numSamples, bool perChannel)
{
    const int channels  = interleavedFmt.numChannels();
    const int nElements = perChannel ? numSamples * channels : numSamples;

    std::vector<float> gammaVec(nElements);
    for (int i = 0; i < nElements; ++i)
    {
        gammaVec[i] = DeterministicGammaValue(i, 0.5f, 2.0f, 47);
    }

    nvcv::Tensor gammaTensor({{nElements}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, gammaVec));

    test::planar::RunTensorParity(planarFmt, interleavedFmt, w, h, w, h, numSamples,
                                  [numSamples, channels, &gammaTensor](cudaStream_t s, const nvcv::Tensor &src,
                                                                       const nvcv::Tensor &dst, nvcv::ImageFormat)
                                  {
                                      cvcuda::GammaContrast op(numSamples, channels);
                                      EXPECT_NO_THROW(op(s, src, dst, gammaTensor));
                                  });
}

} // namespace

// width, height, numSamples, perChannelGamma, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastTensorPlanar,
                  test::ValueList<int, int, int, bool, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2, false,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {123,  66, 2,  true,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 64,  48, 1,  true,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 50,  40, 3, false,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    { 72,  54, 2,  true, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

// clang-format on

TEST_P(OpGammaContrastTensorPlanar, tensor_matches_interleaved)
{
    RunGammaContrastTensorPlanarParityCase(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(),
                                           GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>());
}

// Tensor interleaved (NHWC) correctness against an independent CPU gold (the same gold the var-shape
// path is validated with), exercising the tensor kernel directly.
// width, height, numSamples, format, perChannelGamma
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastTensor,
                  test::ValueList<int, int, int, NVCVImageFormat, bool>{
    {  9, 11, 2,    NVCV_IMAGE_FORMAT_U8,  true},
    { 12,  7, 3,  NVCV_IMAGE_FORMAT_RGB8,  true},
    { 11, 11, 2, NVCV_IMAGE_FORMAT_RGBA8, false},
    {  7,  9, 2, NVCV_IMAGE_FORMAT_RGBA8,  true},
    {  8,  6, 2, NVCV_IMAGE_FORMAT_RGBf32, true},
    {  9,  7, 2, NVCV_IMAGE_FORMAT_RGBAf32, true},
});

// clang-format on

TEST_P(OpGammaContrastTensor, correct_output)
{
    const int               width   = GetParamValue<0>();
    const int               height  = GetParamValue<1>();
    const int               samples = GetParamValue<2>();
    const nvcv::ImageFormat fmt{GetParamValue<3>()};
    const bool              perCh    = GetParamValue<4>();
    const int               channels = fmt.numChannels();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src = nvcv::util::CreateTensor(samples, width, height, fmt);
    nvcv::Tensor dst = nvcv::util::CreateTensor(samples, width, height, fmt);

    TensorTestData testData;
    ASSERT_NO_FATAL_FAILURE(UploadDeterministicTensorInput(src, samples, width, height, fmt, testData));

    // Gamma tensor.
    const int          nElements = perCh ? samples * channels : samples;
    std::vector<float> gammaVec(nElements);
    for (int i = 0; i < nElements; ++i)
    {
        gammaVec[i] = DeterministicGammaValue(i, 0.4f, 2.0f, 61);
    }
    nvcv::Tensor gammaTensor({{nElements}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, gammaVec));

    cvcuda::GammaContrast op(samples, channels);
    EXPECT_NO_THROW(op(stream, src, dst, gammaTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int n = 0; n < samples; ++n)
    {
        std::vector<uint8_t> got;
        std::vector<uint8_t> gold = testData.hostSamples[n];
        ASSERT_NO_FATAL_FAILURE(DownloadTensorSample(dst, n, testData.rowStride, height, got));
        GammaContrastVarShapeCpuOpWrapper(gold, testData.rowStride, {width, height}, testData.hostSamples[n],
                                          testData.rowStride, {width, height}, fmt, gammaVec, n, perCh,
                                          testData.dataType);
        ExpectTensorNear(got, gold, testData.isFloat);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scalar (host-float) gamma/gain tensor path
//
// The scalar overload applies out = gain * in**gamma with a single gamma/gain baked into the kernel
// launch (no gamma tensor, no host->device copy). Three properties are checked below:
//   * planar (NCHW/CHW) output matches interleaved output bit-for-bit (layout parity);
//   * with gain != 1 it matches an independent CPU gold that includes the gain term;
//   * with gain == 1 (an exact IEEE no-op) it is bit-exact with the device-tensor gamma path fed a
//     gamma tensor filled with the same value -- i.e. the already-gold-validated kernel, across every
//     supported dtype.
namespace {

// Independent CPU gold for the scalar path (single gamma/gain for all samples/channels).
void GammaContrastScalarCpuGold(std::vector<uint8_t> &hDst, const std::vector<uint8_t> &hSrc, float gamma, float gain,
                                bool isFloat, NVCVRoundMode roundMode)
{
    if (isFloat)
    {
        const auto  *src = reinterpret_cast<const float *>(hSrc.data());
        auto        *dst = reinterpret_cast<float *>(hDst.data());
        const size_t n   = hSrc.size() / sizeof(float);
        for (size_t i = 0; i < n; ++i)
        {
            dst[i] = nvcv::cuda::clamp(gain * std::pow(src[i], gamma), 0.f, 1.f);
        }
    }
    else
    {
        for (size_t i = 0; i < hSrc.size(); ++i)
        {
            const float tmp = (hSrc[i] + 0.0f) / 255.0f;
            float       out = gain * std::pow(tmp, gamma) * 255.0f;
            out             = nvcv::cuda::clamp(out, 0.0f, 255.0f);
            hDst[i]         = static_cast<uint8_t>(roundMode == NVCV_ROUND_TRUNCATE ? std::trunc(out) : std::rint(out));
        }
    }
}

void RunGammaContrastTensorScalarPlanarParityCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w,
                                                  int h, int numSamples, float gamma, float gain,
                                                  NVCVRoundMode roundMode)
{
    const int channels = interleavedFmt.numChannels();
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, w, h, w, h, numSamples,
        [numSamples, channels, gamma, gain, roundMode](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                                       nvcv::ImageFormat)
        {
            cvcuda::GammaContrast op(numSamples, channels);
            EXPECT_NO_THROW(op(s, src, dst, gamma, gain, roundMode));
        });
}

} // namespace

// width, height, numSamples, gamma, gain, planarFmt, interleavedFmt, roundMode
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastTensorScalarPlanar,
                  test::ValueList<int, int, int, float, float, nvcv::ImageFormat, nvcv::ImageFormat, NVCVRoundMode>{
    {176, 113, 2, 0.75f, 1.0f,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8,  NVCV_ROUND_NEAREST},
    {123,  66, 2, 1.50f, 0.9f,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8, NVCV_ROUND_TRUNCATE},
    { 64,  48, 1, 0.50f, 0.8f,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8,  NVCV_ROUND_NEAREST},
    { 50,  40, 3, 2.00f, 1.0f,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, NVCV_ROUND_TRUNCATE},
    { 72,  54, 2, 0.70f, 1.2f, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32,  NVCV_ROUND_NEAREST},
});

// clang-format on

TEST_P(OpGammaContrastTensorScalarPlanar, planar_matches_interleaved)
{
    RunGammaContrastTensorScalarPlanarParityCase(GetParamValue<5>(), GetParamValue<6>(), GetParamValue<0>(),
                                                 GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                                                 GetParamValue<4>(), GetParamValue<7>());
}

// Scalar path against an independent CPU gold that includes the gain term (interleaved).
// width, height, numSamples, format, gamma, gain, roundMode
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastTensorScalar,
                  test::ValueList<int, int, int, NVCVImageFormat, float, float, NVCVRoundMode>{
    {  9, 11, 2,      NVCV_IMAGE_FORMAT_U8, 0.50f, 0.8f,  NVCV_ROUND_NEAREST},
    { 12,  7, 3,    NVCV_IMAGE_FORMAT_RGB8, 1.50f, 0.9f, NVCV_ROUND_TRUNCATE},
    { 11, 11, 2,   NVCV_IMAGE_FORMAT_RGBA8, 0.80f, 1.0f,  NVCV_ROUND_NEAREST},
    {  8,  6, 2,  NVCV_IMAGE_FORMAT_RGBf32, 2.00f, 1.0f, NVCV_ROUND_TRUNCATE},
    { 10,  5, 2, NVCV_IMAGE_FORMAT_RGBAf32, 0.70f, 1.2f,  NVCV_ROUND_NEAREST},
});

// clang-format on

TEST_P(OpGammaContrastTensorScalar, matches_cpu_gold)
{
    const int               width   = GetParamValue<0>();
    const int               height  = GetParamValue<1>();
    const int               samples = GetParamValue<2>();
    const nvcv::ImageFormat fmt{GetParamValue<3>()};
    const float             gamma     = GetParamValue<4>();
    const float             gain      = GetParamValue<5>();
    const NVCVRoundMode     roundMode = GetParamValue<6>();
    const int               channels  = fmt.numChannels();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src = nvcv::util::CreateTensor(samples, width, height, fmt);
    nvcv::Tensor dst = nvcv::util::CreateTensor(samples, width, height, fmt);

    TensorTestData testData;
    ASSERT_NO_FATAL_FAILURE(UploadDeterministicTensorInput(src, samples, width, height, fmt, testData));

    cvcuda::GammaContrast op(samples, channels);
    EXPECT_NO_THROW(op(stream, src, dst, gamma, gain, roundMode));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int n = 0; n < samples; ++n)
    {
        std::vector<uint8_t> got;
        std::vector<uint8_t> gold(testData.sampleBytes);
        ASSERT_NO_FATAL_FAILURE(DownloadTensorSample(dst, n, testData.rowStride, height, got));
        GammaContrastScalarCpuGold(gold, testData.hostSamples[n], gamma, gain, testData.isFloat, roundMode);
        ExpectTensorNear(got, gold, testData.isFloat);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrastTensorScalar, round_mode_controls_integer_conversion)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src      = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_U8);
    nvcv::Tensor nearest  = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_U8);
    nvcv::Tensor truncate = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_U8);

    auto srcData      = src.exportData<nvcv::TensorDataStridedCuda>();
    auto nearestData  = nearest.exportData<nvcv::TensorDataStridedCuda>();
    auto truncateData = truncate.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && nearestData && truncateData);

    constexpr uint8_t input = 255;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), &input, sizeof(input), cudaMemcpyHostToDevice));

    cvcuda::GammaContrast op(1, 1);
    constexpr float       gain = 1.5f / 255.0f;
    EXPECT_NO_THROW(op(stream, src, nearest, 1.0f, gain, NVCV_ROUND_NEAREST));
    EXPECT_NO_THROW(op(stream, src, truncate, 1.0f, gain, NVCV_ROUND_TRUNCATE));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    uint8_t nearestValue  = 0;
    uint8_t truncateValue = 0;
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(&nearestValue, nearestData->basePtr(), sizeof(nearestValue), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(&truncateValue, truncateData->basePtr(), sizeof(truncateValue), cudaMemcpyDeviceToHost));
    EXPECT_EQ(nearestValue, 2);
    EXPECT_EQ(truncateValue, 1);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, src, nearest, 1.0f, gain, static_cast<NVCVRoundMode>(-1)); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Scalar path with gain == 1 must be bit-exact with the device-tensor gamma path fed a gamma tensor
// filled with the same value (that path is itself validated against the CPU gold above). Exercises the
// full supported dtype set through the scalar kernels.
// width, height, numSamples, format, gamma
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastScalarVsTensor,
                  test::ValueList<int, int, int, NVCVImageFormat, float>{
    {  9, 11, 2,      NVCV_IMAGE_FORMAT_U8, 0.60f},
    { 12,  7, 3,    NVCV_IMAGE_FORMAT_RGB8, 1.40f},
    { 11, 11, 2,   NVCV_IMAGE_FORMAT_RGBA8, 0.80f},
    { 10,  9, 2,      NVCV_IMAGE_FORMAT_U16, 1.20f},
    {  7,  8, 2,      NVCV_IMAGE_FORMAT_S16, 0.90f},
    {  8,  6, 2,      NVCV_IMAGE_FORMAT_S32, 1.10f},
    {  8,  6, 2,  NVCV_IMAGE_FORMAT_RGBf32, 0.70f},
    { 10,  5, 2, NVCV_IMAGE_FORMAT_RGBAf32, 1.30f},
});

// clang-format on

TEST_P(OpGammaContrastScalarVsTensor, bit_exact_gain1)
{
    const int               width   = GetParamValue<0>();
    const int               height  = GetParamValue<1>();
    const int               samples = GetParamValue<2>();
    const nvcv::ImageFormat fmt{GetParamValue<3>()};
    const float             gamma    = GetParamValue<4>();
    const int               channels = fmt.numChannels();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src       = nvcv::util::CreateTensor(samples, width, height, fmt);
    nvcv::Tensor dstScalar = nvcv::util::CreateTensor(samples, width, height, fmt);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(samples, width, height, fmt);

    TensorTestData testData;
    ASSERT_NO_FATAL_FAILURE(UploadDeterministicTensorInput(src, samples, width, height, fmt, testData));

    // Per-image gamma tensor filled with the same value -> every sample/channel uses `gamma`, matching
    // the scalar broadcast; gain == 1 is an exact IEEE no-op, so the two outputs must be identical.
    std::vector<float> gammaVec(samples, gamma);
    nvcv::Tensor       gammaTensor({{samples}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, gammaVec));

    cvcuda::GammaContrast op(samples, channels);
    EXPECT_NO_THROW(op(stream, src, dstScalar, gamma, 1.0f));
    EXPECT_NO_THROW(op(stream, src, dstTensor, gammaTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto compareSample = [&dstScalar, &dstTensor, &testData, height](int n)
    {
        std::vector<uint8_t> scalarOut;
        std::vector<uint8_t> tensorOut;
        ASSERT_NO_FATAL_FAILURE(DownloadTensorSample(dstScalar, n, testData.rowStride, height, scalarOut));
        ASSERT_NO_FATAL_FAILURE(DownloadTensorSample(dstTensor, n, testData.rowStride, height, tensorOut));
        EXPECT_EQ(scalarOut, tensorOut) << "scalar (gain=1) != device-tensor gamma path at sample " << n;
    };

    for (int n = 0; n < samples; ++n)
    {
        ASSERT_NO_FATAL_FAILURE(compareSample(n));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
