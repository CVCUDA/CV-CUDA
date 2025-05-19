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

#include "ConvUtils.hpp"
#include "Definitions.hpp"

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

#define DBG_GAMMA_CONTRAST 0

static void printVec(std::vector<uint8_t> &vec, int height, int rowPitch, int bytesPerPixel, std::string name)
{
#if DBG_GAMMA_CONTRAST
    for (int i = 0; i < bytesPerPixel; i++)
    {
        std::cout << "\nPrint " << name << " for channel: " << i << std::endl;

        for (int k = 0; k < height; k++)
        {
            for (int j = 0; j < static_cast<int>(rowPitch / bytesPerPixel); j++)
            {
                printf("%4d, ", static_cast<int>(vec[k * rowPitch + j * bytesPerPixel + i]));
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
#endif
}

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
                                const std::vector<T> &hSrc, int srcRowStride, nvcv::Size2D srcSize,
                                nvcv::ImageFormat fmt, const std::vector<float> gamma, const int imageIndex,
                                bool perChannel)
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
                T     out       = std::rint(pow(tmp, gamma_tmp) * 255.0f);
                dstPtr[index]   = out;
            }
        }
    }
}

// float cpu op
template<>
void GammaContrastVarShapeCpuOp(std::vector<float> &hDst, int dstRowStride, nvcv::Size2D dstSize,
                                const std::vector<float> &hSrc, int srcRowStride, nvcv::Size2D srcSize,
                                nvcv::ImageFormat fmt, const std::vector<float> gamma, const int imageIndex,
                                bool perChannel)
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
                                       nvcv::ImageFormat fmt, const std::vector<float> gamma, const int imageIndex,
                                       bool perChannel, NVCVDataType nvcvDataType)
{
    if (nvcvDataType == NVCV_DATA_TYPE_F32 || nvcvDataType == NVCV_DATA_TYPE_2F32 || nvcvDataType == NVCV_DATA_TYPE_3F32
        || nvcvDataType == NVCV_DATA_TYPE_4F32)
    {
        std::vector<float> src_tmp(hSrc.size() / sizeof(float));
        std::vector<float> dst_tmp(hDst.size() / sizeof(float));
        size_t             copySize = hSrc.size();
        memcpy(static_cast<void *>(src_tmp.data()), const_cast<void *>(static_cast<const void *>(hSrc.data())),
               copySize);
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
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(format, 0, &nvcvDataType));
    float gamma       = GetParamValue<4>();
    bool  isFloatTest = false;

    bool perChannel = GetParamValue<5>();

    // Create input varshape
    std::default_random_engine            rng;
    std::uniform_int_distribution<int>    udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int>    udistHeight(height * 0.8, height * 1.1);
    std::uniform_real_distribution<float> udistGamma(gamma * 0.8, 1.0);

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
            std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });
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
    std::generate(gammaVec.begin(), gammaVec.end(), [&]() { return udistGamma(rng); });

    int          nElements = gammaVec.size();
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
        std::generate(goldVec.begin(), goldVec.end(), [&]() { return 0; });

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

NVCV_TEST_SUITE_P(OpGammaContrastVarshape_negative, test::ValueList<int, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // batches, inFmt, outFmt
    {6, nvcv::FMT_U8, nvcv::FMT_U8}, // larger than max batches
    {2, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8}, // larger than max channels
    {2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8}, // different format
    {2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8p},
    {2, nvcv::FMT_RGBf16, nvcv::FMT_RGBf16},
    {2, nvcv::FMT_U8, nvcv::FMT_S8},
});

// clang-format on

TEST_P(OpGammaContrastVarshape_negative, op)
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
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);
    std::vector<nvcv::Image>           imgSrc;

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
              nvcv::ProtectCall([&] { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrastVarshape_negative, varshape_hasDifferentFormat)
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

    for (auto testCase : testSet)
    {
        nvcv::ImageFormat inputFmtExtra  = std::get<0>(testCase);
        nvcv::ImageFormat outputFmtExtra = std::get<1>(testCase);

        // Create input and output
        std::default_random_engine         randEng;
        std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.1);
        std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.1);

        nvcv::Tensor gammaTensor({{batches}, "N"}, nvcv::TYPE_F32); // not per channel

        std::vector<nvcv::Image> imgSrc, imgDst;

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
                  nvcv::ProtectCall([&] { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrastVarshape_negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaGammaContrastCreate(nullptr, 4, 4));
}

#undef VEC_EXPECT_NEAR
