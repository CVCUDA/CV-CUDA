/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

// clang-format off

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

static void GammaContrastVarShapeCpuOp(std::vector<uint8_t> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<uint8_t> &hSrc,
                   int srcRowStride, nvcv::Size2D srcSize, nvcv::ImageFormat fmt, const std::vector<float> gamma, const int imageIndex, bool perChannel)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    uint8_t       *dstPtr = hDst.data();
    const uint8_t *srcPtr = hSrc.data();

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            for (int k = 0; k < elementsPerPixel; k++)
            {
                int index = dst_y * dstRowStride + dst_x * elementsPerPixel + k;
                float gamma_tmp = perChannel ? gamma[imageIndex * elementsPerPixel + k] : gamma[imageIndex];
                float tmp   = (srcPtr[index] + 0.0f) / 255.0f;
                uint8_t out  = std::rint(pow(tmp, gamma_tmp) * 255.0f);
                dstPtr[index] = out;
            }
        }
    }
}

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

    float gamma = GetParamValue<4>();

    bool perChannel = GetParamValue<5>();

    // Create input varshape
    std::default_random_engine            rng;
    std::uniform_int_distribution<int>    udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int>    udistHeight(height * 0.8, height * 1.1);
    std::uniform_real_distribution<float> udistGamma(gamma * 0.8, 1.0);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format));

        int srcRowStride   = imgSrc[i]->size().w * format.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i]->size().h * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto *imgData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        ASSERT_NE(imgData, nullptr);

        printVec(srcVec[i], imgSrc[i]->size().h, srcVecRowStride[i], format.numChannels(), "input");

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i]->size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<std::unique_ptr<nvcv::Image>> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(std::make_unique<nvcv::Image>(imgSrc[i]->size(), imgSrc[i]->format()));
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
        auto *dev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(gammaTensor.exportData());
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

        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        ASSERT_EQ(srcData->numPlanes(), 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDst[i]->exportData());
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
        GammaContrastVarShapeCpuOp(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcRowStride,
                                   {srcWidth, srcHeight}, format, gammaVec, i, perChannel);

        printVec(goldVec, srcHeight, dstRowStride, format.numChannels(), "golden output");

        printVec(testVec, srcHeight, dstRowStride, format.numChannels(), "operator output");

        EXPECT_EQ(testVec, goldVec);
    }
}
