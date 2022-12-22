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
#include <cvcuda/OpLaplacian.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

static const float kLaplacianKernel1[] = {0.0f, 1.0f, 0.0f, 1.0f, -4.0f, 1.0f, 0.0f, 1.0f, 0.0f};
static const float kLaplacianKernel3[] = {2.0f, 0.0f, 2.0f, 0.0f, -8.0f, 0.0f, 2.0f, 0.0f, 2.0f};

// clang-format off

NVCV_TEST_SUITE_P(OpLaplacian, test::ValueList<int, int, int, NVCVImageFormat, int, float, NVCVBorderType>
{
    // width, height, batches,                    format, ksize, scale,           borderMode
    {    176,    113,       1,      NVCV_IMAGE_FORMAT_U8,     1,  1.0f, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,     NVCV_IMAGE_FORMAT_U16,     3,  1.0f, NVCV_BORDER_CONSTANT},
    {     77,     55,       3,    NVCV_IMAGE_FORMAT_RGB8,     1,  2.0f, NVCV_BORDER_CONSTANT},
    {     62,    111,       4,   NVCV_IMAGE_FORMAT_RGBA8,     3,  3.0f, NVCV_BORDER_WRAP},
    {      4,      3,       3, NVCV_IMAGE_FORMAT_RGBAf32,     1,  1.0f, NVCV_BORDER_REPLICATE},
    {      3,      3,       4,  NVCV_IMAGE_FORMAT_RGBf32,     3,  1.0f, NVCV_BORDER_REFLECT},
    {      4,      3,       4, NVCV_IMAGE_FORMAT_RGBAf32,     1,  1.0f, NVCV_BORDER_REFLECT101}
});

// clang-format on

TEST_P(OpLaplacian, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int   ksize = GetParamValue<4>();
    float scale = GetParamValue<5>();

    NVCVBorderType borderMode = GetParamValue<6>();

    float4 borderValue = cuda::SetAll<float4>(0);

    int3 shape{width, height, batches};

    nvcv::Tensor inTensor(batches, {width, height}, format);
    nvcv::Tensor outTensor(batches, {width, height}, format);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inTensor.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(outTensor.exportData());

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    long3 inStrides{inData->stride(0), inData->stride(1), inData->stride(2)};
    long3 outStrides{outData->stride(0), outData->stride(1), outData->stride(2)};

    long inBufSize  = inStrides.x * inData->shape(0);
    long outBufSize = outStrides.x * outData->shape(0);

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Laplacian laplacianOp;

    EXPECT_NO_THROW(laplacianOp(stream, inTensor, outTensor, ksize, scale, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    std::vector<float> kernel(9);

    nvcv::Size2D kernelSize{3, 3};
    int2         kernelAnchor{kernelSize.w / 2, kernelSize.h / 2};

    for (int i = 0; i < 9; ++i)
    {
        if (ksize == 1)
        {
            kernel[i] = kLaplacianKernel1[i] * scale;
        }
        else if (ksize == 3)
        {
            kernel[i] = kLaplacianKernel3[i] * scale;
        }
    }

    test::Convolve(goldVec, outStrides, inVec, inStrides, shape, format, kernel, kernelSize, kernelAnchor, borderMode,
                   borderValue);

    EXPECT_EQ(testVec, goldVec);
}

TEST_P(OpLaplacian, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int   ksize = GetParamValue<4>();
    float scale = GetParamValue<5>();

    NVCVBorderType borderMode = GetParamValue<6>();

    float4 borderValue = cuda::SetAll<float4>(0);

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

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

    // Create kernel aperture size tensor
    nvcv::Tensor ksizeTensor({{batches}, "N"}, nvcv::TYPE_S32);
    {
        auto *dev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(ksizeTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int> vec(batches, ksize);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create scale tensor
    nvcv::Tensor scaleTensor({{batches}, "N"}, nvcv::TYPE_F32);
    {
        auto *dev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(scaleTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<float> vec(batches, scale);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::Laplacian laplacianOp;

    EXPECT_NO_THROW(laplacianOp(stream, batchSrc, batchDst, ksizeTensor, scaleTensor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDst[i]->exportData());
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowStride, dstRowStride, format.planePixelStrideBytes(0)};

        std::vector<uint8_t> testVec(shape.y * pitches.y);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // Generate gold result
        std::vector<float> kernel(9);
        nvcv::Size2D       kernelSize{3, 3};
        int2               kernelAnchor{kernelSize.w / 2, kernelSize.h / 2};

        for (int i = 0; i < 9; ++i)
        {
            if (ksize == 1)
            {
                kernel[i] = kLaplacianKernel1[i] * scale;
            }
            else if (ksize == 3)
            {
                kernel[i] = kLaplacianKernel3[i] * scale;
            }
        }

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, format, kernel, kernelSize, kernelAnchor,
                       borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}
