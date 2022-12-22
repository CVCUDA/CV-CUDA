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
#include <cvcuda/OpConv2D.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

#include <random>

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;

// clang-format off

NVCV_TEST_SUITE_P(OpConv2D, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType>
{
    // width, height, numImages, kernelWidth, kernelHeight, kernelAnchorX, kernelAnchorY,           borderMode
    {     32,     33,         1,           3,            3,            -1,            -1, NVCV_BORDER_CONSTANT},
    {    123,    144,         2,           5,            5,            -1,            -1, NVCV_BORDER_CONSTANT},
    {     66,     99,         3,           7,            7,             5,             5, NVCV_BORDER_CONSTANT},
    {     13,     12,        13,           5,            5,             4,             4, NVCV_BORDER_WRAP},
    {      4,      3,         4,           3,            3,             1,             1, NVCV_BORDER_REPLICATE},
    {     44,     55,         5,           3,            3,            -1,            -1, NVCV_BORDER_REFLECT},
    {    244,    155,         6,           5,            5,            -1,            -1, NVCV_BORDER_REFLECT101}
});

// clang-format on

TEST_P(OpConv2D, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width         = GetParamValue<0>();
    int height        = GetParamValue<1>();
    int numImages     = GetParamValue<2>();
    int kernelWidth   = GetParamValue<3>();
    int kernelHeight  = GetParamValue<4>();
    int kernelAnchorX = GetParamValue<5>();
    int kernelAnchorY = GetParamValue<6>();

    NVCVBorderType borderMode = GetParamValue<7>();

    nvcv::ImageFormat imageFormat  = nvcv::FMT_RGBA8;
    nvcv::ImageFormat kernelFormat = nvcv::FMT_F32;

    nvcv::Size2D kernelSize{kernelWidth, kernelHeight};
    int2         kernelAnchor{kernelAnchorX, kernelAnchorY};

    float4 borderValue = cuda::SetAll<float4>(0);

    // Create input varshape

    std::default_random_engine rng;

    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  srcVecRowStride(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(
            std::make_unique<nvcv::Image>(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, imageFormat));

        int srcRowStride   = imgSrc[i]->size().w * imageFormat.numChannels();
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i]->size().h * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto *imgData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        assert(imgData != nullptr);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i]->size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape

    std::vector<std::unique_ptr<nvcv::Image>> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgDst.emplace_back(std::make_unique<nvcv::Image>(imgSrc[i]->size(), imgSrc[i]->format()));
    }
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel varshape

    std::vector<std::unique_ptr<nvcv::Image>> kernel;
    std::vector<std::vector<float>>           kernelVec(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        kernel.emplace_back(std::make_unique<nvcv::Image>(kernelSize, kernelFormat));

        int rowStride = kernel[i]->size().w * sizeof(float);

        std::uniform_real_distribution<float> udist(0.f, 1.f);

        kernelVec[i].resize(kernel[i]->size().h * kernel[i]->size().w);

        std::generate(kernelVec[i].begin(), kernelVec[i].end(), [&]() { return udist(rng); });

        auto *data = dynamic_cast<const nvcv::IImageDataStridedCuda *>(kernel[i]->exportData());
        assert(data != nullptr);

        // Copy kernel data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(data->plane(0).basePtr, data->plane(0).rowStride, kernelVec[i].data(), rowStride,
                                    rowStride, kernel[i]->size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchKernel(numImages);
    batchKernel.pushBack(kernel.begin(), kernel.end());

    // Create kernel anchor tensor

    nvcv::Tensor kernelAnchorTensor({{numImages}, "N"}, nvcv::TYPE_2S32);

    {
        auto *dev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(kernelAnchorTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(numImages, kernelAnchor);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Generate test result

    cvcuda::Conv2D conv2dOp;
    EXPECT_NO_THROW(conv2dOp(stream, batchSrc, batchDst, batchKernel, kernelAnchorTensor, borderMode));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDst[i]->exportData());
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowStride, dstRowStride, 4};

        std::vector<uint8_t> testVec(shape.y * pitches.y);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        // Generate gold result
        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, imageFormat, kernelVec[i], kernelSize, kernelAnchor,
                       borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}
