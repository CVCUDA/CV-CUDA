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

#include "Definitions.hpp"

#include <cvcuda/OpErase.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <iostream>

TEST(OpErase, OpErase_Tensor)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn(1, {640, 480}, nvcv::FMT_U8);
    nvcv::Tensor imgOut(1, {640, 480}, nvcv::FMT_U8);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*imgIn.exportData());
    ASSERT_TRUE(inAccess);

    ASSERT_EQ(1, inAccess->numSamples());

    // setup the buffer
    EXPECT_EQ(cudaSuccess, cudaMemset2D(inAccess->planeData(0), inAccess->rowStride(), 0,
                                        inAccess->numCols() * inAccess->colStride(), inAccess->numRows()));

    const auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgOut.exportData());
    ASSERT_NE(nullptr, outData);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int64_t outBufferSize = outAccess->sampleStride() * outAccess->numSamples();

    // Set output buffer to dummy value
    EXPECT_EQ(cudaSuccess, cudaMemset(outAccess->sampleData(0), 0xFA, outBufferSize));

    //parameters
    int          num_erasing_area = 2;
    nvcv::Tensor anchor({{num_erasing_area}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor erasing({{num_erasing_area}, "N"}, nvcv::TYPE_3S32);
    nvcv::Tensor values({{num_erasing_area}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor imgIdx({{num_erasing_area}, "N"}, nvcv::TYPE_S32);

    const auto *anchorData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(anchor.exportData());
    const auto *erasingData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(erasing.exportData());
    const auto *valuesData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(values.exportData());
    const auto *imgIdxData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgIdx.exportData());

    ASSERT_NE(nullptr, anchorData);
    ASSERT_NE(nullptr, erasingData);
    ASSERT_NE(nullptr, valuesData);
    ASSERT_NE(nullptr, imgIdxData);

    std::vector<int2>  anchorVec(num_erasing_area);
    std::vector<int3>  erasingVec(num_erasing_area);
    std::vector<int>   imgIdxVec(num_erasing_area);
    std::vector<float> valuesVec(num_erasing_area);

    anchorVec[0].x  = 0;
    anchorVec[0].y  = 0;
    erasingVec[0].x = 10;
    erasingVec[0].y = 10;
    erasingVec[0].z = 0x1;
    imgIdxVec[0]    = 0;
    valuesVec[0]    = 1.f;

    anchorVec[1].x  = 10;
    anchorVec[1].y  = 10;
    erasingVec[1].x = 20;
    erasingVec[1].y = 20;
    erasingVec[1].z = 0x1;
    imgIdxVec[1]    = 0;
    valuesVec[1]    = 1.f;

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchorData->basePtr(), anchorVec.data(), anchorVec.size() * sizeof(int2),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingData->basePtr(), erasingVec.data(), erasingVec.size() * sizeof(int3),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgIdxData->basePtr(), imgIdxVec.data(), imgIdxVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(valuesData->basePtr(), valuesVec.data(), valuesVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    // Call operator
    unsigned int  seed                 = 0;
    bool          random               = false;
    int           max_num_erasing_area = 2;
    cvcuda::Erase eraseOp(max_num_erasing_area);
    EXPECT_NO_THROW(eraseOp(stream, imgIn, imgOut, anchor, erasing, values, imgIdx, random, seed));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> test(outBufferSize, 0xA);

    //Check data
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->basePtr(), outBufferSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test[0], 1);
    EXPECT_EQ(test[9], 1);
    EXPECT_EQ(test[10], 0);
    EXPECT_EQ(test[9 * 640], 1);
    EXPECT_EQ(test[9 * 640 + 9], 1);
    EXPECT_EQ(test[9 * 640 + 10], 0);
    EXPECT_EQ(test[10 * 640], 0);
    EXPECT_EQ(test[10 * 640 + 10], 1);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpErase, OpErase_Varshape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc;
    imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{640, 480}, nvcv::FMT_U8));

    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    for (int i = 0; i < 1; ++i)
    {
        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * nvcv::FMT_U8.planePixelStrideBytes(0);

        EXPECT_EQ(cudaSuccess, cudaMemset2D(srcData->plane(0).basePtr, srcRowStride, 0, srcRowStride, srcHeight));
    }

    //parameters
    int          num_erasing_area = 2;
    nvcv::Tensor anchor({{num_erasing_area}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor erasing({{num_erasing_area}, "N"}, nvcv::TYPE_3S32);
    nvcv::Tensor values({{num_erasing_area}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor imgIdx({{num_erasing_area}, "N"}, nvcv::TYPE_S32);

    const auto *anchorData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(anchor.exportData());
    const auto *erasingData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(erasing.exportData());
    const auto *valuesData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(values.exportData());
    const auto *imgIdxData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgIdx.exportData());

    ASSERT_NE(nullptr, anchorData);
    ASSERT_NE(nullptr, erasingData);
    ASSERT_NE(nullptr, valuesData);
    ASSERT_NE(nullptr, imgIdxData);

    std::vector<int2>  anchorVec(num_erasing_area);
    std::vector<int3>  erasingVec(num_erasing_area);
    std::vector<int>   imgIdxVec(num_erasing_area);
    std::vector<float> valuesVec(num_erasing_area);

    anchorVec[0].x  = 0;
    anchorVec[0].y  = 0;
    erasingVec[0].x = 10;
    erasingVec[0].y = 10;
    erasingVec[0].z = 0x1;
    imgIdxVec[0]    = 0;
    valuesVec[0]    = 1.f;

    anchorVec[1].x  = 10;
    anchorVec[1].y  = 10;
    erasingVec[1].x = 20;
    erasingVec[1].y = 20;
    erasingVec[1].z = 0x1;
    imgIdxVec[1]    = 0;
    valuesVec[1]    = 1.f;

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchorData->basePtr(), anchorVec.data(), anchorVec.size() * sizeof(int2),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingData->basePtr(), erasingVec.data(), erasingVec.size() * sizeof(int3),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgIdxData->basePtr(), imgIdxVec.data(), imgIdxVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(valuesData->basePtr(), valuesVec.data(), valuesVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    // Call operator
    unsigned int  seed                 = 0;
    bool          random               = false;
    int           max_num_erasing_area = 2;
    cvcuda::Erase eraseOp(max_num_erasing_area);
    EXPECT_NO_THROW(eraseOp(stream, batchSrc, batchSrc, anchor, erasing, values, imgIdx, random, seed));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[0]->exportData());
    assert(dstData->numPlanes() == 1);

    int dstWidth  = dstData->plane(0).width;
    int dstHeight = dstData->plane(0).height;

    int dstRowStride = dstWidth * nvcv::FMT_U8.planePixelStrideBytes(0);

    std::vector<uint8_t> test(dstHeight * dstRowStride, 0xFF);

    // Copy output data to Host
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(test.data(), dstRowStride, dstData->plane(0).basePtr,
                                        dstData->plane(0).rowStride, dstRowStride, dstHeight, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test[0], 1);
    EXPECT_EQ(test[9], 1);
    EXPECT_EQ(test[10], 0);
    EXPECT_EQ(test[9 * 640], 1);
    EXPECT_EQ(test[9 * 640 + 9], 1);
    EXPECT_EQ(test[9 * 640 + 10], 0);
    EXPECT_EQ(test[10 * 640], 0);
    EXPECT_EQ(test[10 * 640 + 10], 1);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
