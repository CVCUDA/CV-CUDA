/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "GaussianNoiseUtils.cuh"

#include <common/ValueTests.hpp>
#include <cvcuda/OpGaussianNoise.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <cmath>
#include <iostream>
#include <random>

inline uint8_t cast(float value)
{
    int v = (int)(value + (value >= 0 ? 0.5 : -0.5));
    return (uint8_t)((unsigned)v <= 255 ? v : v > 0 ? 255 : 0);
}

//test for RGB8
template<typename T>
static void GaussianNoise(std::vector<T> &src, std::vector<T> &dst, float mu, float sigma, int batch, bool per_channel)
{
    int mem_size = src.size();
    if (!per_channel)
        mem_size /= 3;
    float *rand_h = (float *)malloc(sizeof(float) * mem_size);
    get_random(rand_h, per_channel, batch, mem_size);

    int img_size = src.size() / 3;
    for (int i = 0; i < img_size; i++)
    {
        if (per_channel)
        {
            for (int ch = 0; ch < 3; ch++)
            {
                float delta     = mu + rand_h[i * 3 + ch] * sigma;
                dst[i * 3 + ch] = cast(src[i * 3 + ch] + delta);
            }
        }
        else
        {
            float delta    = mu + rand_h[i] * sigma;
            dst[i * 3]     = cast(src[i * 3] + delta);
            dst[i * 3 + 1] = cast(src[i * 3 + 1] + delta);
            dst[i * 3 + 2] = cast(src[i * 3 + 2] + delta);
        }
    }
    free(rand_h);
}

// clang-format off
NVCV_TEST_SUITE_P(OpGaussianNoise, nvcv::test::ValueList<int, int, int, float, float, bool>
{
    //batch,    height,     width,      mu,       sigma,     per_channel
    {     1,       480,       360,       0,       0.005,       false  },
    {     4,       100,       101,       0,       0.008,        true  },
    {     3,       360,       480,       0,       0.004,       false  },
    {     1,       800,       600,       0,       0.006,        true  },
});

// clang-format on

TEST_P(OpGaussianNoise, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int   batch       = GetParamValue<0>();
    int   height      = GetParamValue<1>();
    int   width       = GetParamValue<2>();
    float mu          = GetParamValue<3>();
    float sigma       = GetParamValue<4>();
    bool  per_channel = GetParamValue<5>();

    nvcv::ImageFormat fmt = nvcv::FMT_RGB8;
    using datatype        = uint8_t;
    nvcv::Tensor imgIn    = nvcv::util::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor imgOut   = nvcv::util::CreateTensor(batch, width, height, fmt);

    auto inData = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inData);
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);
    ASSERT_EQ(batch, inAccess->numSamples());

    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);
    ASSERT_EQ(batch, outAccess->numSamples());

    int64_t outSampleStride = outAccess->sampleStride();

    if (outData->rank() == 3)
    {
        outSampleStride = outAccess->numRows() * outAccess->rowStride();
    }

    int64_t outBufferSize = outSampleStride * outAccess->numSamples();

    // Set output buffer to dummy value
    EXPECT_EQ(cudaSuccess, cudaMemset(outAccess->sampleData(0), 0xFA, outBufferSize));

    //parameters
    nvcv::Tensor muval({{batch}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaval({{batch}, "N"}, nvcv::TYPE_F32);

    auto muData    = muval.exportData<nvcv::TensorDataStridedCuda>();
    auto sigmaData = sigmaval.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, muData);
    ASSERT_NE(nullptr, sigmaData);

    std::vector<float> muVec(batch, mu);
    std::vector<float> sigmaVec(batch, sigma);

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(muData->basePtr(), muVec.data(), muVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(sigmaData->basePtr(), sigmaVec.data(), sigmaVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    //Generate input
    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::default_random_engine        randEng;
    int                               rowStride = width * fmt.planePixelStrideBytes(0);

    for (int i = 0; i < batch; i++)
    {
        std::uniform_int_distribution<uint8_t> rand(0, 255);
        srcVec[i].resize(height * rowStride / sizeof(datatype));
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec[i].data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));
    }

    // Call operator
    int                   maxBatch = 4;
    unsigned long long    seed     = 12345;
    cvcuda::GaussianNoise GaussianNoiseOp(maxBatch);
    EXPECT_NO_THROW(GaussianNoiseOp(stream, imgIn, imgOut, muval, sigmaval, per_channel, seed));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        std::vector<datatype> testVec(height * rowStride / sizeof(datatype));
        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(i), outAccess->rowStride(),
                                            rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<datatype> goldVec(height * rowStride / sizeof(datatype));
        GaussianNoise<datatype>(srcVec[i], goldVec, mu, sigma, i, per_channel);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpGaussianNoise, varshape_correct_shape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int   batch       = GetParamValue<0>();
    int   height      = GetParamValue<1>();
    int   width       = GetParamValue<2>();
    float mu          = GetParamValue<3>();
    float sigma       = GetParamValue<4>();
    bool  per_channel = GetParamValue<5>();

    nvcv::ImageFormat fmt = nvcv::FMT_RGB8;
    using datatype        = uint8_t;

    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> rndHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc, imgDst;
    for (int i = 0; i < batch; ++i)
    {
        int rw = rndWidth(randEng);
        int rh = rndHeight(randEng);
        imgSrc.emplace_back(nvcv::Size2D{rw, rh}, fmt);
        imgDst.emplace_back(nvcv::Size2D{rw, rh}, fmt);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    //parameters
    nvcv::Tensor muval({{batch}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaval({{batch}, "N"}, nvcv::TYPE_F32);

    auto muData    = muval.exportData<nvcv::TensorDataStridedCuda>();
    auto sigmaData = sigmaval.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, muData);
    ASSERT_NE(nullptr, sigmaData);

    std::vector<float> muVec(batch, mu);
    std::vector<float> sigmaVec(batch, sigma);

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(muData->basePtr(), muVec.data(), muVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(sigmaData->basePtr(), sigmaVec.data(), sigmaVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    //Generate input
    std::vector<std::vector<datatype>> srcVec(batch);

    for (int i = 0; i < batch; i++)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcRowStride / sizeof(datatype));
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));
    }

    // Call operator
    int                   maxBatch = 4;
    unsigned long long    seed     = 12345;
    cvcuda::GaussianNoise GaussianNoiseOp(maxBatch);
    EXPECT_NO_THROW(GaussianNoiseOp(stream, batchSrc, batchDst, muval, sigmaval, per_channel, seed));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<datatype> testVec(dstHeight * dstRowStride / sizeof(datatype));

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<datatype> goldVec(dstHeight * dstRowStride / sizeof(datatype));
        GaussianNoise<datatype>(srcVec[i], goldVec, mu, sigma, i, per_channel);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
