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

#include <common/BorderUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpAdaptiveThreshold.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

using uchar = unsigned char;

// clang-format off

NVCV_TEST_SUITE_P(OpAdaptiveThreshold, test::ValueList<int, int, int, double, NVCVAdaptiveThresholdType, NVCVThresholdType, int, double>
{
    // width, height, batch, maxValue,      NVCVAdaptiveThresholdType,      NVCVThresholdType, blockSize,    c
    {    640,    360,     1,    100.0,    NVCV_ADAPTIVE_THRESH_MEAN_C,     NVCV_THRESH_BINARY,         3,  2.5},
    {   1280,    720,     2,    127.0,    NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY_INV,         5, -4.3},
    {    123,     33,     3,    100.0,NVCV_ADAPTIVE_THRESH_GAUSSIAN_C,     NVCV_THRESH_BINARY,         7,  9.2},
    {    361,    768,     4,    127.0,NVCV_ADAPTIVE_THRESH_GAUSSIAN_C, NVCV_THRESH_BINARY_INV,         3, -2.8}
});

// clang-format on

static void AdaptiveThreshold(std::vector<uint8_t> &hDst, const std::vector<uint8_t> &hSrc, long3 strides, int3 shape,
                              nvcv::ImageFormat fmt, const std::vector<float> &kernel, const nvcv::Size2D &kernelSize,
                              double maxValue, NVCVThresholdType thresholdType, double c)
{
    int2   kernelAnchor{-1, -1};
    float4 borderValue{0.f, 0.f, 0.f, 0.f};
    test::Convolve(hDst, strides, hSrc, strides, shape, fmt, kernel, kernelSize, kernelAnchor, NVCV_BORDER_REPLICATE,
                   borderValue);

    uchar iMaxValue = cuda::SaturateCast<uchar>(maxValue);
    int   idelta    = thresholdType == NVCV_THRESH_BINARY ? (int)std::ceil(c) : (int)std::floor(c);

    for (int b = 0; b < shape.z; ++b)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                uchar srcV = test::detail::ValueAt<uchar>(hSrc, strides, b, y, x);
                uchar res  = test::detail::ValueAt<uchar>(hDst, strides, b, y, x);
                uchar t;
                if (thresholdType == NVCV_THRESH_BINARY)
                {
                    t = srcV + idelta > cuda::SaturateCast<uchar>(res) ? iMaxValue : 0;
                }
                else
                {
                    t = srcV + idelta > cuda::SaturateCast<uchar>(res) ? 0 : iMaxValue;
                }

                test::detail::ValueAt<uchar>(hDst, strides, b, y, x) = t;
            }
        }
    }
}

TEST_P(OpAdaptiveThreshold, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                       width                 = GetParamValue<0>();
    int                       height                = GetParamValue<1>();
    int                       batch                 = GetParamValue<2>();
    double                    maxValue              = GetParamValue<3>();
    NVCVAdaptiveThresholdType adaptiveThresholdType = GetParamValue<4>();
    NVCVThresholdType         thresholdType         = GetParamValue<5>();
    int                       blockSize             = GetParamValue<6>();
    double                    c                     = GetParamValue<7>();

    nvcv::ImageFormat fmt    = nvcv::FMT_U8;
    nvcv::Tensor      imgIn  = nvcv::util::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor      imgOut = nvcv::util::CreateTensor(batch, width, height, fmt);

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

    //Generate input
    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::default_random_engine        randEng;
    int                               rowStride = width * fmt.planePixelStrideBytes(0);

    for (int i = 0; i < batch; i++)
    {
        std::uniform_int_distribution<uint8_t> rand(0, 255);
        srcVec[i].resize(height * rowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec[i].data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));
    }

    // Call operator
    cvcuda::AdaptiveThreshold adaptiveThresholdOp(blockSize, 1);

    EXPECT_NO_THROW(
        adaptiveThresholdOp(stream, imgIn, imgOut, maxValue, adaptiveThresholdType, thresholdType, blockSize, c));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // generate gold result
    std::vector<float> kernel;
    nvcv::Size2D       kernelSize(blockSize, blockSize);
    if (adaptiveThresholdType == NVCV_ADAPTIVE_THRESH_MEAN_C)
    {
        kernel = test::ComputeMeanKernel(kernelSize);
    }
    else
    {
        double2 sigma;
        sigma.x = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8;
        sigma.y = sigma.x;
        kernel  = test::ComputeGaussianKernel(kernelSize, sigma);
    }
    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        int                  vecSize = height * rowStride;
        std::vector<uint8_t> testVec(vecSize);
        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(i), outAccess->rowStride(),
                                            rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(vecSize);

        long3 strides{height * rowStride, rowStride, fmt.planePixelStrideBytes(0)};
        int3  shape{width, height, 1};
        AdaptiveThreshold(goldVec, srcVec[i], strides, shape, fmt, kernel, kernelSize, maxValue, thresholdType, c);

        EXPECT_EQ(testVec, goldVec);
    }
}

TEST_P(OpAdaptiveThreshold, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                       width                 = GetParamValue<0>();
    int                       height                = GetParamValue<1>();
    int                       batch                 = GetParamValue<2>();
    double                    maxValue              = GetParamValue<3>();
    NVCVAdaptiveThresholdType adaptiveThresholdType = GetParamValue<4>();
    NVCVThresholdType         thresholdType         = GetParamValue<5>();
    int                       blockSize             = GetParamValue<6>();
    double                    c                     = GetParamValue<7>();

    nvcv::ImageFormat fmt = nvcv::FMT_U8;

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::vector<int>                  srcVecRowStride(batch);

    for (int i = 0; i < batch; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);

        int srcRowStride   = imgSrc[i].size().w * fmt.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batch; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create maxValue tensor
    nvcv::Tensor maxValueTensor({{batch}, "N"}, nvcv::TYPE_F64);
    {
        auto dev = maxValueTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<double> vec(batch, maxValue);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create blockSize tensor
    nvcv::Tensor blockSizeTensor({{batch}, "N"}, nvcv::TYPE_S32);
    {
        auto dev = blockSizeTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int> vec(batch, blockSize);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create c tensor
    nvcv::Tensor cTensor({{batch}, "N"}, nvcv::TYPE_F64);
    {
        auto dev = cTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<double> vec(batch, c);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::AdaptiveThreshold adaptiveThresholdOp(blockSize, batch);

    EXPECT_NO_THROW(adaptiveThresholdOp(stream, batchSrc, batchDst, maxValueTensor, adaptiveThresholdType,
                                        thresholdType, blockSizeTensor, cTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    std::vector<float> kernel;
    nvcv::Size2D       kernelSize(blockSize, blockSize);
    if (adaptiveThresholdType == NVCV_ADAPTIVE_THRESH_MEAN_C)
    {
        kernel = test::ComputeMeanKernel(kernelSize);
    }
    else
    {
        double2 sigma;
        sigma.x = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8;
        sigma.y = sigma.x;
        kernel  = test::ComputeGaussianKernel(kernelSize, sigma);
    }
    for (int i = 0; i < batch; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int  dstRowStride = srcVecRowStride[i];
        int3 shape{srcData->plane(0).width, srcData->plane(0).height, 1};

        std::vector<uint8_t> testVec(shape.y * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // Generate gold result
        std::vector<uint8_t> goldVec(shape.y * dstRowStride);
        long3                strides{shape.y * dstRowStride, dstRowStride, fmt.planePixelStrideBytes(0)};
        AdaptiveThreshold(goldVec, srcVec[i], strides, shape, fmt, kernel, kernelSize, maxValue, thresholdType, c);

        EXPECT_EQ(testVec, goldVec);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpAdaptiveThresholdVarshape_Negative, test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, int, NVCVAdaptiveThresholdType, NVCVThresholdType>{
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 6, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY}, // exceed max batch size
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 2, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY}, // data format is different
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, 2, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY}, // data format is not kNHWC/kHWC
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, 2, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY}, // invalid data type
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY}, // invalid channel
#ifndef ENABLE_SANITIZER
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 2, static_cast<NVCVAdaptiveThresholdType>(255), NVCV_THRESH_BINARY},
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 2, NVCV_ADAPTIVE_THRESH_MEAN_C, static_cast<NVCVThresholdType>(255)},
#endif
});

NVCV_TEST_SUITE_P(OpAdaptiveThreshold_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVAdaptiveThresholdType, NVCVThresholdType, int>{
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY, 3},
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY, 3},
    {nvcv::FMT_U8, nvcv::FMT_S16, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY, 3},
    {nvcv::FMT_S16, nvcv::FMT_S16, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY, 3},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY, 3},
    {nvcv::FMT_U8, nvcv::FMT_U8, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY, 2},
#ifndef ENABLE_SANITIZER
    {nvcv::FMT_U8, nvcv::FMT_U8, static_cast<NVCVAdaptiveThresholdType>(255), NVCV_THRESH_BINARY, 3},
    {nvcv::FMT_U8, nvcv::FMT_U8, NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_OTSU, 3},
#endif
});
// clang-format on

TEST_P(OpAdaptiveThreshold_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    width    = 24;
    const int    height   = 24;
    const int    batch    = 2;
    const double maxValue = 127.0;
    double       c        = 2.5;

    nvcv::ImageFormat               inputFmt              = GetParamValue<0>();
    nvcv::ImageFormat               outputFmt             = GetParamValue<1>();
    const NVCVAdaptiveThresholdType adaptiveThresholdType = GetParamValue<2>();
    const NVCVThresholdType         thresholdType         = GetParamValue<3>();
    const int                       blockSize             = GetParamValue<4>();

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batch, width, height, inputFmt);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batch, width, height, outputFmt);

    // Call operator
    cvcuda::AdaptiveThreshold adaptiveThresholdOp(blockSize, 1);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&] {
                                                   adaptiveThresholdOp(stream, imgIn, imgOut, maxValue,
                                                                       adaptiveThresholdType, thresholdType, blockSize,
                                                                       c);
                                               }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpAdaptiveThresholdVarshape_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus                expectedReturnCode    = GetParamValue<0>();
    nvcv::ImageFormat         inputFmt              = GetParamValue<1>();
    nvcv::ImageFormat         outputFmt             = GetParamValue<2>();
    int                       batch                 = GetParamValue<3>();
    NVCVAdaptiveThresholdType adaptiveThresholdType = GetParamValue<4>();
    NVCVThresholdType         thresholdType         = GetParamValue<5>();

    int width     = 24;
    int height    = 24;
    int blockSize = 3;
    int maxBatch  = 5;

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc;

    for (int i = 0; i < batch; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inputFmt);
    }
    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batch; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), outputFmt);
    }
    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create maxValue tensor
    nvcv::Tensor maxValueTensor({{batch}, "N"}, nvcv::TYPE_F64);
    // Create blockSize tensor
    nvcv::Tensor blockSizeTensor({{batch}, "N"}, nvcv::TYPE_S32);
    // Create c tensor
    nvcv::Tensor cTensor({{batch}, "N"}, nvcv::TYPE_F64);

    // Run operator
    cvcuda::AdaptiveThreshold adaptiveThresholdOp(blockSize, maxBatch);

    EXPECT_EQ(expectedReturnCode, nvcv::ProtectCall(
                                      [&]
                                      {
                                          adaptiveThresholdOp(stream, batchSrc, batchDst, maxValueTensor,
                                                              adaptiveThresholdType, thresholdType, blockSizeTensor,
                                                              cTensor);
                                      }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdaptiveThresholdVarshape_Negative, varshape_hasDifferentFormat)
{
    nvcv::ImageFormat         fmt                   = nvcv::FMT_U8;
    int                       batch                 = 2;
    NVCVAdaptiveThresholdType adaptiveThresholdType = NVCV_ADAPTIVE_THRESH_MEAN_C;
    NVCVThresholdType         thresholdType         = NVCV_THRESH_BINARY;

    int width     = 24;
    int height    = 24;
    int blockSize = 3;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_U16,           fmt},
        {          fmt, nvcv::FMT_U16}
    };

    for (auto testCase : testSet)
    {
        nvcv::ImageFormat inputFmtExtra  = std::get<0>(testCase);
        nvcv::ImageFormat outputFmtExtra = std::get<1>(testCase);

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        // Create input varshape
        std::default_random_engine         rng;
        std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
        std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

        std::vector<nvcv::Image> imgSrc;

        for (int i = 0; i < batch - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        nvcv::ImageBatchVarShape batchSrc(batch);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        // Create output varshape
        std::vector<nvcv::Image> imgDst;
        for (int i = 0; i < batch - 1; ++i)
        {
            imgDst.emplace_back(imgSrc[i].size(), fmt);
        }
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);
        nvcv::ImageBatchVarShape batchDst(batch);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Create maxValue tensor
        nvcv::Tensor              maxValueTensor({{batch}, "N"}, nvcv::TYPE_F64);
        // Create blockSize tensor
        nvcv::Tensor              blockSizeTensor({{batch}, "N"}, nvcv::TYPE_S32);
        // Create c tensor
        nvcv::Tensor              cTensor({{batch}, "N"}, nvcv::TYPE_F64);
        // Run operator
        cvcuda::AdaptiveThreshold adaptiveThresholdOp(blockSize, batch);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                                   [&]
                                                   {
                                                       adaptiveThresholdOp(stream, batchSrc, batchDst, maxValueTensor,
                                                                           adaptiveThresholdType, thresholdType,
                                                                           blockSizeTensor, cTensor);
                                                   }));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
}
