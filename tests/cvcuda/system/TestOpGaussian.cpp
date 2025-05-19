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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

// clang-format off

NVCV_TEST_SUITE_P(OpGaussian, test::ValueList<int, int, int, NVCVImageFormat, int, int, double, double, NVCVBorderType>
{
    // width, height, batches,                    format, ksizeX, ksizeY, sigmaX, sigmaY,           borderMode
    {    176,    113,       1,      NVCV_IMAGE_FORMAT_U8,      3,      3,    0.5,    0.5, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,      NVCV_IMAGE_FORMAT_U8,      5,      5,   0.75,   0.75, NVCV_BORDER_CONSTANT},
    {    123,     33,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    1.0,    1.0, NVCV_BORDER_WRAP},
    {     42,     53,       4,   NVCV_IMAGE_FORMAT_RGBA8,      7,      7,    0.4,    0.4, NVCV_BORDER_REPLICATE},
    {     13,     42,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    0.9,    0.9, NVCV_BORDER_REFLECT},
    {     62,    111,       4,   NVCV_IMAGE_FORMAT_RGBA8,      9,      9,    0.8,    0.8, NVCV_BORDER_REFLECT101}
});

// clang-format on

TEST_P(OpGaussian, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int    ksizeX = GetParamValue<4>();
    int    ksizeY = GetParamValue<5>();
    double sigmaX = GetParamValue<6>();
    double sigmaY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    int3 shape{width, height, batches};

    double2 sigma{sigmaX, sigmaY};
    int2    kernelAnchor{-1, -1};

    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    long3 inStrides{inAccess->sampleStride(), inAccess->rowStride(), inAccess->colStride()};
    long3 outStrides{outAccess->sampleStride(), outAccess->rowStride(), outAccess->colStride()};

    if (inData->rank() == 3)
    {
        inStrides.x  = inAccess->numRows() * inAccess->rowStride();
        outStrides.x = outAccess->numRows() * outAccess->rowStride();
    }

    long inBufSize  = inStrides.x * inAccess->numSamples();
    long outBufSize = outStrides.x * outAccess->numSamples();

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Gaussian gaussianOp(kernelSize, 1);

    EXPECT_NO_THROW(gaussianOp(stream, inTensor, outTensor, kernelSize, sigma, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    std::vector<float> kernel = test::ComputeGaussianKernel(kernelSize, sigma);

    test::Convolve(goldVec, outStrides, inVec, inStrides, shape, format, kernel, kernelSize, kernelAnchor, borderMode,
                   borderValue);

    EXPECT_EQ(testVec, goldVec);
}

TEST_P(OpGaussian, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int    ksizeX = GetParamValue<4>();
    int    ksizeY = GetParamValue<5>();
    double sigmaX = GetParamValue<6>();
    double sigmaY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    double2 sigma{sigmaX, sigmaY};
    int2    kernelAnchor{-1, -1};

    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
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

    // Create kernel size tensor
    nvcv::Tensor kernelSizeTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = kernelSizeTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{ksizeX, ksizeY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create sigma tensor
    nvcv::Tensor sigmaTensor({{batches}, "N"}, nvcv::TYPE_2F64);
    {
        auto dev = sigmaTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<double2> vec(batches, sigma);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(double2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::Gaussian gaussianOp(kernelSize, batches);

    EXPECT_NO_THROW(gaussianOp(stream, batchSrc, batchDst, kernelSizeTensor, sigmaTensor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
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
        std::vector<float> kernel = test::ComputeGaussianKernel(kernelSize, sigma);

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, format, kernel, kernelSize, kernelAnchor,
                       borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpGaussian_Negative, nvcv::test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, int, int, double, double, NVCVBorderType>{
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U16, 3, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT}, // data type is different
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 3, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT}, // data format is different
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, 3, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT}, // data format is not kNHWC/kHWC
#ifndef ENABLE_SANITIZER
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 3, 3, 0.5, 0.5, static_cast<NVCVBorderType>(255)}, // invalid borderType
#endif
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, 3, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT}, // invalid data type
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 4, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT}, // invalid kernel size
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 3, 4, 0.5, 0.5, NVCV_BORDER_CONSTANT}, // invalid kernel size
});

// clang-format on

TEST_P(OpGaussian_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat outputFmt          = GetParamValue<2>();
    int               ksizeX             = GetParamValue<3>();
    int               ksizeY             = GetParamValue<4>();
    double            sigmaX             = GetParamValue<5>();
    double            sigmaY             = GetParamValue<6>();
    NVCVBorderType    borderMode         = GetParamValue<7>();

    int width   = 24;
    int height  = 24;
    int batches = 1;

    double2      sigma{sigmaX, sigmaY};
    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, inputFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, outputFmt);

    // run operator
    cvcuda::Gaussian gaussianOp({11, 11}, 1);

    EXPECT_EQ(expectedReturnCode,
              nvcv::ProtectCall([&] { gaussianOp(stream, inTensor, outTensor, kernelSize, sigma, borderMode); }));
}

// clang-format off
NVCV_TEST_SUITE_P(OpGaussianVarshape_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType, int, int>{
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT, 3, 3},
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT, 3, 3},
    {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, NVCV_BORDER_CONSTANT, 3, 3},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 3, -1},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 5, 3},
#ifndef ENABLE_SANITIZER
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, static_cast<NVCVBorderType>(255), 3, 3},
#endif
});
// clang-format on

TEST_P(OpGaussianVarshape_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int          width  = 32;
    int          height = 32;
    nvcv::Size2D kernelSize(3, 3);

    nvcv::ImageFormat inputFmt   = GetParamValue<0>();
    nvcv::ImageFormat outputFmt  = GetParamValue<1>();
    NVCVBorderType    borderMode = GetParamValue<2>();
    int               batches    = GetParamValue<3>();
    int               maxBatches = GetParamValue<4>();

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inputFmt);
        imgDst.emplace_back(imgSrc[i].size(), outputFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel size tensor
    nvcv::Tensor kernelSizeTensor({{batches}, "N"}, nvcv::TYPE_2S32);

    // Create sigma tensor
    nvcv::Tensor sigmaTensor({{batches}, "N"}, nvcv::TYPE_2F64);

    // Run operator
    cvcuda::Gaussian gaussianOp(kernelSize, maxBatches);

    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&] { gaussianOp(stream, batchSrc, batchDst, kernelSizeTensor, sigmaTensor, borderMode); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGaussianVarshape_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_U8,          fmt},
        {         fmt, nvcv::FMT_U8}
    };

    for (auto testCase : testSet)
    {
        nvcv::ImageFormat inputFmtExtra  = std::get<0>(testCase);
        nvcv::ImageFormat outputFmtExtra = std::get<1>(testCase);

        int            width   = 32;
        int            height  = 32;
        int            batches = 3;
        nvcv::Size2D   kernelSize(3, 3);
        NVCVBorderType borderMode = NVCV_BORDER_CONSTANT;

        // Create input varshape
        std::default_random_engine         rng;
        std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
        std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < batches - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);
            imgDst.emplace_back(imgSrc[i].size(), fmt);
        }
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(batches);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        nvcv::ImageBatchVarShape batchDst(batches);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Create kernel size tensor
        nvcv::Tensor kernelSizeTensor({{batches}, "N"}, nvcv::TYPE_2S32);

        // Create sigma tensor
        nvcv::Tensor sigmaTensor({{batches}, "N"}, nvcv::TYPE_2F64);

        // Run operator
        cvcuda::Gaussian gaussianOp(kernelSize, batches);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall(
                      [&] { gaussianOp(stream, batchSrc, batchDst, kernelSizeTensor, sigmaTensor, borderMode); }));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
