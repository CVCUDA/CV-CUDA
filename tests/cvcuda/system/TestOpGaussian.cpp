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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp> // for round
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

namespace {

// builds Gaussian and its per-image parameter tensors for a var-shape negative case
inline void InvokeGaussianVarShapeNegative(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                           const nvcv::ImageBatchVarShape &dst, int maxBatches,
                                           NVCVBorderType borderMode)
{
    const nvcv::Size2D kernelSize(3, 3);
    const int          numImages = src.numImages();
    auto               kernelSizeTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
    auto             sigmaTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2F64, double2{0.5, 0.5});
    cvcuda::Gaussian op(kernelSize, maxBatches);
    op(stream, src, dst, kernelSizeTensor, sigmaTensor, borderMode);
}

} // namespace

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

// clang-format off

NVCV_TEST_SUITE_P(OpGaussian, test::ValueList<int, int, int, NVCVImageFormat, int, int, double, double, NVCVBorderType>
{
    // width, height, batches,                    format, ksizeX, ksizeY, sigmaX, sigmaY,           borderMode
    {    176,    113,       1,      NVCV_IMAGE_FORMAT_U8,      3,      3,    0.5,    0.5, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,      NVCV_IMAGE_FORMAT_U8,      5,      5,   0.75,   0.75, NVCV_BORDER_CONSTANT},
    {    123,     33,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    1.0,    1.0, NVCV_BORDER_WRAP},
    {    111,     33,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    1.0,   -1.0, NVCV_BORDER_WRAP},
    {     42,     53,       4,   NVCV_IMAGE_FORMAT_RGBA8,      7,      7,    0.4,    0.4, NVCV_BORDER_REPLICATE},
    {     13,     42,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    0.9,    0.9, NVCV_BORDER_REFLECT},
    {     62,    111,       4,   NVCV_IMAGE_FORMAT_RGBA8,      9,      9,    0.8,    0.8, NVCV_BORDER_REFLECT101},
    {    128,    128,       1,      NVCV_IMAGE_FORMAT_U8,     -1,     -1,    0.5,    0.5, NVCV_BORDER_CONSTANT}
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

    int newKsizeX = ksizeX;
    int newKsizeY = ksizeY;

    // auto detection of kernel size from sigma
    bool is8U = format.planeDataType(0) == nvcv::TYPE_U8;
    if (ksizeX <= 0 && sigmaX > 0)
    {
        newKsizeX = nvcv::cuda::round<int>(sigmaX * (is8U ? 3 : 4) * 2 + 1) | 1;
    }
    if (ksizeY <= 0 && sigmaY > 0)
    {
        newKsizeY = nvcv::cuda::round<int>(sigmaY * (is8U ? 3 : 4) * 2 + 1) | 1;
    }

    nvcv::Size2D kernelSize(ksizeX, ksizeY);
    nvcv::Size2D newKernelSize(newKsizeX, newKsizeY);

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

    std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Gaussian gaussianOp(newKernelSize, 1);

    EXPECT_NO_THROW(gaussianOp(stream, inTensor, outTensor, kernelSize, sigma, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    std::vector<float> kernel = test::ComputeGaussianKernel(newKernelSize, sigma);

    test::Convolve(goldVec, outStrides, inVec, inStrides, shape, format, kernel, newKernelSize, kernelAnchor,
                   borderMode, borderValue);

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

    int newKsizeX = ksizeX;
    int newKsizeY = ksizeY;

    // auto detection of kernel size from sigma
    bool is8U = format.planeDataType(0) == nvcv::TYPE_U8;
    if (ksizeX <= 0 && sigmaX > 0)
    {
        newKsizeX = nvcv::cuda::round<int>(sigmaX * (is8U ? 3 : 4) * 2 + 1) | 1;
    }
    if (ksizeY <= 0 && sigmaY > 0)
    {
        newKsizeY = nvcv::cuda::round<int>(sigmaY * (is8U ? 3 : 4) * 2 + 1) | 1;
    }

    nvcv::Size2D kernelSize(newKsizeX, newKsizeY);

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

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
        std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });

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

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// Gaussian filters each channel independently, so a planar input is filtered plane-by-plane and
// must produce exactly the same pixels as the interleaved path. These tests feed identical uint8
// data through cvcuda::Gaussian in both layouts and require the (re-interleaved) planar output to
// match the interleaved output bit-for-bit.
// =============================================================================

// Parameters: width, height, kernelWidth, kernelHeight, sigmaX, sigmaY, borderMode, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpGaussianPlanar,
                  test::ValueList<int, int, int, int, double, double, NVCVBorderType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    { 64, 48, 3, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT,  2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 67, 51, 5, 3, 0.9, 0.7,   NVCV_BORDER_REFLECT, 1,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 50, 40, 7, 7, 1.2, 1.2, NVCV_BORDER_REPLICATE, 2, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    { 64, 48, 9, 5, 1.4, 0.8, NVCV_BORDER_REFLECT101, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 32, 28, 3, 5, 0.6, 1.1,      NVCV_BORDER_WRAP, 1, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    { 33, 29, 67, 67, 14.0, 14.0,  NVCV_BORDER_CONSTANT, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 33, 29, 75, 75, 16.0, 16.0, NVCV_BORDER_REPLICATE, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 33, 29, -1, -1, 15.0, 15.0,  NVCV_BORDER_CONSTANT, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 33, 29, 95, 95, 18.0, 18.0,   NVCV_BORDER_REFLECT, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 33, 29, -1, -1, 16.0, 16.0, NVCV_BORDER_REPLICATE, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
});

// clang-format on

TEST_P(OpGaussianPlanar, tensor_matches_interleaved)
{
    nvcv::Size2D kernelSize{GetParamValue<2>(), GetParamValue<3>()};
    double2      sigma{GetParamValue<4>(), GetParamValue<5>()};
    nvcv::Size2D maxKernelSize = kernelSize;
    if (maxKernelSize.w <= 0)
        maxKernelSize.w = nvcv::cuda::round<int>(sigma.x * 3 * 2 + 1) | 1;
    if (maxKernelSize.h <= 0)
        maxKernelSize.h = nvcv::cuda::round<int>(sigma.y * 3 * 2 + 1) | 1;
    NVCVBorderType borderMode = GetParamValue<6>();
    int            numImages  = GetParamValue<7>();

    test::planar::RunTensorParity(
        GetParamValue<8>(), GetParamValue<9>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), numImages,
        [kernelSize, maxKernelSize, sigma, borderMode, numImages](cudaStream_t s, const nvcv::Tensor &src,
                                                                  const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Gaussian op(maxKernelSize, numImages);
            EXPECT_NO_THROW(op(s, src, dst, kernelSize, sigma, borderMode));
        });
}

TEST_P(OpGaussianPlanar, varshape_matches_interleaved)
{
    nvcv::Size2D kernelSize{GetParamValue<2>(), GetParamValue<3>()};
    double2      sigma{GetParamValue<4>(), GetParamValue<5>()};
    nvcv::Size2D maxKernelSize = kernelSize;
    if (maxKernelSize.w <= 0)
        maxKernelSize.w = nvcv::cuda::round<int>(sigma.x * 3 * 2 + 1) | 1;
    if (maxKernelSize.h <= 0)
        maxKernelSize.h = nvcv::cuda::round<int>(sigma.y * 3 * 2 + 1) | 1;
    NVCVBorderType borderMode = GetParamValue<6>();
    int            numImages  = GetParamValue<7>();

    auto kernelSizeTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
    auto sigmaTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2F64, sigma);

    test::planar::RunVarShapeParity(
        GetParamValue<8>(), GetParamValue<9>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), numImages,
        [&kernelSizeTensor, &sigmaTensor, maxKernelSize, borderMode, numImages](
            cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::Gaussian op(maxKernelSize, numImages);
            EXPECT_NO_THROW(op(s, src, dst, kernelSizeTensor, sigmaTensor, borderMode));
        });
}

static auto OpGaussianNegativeParams()
{
    nvcv::test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, int, int, double, double, NVCVBorderType>
        params{
            {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,   nvcv::FMT_U16, 3, 3, 0.5, 0.5,
             NVCV_BORDER_CONSTANT}, // data type is different
            {NVCV_ERROR_INVALID_ARGUMENT,  nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 3, 3, 0.5, 0.5,
             NVCV_BORDER_CONSTANT}, // data format is different
            {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, 3, 3, 0.5, 0.5,
             NVCV_BORDER_CONSTANT}, // data format is different
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 3, 3, 0.5, 0.5,
                        static_cast<NVCVBorderType>(255));
#endif
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, 3, 3, 0.5, 0.5,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 4, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 3, 4, 0.5, 0.5, NVCV_BORDER_CONSTANT);
    return params;
}

NVCV_TEST_SUITE_P(OpGaussian_Negative, OpGaussianNegativeParams());

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
              nvcv::ProtectCall([&gaussianOp, &stream, &inTensor, &outTensor, &kernelSize, &sigma, &borderMode]
                                { gaussianOp(stream, inTensor, outTensor, kernelSize, sigma, borderMode); }));
}

NVCV_TEST_SUITE_P(OpGaussianVarshape_Negative, test::PlanarFilterVarShapeNegativeParams());

TEST_P(OpGaussianVarshape_Negative, op)
{
    test::planar::ExpectVarShapeUniformFormatRejected(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<3>(),
                                                      GetParamValue<4>(), GetParamValue<2>(),
                                                      InvokeGaussianVarShapeNegative);
}

TEST(OpGaussianVarshape_Negative, varshape_hasDifferentFormat)
{
    test::planar::ExpectVarShapeMixedFormatRejected(InvokeGaussianVarShapeNegative);
}

TEST(OpGaussian_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaGaussianCreate(nullptr, 224, 224, 10), NVCV_ERROR_INVALID_ARGUMENT);
}
