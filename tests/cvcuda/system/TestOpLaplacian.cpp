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
#include <cvcuda/OpLaplacian.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

namespace {

// Negative cases intentionally vary parameter-tensor lengths to exercise validation.
inline void InvokeLaplacianVarShapeNegative(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                            const nvcv::ImageBatchVarShape &dst, int maxBatches,
                                            NVCVBorderType borderMode)
{
    const int         ksize       = 3;
    const int         numParams   = maxBatches < 0 ? 0 : maxBatches;
    auto              ksizeTensor = test::planar::MakePerImageTensor(numParams, nvcv::TYPE_S32, ksize);
    auto              scaleTensor = test::planar::MakePerImageTensor(numParams, nvcv::TYPE_F32, 1.0f);
    cvcuda::Laplacian op;
    op(stream, src, dst, ksizeTensor, scaleTensor, borderMode);
}

} // namespace

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

static constexpr std::array<float, 9> kLaplacianKernel1 = {0.0f, 1.0f, 0.0f, 1.0f, -4.0f, 1.0f, 0.0f, 1.0f, 0.0f};
static constexpr std::array<float, 9> kLaplacianKernel3 = {2.0f, 0.0f, 2.0f, 0.0f, -8.0f, 0.0f, 2.0f, 0.0f, 2.0f};

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

    // Create kernel aperture size tensor
    nvcv::Tensor ksizeTensor({{batches}, "N"}, nvcv::TYPE_S32);
    {
        auto dev = ksizeTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int> vec(batches, ksize);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create scale tensor
    nvcv::Tensor scaleTensor({{batches}, "N"}, nvcv::TYPE_F32);
    {
        auto dev = scaleTensor.exportData<nvcv::TensorDataStridedCuda>();
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
        std::vector<float> kernel(9);
        nvcv::Size2D       kernelSize{3, 3};
        int2               kernelAnchor{kernelSize.w / 2, kernelSize.h / 2};

        for (int kernelIndex = 0; kernelIndex < 9; ++kernelIndex)
        {
            if (ksize == 1)
            {
                kernel[kernelIndex] = kLaplacianKernel1[kernelIndex] * scale;
            }
            else if (ksize == 3)
            {
                kernel[kernelIndex] = kLaplacianKernel3[kernelIndex] * scale;
            }
        }

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, format, kernel, kernelSize, kernelAnchor,
                       borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}

// Laplacian filters each channel independently, so a planar input is filtered plane-by-plane and
// must produce exactly the same pixels as the interleaved path. These tests feed identical data
// through cvcuda::Laplacian in both layouts and require the re-interleaved planar output to match
// the interleaved output bit-for-bit.
// =============================================================================

// Parameters: width, height, ksize, scale, borderMode, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpLaplacianPlanar,
                  test::ValueList<int, int, int, float, NVCVBorderType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    { 64, 48, 1, 1.0f,   NVCV_BORDER_CONSTANT, 2,  nvcv::FMT_RGB8p,   nvcv::FMT_RGB8},
    { 67, 51, 3, 2.0f,    NVCV_BORDER_REFLECT, 1,  nvcv::FMT_RGB8p,   nvcv::FMT_RGB8},
    { 50, 40, 1, 1.0f,  NVCV_BORDER_REPLICATE, 2, nvcv::FMT_RGBA8p,  nvcv::FMT_RGBA8},
    { 64, 48, 3, 1.0f, NVCV_BORDER_REFLECT101, 1, nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    { 32, 28, 1, 3.0f,       NVCV_BORDER_WRAP, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    { 40, 33, 3, 1.0f,   NVCV_BORDER_CONSTANT, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
});

// clang-format on

TEST_P(OpLaplacianPlanar, tensor_matches_interleaved)
{
    int            ksize      = GetParamValue<2>();
    float          scale      = GetParamValue<3>();
    NVCVBorderType borderMode = GetParamValue<4>();
    int            numImages  = GetParamValue<5>();

    test::planar::RunTensorParity(
        GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), numImages,
        [ksize, scale, borderMode](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Laplacian op;
            EXPECT_NO_THROW(op(s, src, dst, ksize, scale, borderMode));
        });
}

TEST_P(OpLaplacianPlanar, varshape_matches_interleaved)
{
    int            ksize      = GetParamValue<2>();
    float          scale      = GetParamValue<3>();
    NVCVBorderType borderMode = GetParamValue<4>();
    int            numImages  = GetParamValue<5>();

    auto ksizeTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_S32, ksize);
    auto scaleTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_F32, scale);

    test::planar::RunVarShapeParity(
        GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), numImages,
        [&ksizeTensor, &scaleTensor, borderMode](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                 const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::Laplacian op;
            EXPECT_NO_THROW(op(s, src, dst, ksizeTensor, scaleTensor, borderMode));
        });
}

static auto OpLaplacianNegativeParams()
{
    nvcv::test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, int, float, NVCVBorderType> params{
        {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,    nvcv::FMT_U8, 7, 0.5,NVCV_BORDER_CONSTANT        }, // invalid kernel size
        {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,   nvcv::FMT_U16, 3, 0.5,
         NVCV_BORDER_CONSTANT}, // data type is different
        {NVCV_ERROR_INVALID_ARGUMENT,  nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 3, 0.5,
         NVCV_BORDER_CONSTANT}, // interleaved in, planar out
        {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, 3, 0.5,
         NVCV_BORDER_CONSTANT}, // planar in, interleaved out
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 3, 0.5,
                        static_cast<NVCVBorderType>(255));
#endif
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, 3, 0.5, NVCV_BORDER_CONSTANT);
    return params;
}

NVCV_TEST_SUITE_P(OpLaplacian_Negative, OpLaplacianNegativeParams());

TEST_P(OpLaplacian_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat outputFmt          = GetParamValue<2>();
    int               ksize              = GetParamValue<3>();
    float             scale              = GetParamValue<4>();
    NVCVBorderType    borderMode         = GetParamValue<5>();

    int width   = 24;
    int height  = 24;
    int batches = 1;

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, inputFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, outputFmt);

    // run operator
    cvcuda::Laplacian laplacianOp;
    EXPECT_EQ(expectedReturnCode,
              nvcv::ProtectCall([&laplacianOp, &stream, &inTensor, &outTensor, &ksize, &scale, &borderMode]
                                { laplacianOp(stream, inTensor, outTensor, ksize, scale, borderMode); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

NVCV_TEST_SUITE_P(OpLaplacianVarshape_Negative, test::PlanarFilterVarShapeNegativeParams());

TEST_P(OpLaplacianVarshape_Negative, op)
{
    test::planar::ExpectVarShapeUniformFormatRejected(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<3>(),
                                                      GetParamValue<4>(), GetParamValue<2>(),
                                                      InvokeLaplacianVarShapeNegative);
}

TEST(OpLaplacianVarshape_Negative, varshape_hasDifferentFormat)
{
    test::planar::ExpectVarShapeMixedFormatRejected(InvokeLaplacianVarShapeNegative);
}

TEST(OpLaplacian_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaLaplacianCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
