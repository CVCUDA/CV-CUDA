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
#include "HalfTestUtils.hpp"
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cuda_fp16.h>
#include <cvcuda/OpAverageBlur.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstring>
#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

namespace {

// builds AverageBlur and its per-image parameter tensors for a var-shape negative case
inline void InvokeAverageBlurVarShapeNegative(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                              const nvcv::ImageBatchVarShape &dst, int maxBatches,
                                              NVCVBorderType borderMode)
{
    const nvcv::Size2D kernelSize(3, 3);
    const int          numImages = src.numImages();
    auto               kernelSizeTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
    auto                kernelAnchorTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{-1, -1});
    cvcuda::AverageBlur op(kernelSize, maxBatches);
    op(stream, src, dst, kernelSizeTensor, kernelAnchorTensor, borderMode);
}

} // namespace

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

// clang-format off

NVCV_TEST_SUITE_P(OpAverageBlur, test::ValueList<int, int, int, NVCVImageFormat, int, int, int, int, NVCVBorderType>
{
    // width, height, batches,                    format, ksizeX, ksizeY, kanchorX, kanchorY,           borderMode
    {    176,    113,       1,      NVCV_IMAGE_FORMAT_U8,      3,      3,       -1,       -1, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,      NVCV_IMAGE_FORMAT_U8,      5,      5,        2,        2, NVCV_BORDER_CONSTANT},
    {    123,     33,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,        2,        2, NVCV_BORDER_WRAP},
    {     42,     53,       4,   NVCV_IMAGE_FORMAT_RGBA8,      7,      7,        5,        5, NVCV_BORDER_REPLICATE},
    {     13,     42,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,        1,        1, NVCV_BORDER_REFLECT},
    {     62,    111,       4,   NVCV_IMAGE_FORMAT_RGBA8,      9,      9,        8,        8, NVCV_BORDER_REFLECT101}
});

// clang-format on

TEST_P(OpAverageBlur, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int ksizeX   = GetParamValue<4>();
    int ksizeY   = GetParamValue<5>();
    int kanchorX = GetParamValue<6>();
    int kanchorY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    int3 shape{width, height, batches};

    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    int2 kernelAnchor{kanchorX, kanchorY};

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

    std::default_random_engine             randEng(0);
    std::uniform_int_distribution<uint8_t> rand(0, 255);

    std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::AverageBlur averageBlurOp(kernelSize, 1);

    EXPECT_NO_THROW(averageBlurOp(stream, inTensor, outTensor, kernelSize, kernelAnchor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    std::vector<float> kernel = test::ComputeMeanKernel(kernelSize);

    test::Convolve(goldVec, outStrides, inVec, inStrides, shape, format, kernel, kernelSize, kernelAnchor, borderMode,
                   borderValue);

    EXPECT_EQ(testVec, goldVec);
}

TEST_P(OpAverageBlur, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int ksizeX   = GetParamValue<4>();
    int ksizeY   = GetParamValue<5>();
    int kanchorX = GetParamValue<6>();
    int kanchorY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    int2 kernelAnchor{kanchorX, kanchorY};

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

    // Create kernel anchor tensor
    nvcv::Tensor kernelAnchorTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = kernelAnchorTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, kernelAnchor);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::AverageBlur averageBlurOp(kernelSize, batches);

    EXPECT_NO_THROW(averageBlurOp(stream, batchSrc, batchDst, kernelSizeTensor, kernelAnchorTensor, borderMode));

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
        std::vector<float> kernel = test::ComputeMeanKernel(kernelSize);

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, format, kernel, kernelSize, kernelAnchor,
                       borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}

// =============================================================================
// F16 (half) support
//
// F16 rides the same float-accumulating kernels as every other dtype, so the gold runs the FP32
// CPU reference (test::Convolve with an F32 format) on the widened half input. Bit-exactness
// against that gold is not applicable: the reference keeps the accumulator unrounded while the
// kernel narrows it once to half at the store. Per the HalfTestUtils.hpp policy the bound is
// kUlps = 4 half-ULPs (n-tap box average: one store rounding plus device FMA-contraction drift
// across the taps, the same n-tap-filter band Gaussian uses).
// =============================================================================

// clang-format off

NVCV_TEST_SUITE_P(OpAverageBlurF16, test::ValueList<int, int, int, NVCVImageFormat, int, int, int, int, NVCVBorderType>
{
    // width, height, batches,                    format, ksizeX, ksizeY, kanchorX, kanchorY,           borderMode
    {    176,    113,       1,     NVCV_IMAGE_FORMAT_F16,      3,      3,       -1,       -1, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,  NVCV_IMAGE_FORMAT_RGBf16,      5,      5,        2,        2, NVCV_BORDER_REPLICATE},
    {     42,     53,       4, NVCV_IMAGE_FORMAT_RGBAf16,      7,      7,        5,        5, NVCV_BORDER_REFLECT},
    {     62,     33,       3,  NVCV_IMAGE_FORMAT_RGBf16,      3,      3,        1,        1, NVCV_BORDER_WRAP},
    {    128,     64,       2,     NVCV_IMAGE_FORMAT_F16,      9,      9,       -1,       -1, NVCV_BORDER_REFLECT101},
});

// clang-format on

TEST_P(OpAverageBlurF16, tensor_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int ksizeX   = GetParamValue<4>();
    int ksizeY   = GetParamValue<5>();
    int kanchorX = GetParamValue<6>();
    int kanchorY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    int3 shape{width, height, batches};

    nvcv::Size2D kernelSize(ksizeX, ksizeY);
    int2         kernelAnchor{kanchorX, kanchorY};

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

    // [0, 1] input already quantized to half, so the FP32 gold consumes exactly the values the
    // kernel reads.
    std::vector<uint8_t>       inVec(inBufSize);
    std::default_random_engine randEng(0); // NOSONAR: deterministic test data, not security-sensitive.
    test::FillRandomHalfBytes(inVec, randEng, 0.f, 1.f);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    cvcuda::AverageBlur averageBlurOp(kernelSize, 1);

    EXPECT_NO_THROW(averageBlurOp(stream, inTensor, outTensor, kernelSize, kernelAnchor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> testVec(outBufSize);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // FP32 gold: widen the half input into a float buffer with doubled strides (__half is 2
    // bytes, float is 4) and run the shared F32 CPU reference on it.
    std::vector<float>   inFloats = test::HalfBytesToFloat(inVec);
    std::vector<uint8_t> inVecF(inBufSize * 2);
    std::memcpy(inVecF.data(), inFloats.data(), inVecF.size());

    long3 inStridesF{inStrides.x * 2, inStrides.y * 2, inStrides.z * 2};
    long3 outStridesF{outStrides.x * 2, outStrides.y * 2, outStrides.z * 2};

    std::vector<float> kernel = test::ComputeMeanKernel(kernelSize);

    std::vector<uint8_t> goldVecF(outBufSize * 2);
    test::Convolve(goldVecF, outStridesF, inVecF, inStridesF, shape, test::EquivalentFloatFormat(format), kernel,
                   kernelSize, kernelAnchor, borderMode, borderValue);

    // Gather per-element (skipping row padding) and compare within 4 half-ULPs (rationale above).
    const int           channels = format.numChannels();
    std::vector<float>  goldVals;
    std::vector<__half> testVals;
    const size_t        numValues = static_cast<size_t>(batches) * height * width * channels;
    for (size_t i = 0; i < numValues; ++i)
    {
        const int c = i % channels;
        const int x = (i / channels) % width;
        const int y = (i / channels / width) % height;
        const int b = i / channels / width / height;
        goldVals.push_back(*reinterpret_cast<const float *>(
            &goldVecF[b * outStridesF.x + y * outStridesF.y + x * outStridesF.z + c * sizeof(float)]));
        testVals.push_back(*reinterpret_cast<const __half *>(
            &testVec[b * outStrides.x + y * outStrides.y + x * outStrides.z + c * sizeof(__half)]));
    }
    test::ExpectNearHalfUlps(goldVals, testVals, 4.f);
}

TEST_P(OpAverageBlurF16, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int ksizeX   = GetParamValue<4>();
    int ksizeY   = GetParamValue<5>();
    int kanchorX = GetParamValue<6>();
    int kanchorY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    nvcv::Size2D kernelSize(ksizeX, ksizeY);
    int2         kernelAnchor{kanchorX, kanchorY};

    // Create input varshape with [0, 1] values already quantized to half
    std::default_random_engine    rng; // NOSONAR: deterministic test data, not security-sensitive.
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

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        test::FillRandomHalfBytes(srcVec[i], rng, 0.f, 1.f);

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    auto kernelSizeTensor   = test::planar::MakePerImageTensor(batches, nvcv::TYPE_2S32, int2{ksizeX, ksizeY});
    auto kernelAnchorTensor = test::planar::MakePerImageTensor(batches, nvcv::TYPE_2S32, kernelAnchor);

    cvcuda::AverageBlur averageBlurOp(kernelSize, batches);

    EXPECT_NO_THROW(averageBlurOp(stream, batchSrc, batchDst, kernelSizeTensor, kernelAnchorTensor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

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

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // FP32 gold on the widened half input (doubled strides), as in the tensor test above.
        std::vector<float>   srcFloats = test::HalfBytesToFloat(srcVec[i]);
        std::vector<uint8_t> srcVecF(srcVec[i].size() * 2);
        std::memcpy(srcVecF.data(), srcFloats.data(), srcVecF.size());

        long3 pitchesF{pitches.x * 2, pitches.y * 2, pitches.z * 2};

        std::vector<float> kernel = test::ComputeMeanKernel(kernelSize);

        std::vector<uint8_t> goldVecF(shape.y * pitchesF.y);
        test::Convolve(goldVecF, pitchesF, srcVecF, pitchesF, shape, test::EquivalentFloatFormat(format), kernel,
                       kernelSize, kernelAnchor, borderMode, borderValue);

        // 4 half-ULPs: n-tap FP32-accumulated filter with one half store rounding (see suite
        // comment). Both buffers are packed (no row padding), so they compare whole.
        std::vector<float> goldFloats(goldVecF.size() / sizeof(float));
        std::memcpy(goldFloats.data(), goldVecF.data(), goldVecF.size());
        test::ExpectNearHalfUlps(goldFloats, test::HalfBytesToFloat(testVec), 4.f);
    }
}

// AverageBlur filters each channel independently, so a planar input is filtered plane-by-plane and
// must produce exactly the same pixels as the interleaved path. These tests feed identical uint8
// data through cvcuda::AverageBlur in both layouts and require the re-interleaved planar output to
// match the interleaved output bit-for-bit.
// =============================================================================

// Parameters: width, height, kernelWidth, kernelHeight, borderMode, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpAverageBlurPlanar,
                  test::ValueList<int, int, int, int, NVCVBorderType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    { 64, 48, 3, 3, NVCV_BORDER_CONSTANT,  2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 67, 51, 5, 3,   NVCV_BORDER_REFLECT, 1,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 65, 49, 5, 5,  NVCV_BORDER_CONSTANT, 2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 50, 40, 7, 7, NVCV_BORDER_REPLICATE, 2, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    { 64, 48, 9, 5, NVCV_BORDER_REFLECT101, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 32, 28, 3, 5,      NVCV_BORDER_WRAP, 1, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    // F16 parity is bit-exact like the other dtypes: both layouts run the same float-accumulating
    // taps in the same order, so the single half store rounding matches.
    { 64, 48, 3, 3,  NVCV_BORDER_CONSTANT, 2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
    { 50, 40, 7, 7, NVCV_BORDER_REPLICATE, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

// clang-format on

TEST_P(OpAverageBlurPlanar, tensor_matches_interleaved)
{
    nvcv::Size2D   kernelSize{GetParamValue<2>(), GetParamValue<3>()};
    NVCVBorderType borderMode = GetParamValue<4>();
    int            numImages  = GetParamValue<5>();

    test::planar::RunTensorParity(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), numImages,
                                  [kernelSize, borderMode, numImages](cudaStream_t s, const nvcv::Tensor &src,
                                                                      const nvcv::Tensor &dst, nvcv::ImageFormat)
                                  {
                                      cvcuda::AverageBlur op(kernelSize, numImages);
                                      int2                kernelAnchor{-1, -1};
                                      EXPECT_NO_THROW(op(s, src, dst, kernelSize, kernelAnchor, borderMode));
                                  });
}

TEST_P(OpAverageBlurPlanar, varshape_matches_interleaved)
{
    nvcv::Size2D   kernelSize{GetParamValue<2>(), GetParamValue<3>()};
    NVCVBorderType borderMode = GetParamValue<4>();
    int            numImages  = GetParamValue<5>();

    auto kernelSizeTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
    auto kernelAnchorTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{-1, -1});

    test::planar::RunVarShapeParity(
        GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), numImages,
        [&kernelSizeTensor, &kernelAnchorTensor, kernelSize, borderMode, numImages](
            cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::AverageBlur op(kernelSize, numImages);
            EXPECT_NO_THROW(op(s, src, dst, kernelSizeTensor, kernelAnchorTensor, borderMode));
        });
}

static auto OpAverageBlurNegativeParams()
{
    nvcv::test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int, int, int, NVCVBorderType> params{
        {   nvcv::FMT_U8,   nvcv::FMT_U16, 3, 3, -1, -1, NVCV_BORDER_CONSTANT}, // data type is different
        { nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 3, 3, -1, -1, NVCV_BORDER_CONSTANT}, // interleaved in, planar out
        {nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, 3, 3, -1, -1, NVCV_BORDER_CONSTANT}, // planar in, interleaved out
        {  nvcv::FMT_F64,   nvcv::FMT_F64, 3, 3, -1, -1, NVCV_BORDER_CONSTANT}, // invalid data type (F16 is now valid)
        {   nvcv::FMT_U8,    nvcv::FMT_U8, 4, 3, -1, -1, NVCV_BORDER_CONSTANT}, // invalid kernel size
        {   nvcv::FMT_U8,    nvcv::FMT_U8, 3, 4, -1, -1, NVCV_BORDER_CONSTANT}, // invalid kernel size
        {   nvcv::FMT_U8,    nvcv::FMT_U8, 3, 3, -1, -2, NVCV_BORDER_CONSTANT}, // invalid kernel anchor
        {   nvcv::FMT_U8,    nvcv::FMT_U8, 3, 3, -2, -1, NVCV_BORDER_CONSTANT}, // invalid kernel anchor
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(nvcv::FMT_U8, nvcv::FMT_U8, 3, 3, -1, -1, static_cast<NVCVBorderType>(255));
#endif
    return params;
}

NVCV_TEST_SUITE_P(OpAverageBlur_Negative, OpAverageBlurNegativeParams());

TEST_P(OpAverageBlur_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inputFmt   = GetParamValue<0>();
    nvcv::ImageFormat outputFmt  = GetParamValue<1>();
    int               ksizeX     = GetParamValue<2>();
    int               ksizeY     = GetParamValue<3>();
    int               kanchorX   = GetParamValue<4>();
    int               kanchorY   = GetParamValue<5>();
    NVCVBorderType    borderMode = GetParamValue<6>();

    int width   = 24;
    int height  = 24;
    int batches = 1;

    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    int2 kernelAnchor{kanchorX, kanchorY};

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, inputFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, outputFmt);

    // run operator
    cvcuda::AverageBlur averageBlurOp(kernelSize, 1);

    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&averageBlurOp, &stream, &inTensor, &outTensor, &kernelSize, &kernelAnchor, &borderMode]
                          { averageBlurOp(stream, inTensor, outTensor, kernelSize, kernelAnchor, borderMode); }));
}

// Negative var-shape cases: interleaved<->planar layout mismatches, an unsupported dtype
// (F64 -- every dtype up to F32/F16 is accepted), invalid kernel anchor/size combinations,
// and an invalid border enum.
NVCV_TEST_SUITE_P(OpAverageBlurVarshape_Negative, test::PlanarFilterVarShapeNegativeParams());

TEST_P(OpAverageBlurVarshape_Negative, varshape_correct_output)
{
    test::planar::ExpectVarShapeUniformFormatRejected(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<3>(),
                                                      GetParamValue<4>(), GetParamValue<2>(),
                                                      InvokeAverageBlurVarShapeNegative);
}

TEST_P(OpAverageBlurVarshape_Negative, varshape_hasDifferentFormat)
{
    test::planar::ExpectVarShapeMixedFormatRejected(InvokeAverageBlurVarShapeNegative);
}

TEST(OpAverageBlur_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaAverageBlurCreate(nullptr, 224, 224, 10), NVCV_ERROR_INVALID_ARGUMENT);
}
