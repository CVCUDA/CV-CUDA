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
#include <cvcuda/OpLaplacian.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <tuple>

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
    // Preserve valid image dimensions when a one-pixel case is scaled down.
    return std::max(1, static_cast<int>(size * scale));
}

static void FillFiniteScalarFloatBytes(std::vector<uint8_t> &bytes, std::default_random_engine &rng)
{
    std::uniform_real_distribution<float> values(-1.f, 1.f);
    for (size_t offset = 0; offset + sizeof(float) <= bytes.size(); offset += sizeof(float))
    {
        const float value = values(rng);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }
}

static bool IsFiniteWarpRowFixture(int width, int height, nvcv::ImageFormat format, NVCVBorderType border)
{
    return (format == NVCV_IMAGE_FORMAT_RGBf32 || format == NVCV_IMAGE_FORMAT_RGBAf32)
        && border == NVCV_BORDER_REFLECT101
        && ((width == 31 && height == 7) || (width == 33 && height == 9) || (width == 65 && height == 17));
}

static constexpr std::array<float, 9> kLaplacianKernel1 = {0.0f, 1.0f, 0.0f, 1.0f, -4.0f, 1.0f, 0.0f, 1.0f, 0.0f};
static constexpr std::array<float, 9> kLaplacianKernel3 = {2.0f, 0.0f, 2.0f, 0.0f, -8.0f, 0.0f, 2.0f, 0.0f, 2.0f};

namespace {

using GoldParams = std::tuple<int, int, int, NVCVImageFormat, int, float, NVCVBorderType>;

void CheckLaplacianGold(const std::vector<uint8_t> &input, long3 inputStrides, const std::vector<uint8_t> &actual,
                        long3 outputStrides, int3 shape, nvcv::ImageFormat format, int ksize, float scale,
                        NVCVBorderType border)
{
    const bool           half     = test::IsF16Format(format);
    const int            channels = format.numChannels();
    std::vector<uint8_t> widened;
    if (half)
    {
        const auto floats = test::HalfBytesToFloat(input);
        widened.resize(input.size() * 2);
        std::memcpy(widened.data(), floats.data(), widened.size());
        inputStrides = long3{inputStrides.x * 2, inputStrides.y * 2, inputStrides.z * 2};
        format       = test::EquivalentFloatFormat(format);
    }
    const long3 goldStrides
        = half ? long3{outputStrides.x * 2, outputStrides.y * 2, outputStrides.z * 2} : outputStrides;
    std::vector<uint8_t> gold(actual.size() * (half ? 2 : 1));
    std::vector<float>   kernel(9);
    for (int i = 0; i < 9; ++i) kernel[i] = (ksize == 1 ? kLaplacianKernel1[i] : kLaplacianKernel3[i]) * scale;
    int2 anchor{1, 1};
    test::Convolve(gold, goldStrides, half ? widened : input, inputStrides, shape, format, kernel, nvcv::Size2D{3, 3},
                   anchor, border, cuda::SetAll<float4>(0));
    if (!half)
    {
        EXPECT_EQ(actual, gold);
        return;
    }
    // Cancellation needs an absolute floor at the [0,1] input scale: retain four half ULPs.
    for (size_t i = 0; i < static_cast<size_t>(shape.x) * shape.y * shape.z * channels; ++i)
    {
        const size_t channel = i % channels;
        const size_t column  = (i / channels) % shape.x;
        const size_t row     = (i / channels / shape.x) % shape.y;
        const size_t batch   = i / channels / shape.x / shape.y;
        float        expected;
        __half       value;
        std::memcpy(&expected,
                    &gold[batch * goldStrides.x + row * goldStrides.y + column * goldStrides.z + channel * 4], 4);
        std::memcpy(&value,
                    &actual[batch * outputStrides.x + row * outputStrides.y + column * outputStrides.z + channel * 2],
                    2);
        EXPECT_NEAR(__half2float(value), expected, 4.f * test::HalfUlp(std::max(std::fabs(expected), 1.f)))
            << "F16 output differs at index " << i;
    }
}

} // namespace

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
    {      4,      3,       4, NVCV_IMAGE_FORMAT_RGBAf32,     1,  1.0f, NVCV_BORDER_REFLECT101},
    // Cover a full 128-pixel block and its one-, two-, and three-pixel tails.
    {    128,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {    129,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    {    130,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {    131,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    // Float row-geometry probes cover full warps, partial warps and row-block tails.
    {     31,      7,       2,    NVCV_IMAGE_FORMAT_RGBf32,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {     31,      7,       2,    NVCV_IMAGE_FORMAT_RGBf32,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {     33,      9,       2,    NVCV_IMAGE_FORMAT_RGBf32,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {     33,      9,       2,    NVCV_IMAGE_FORMAT_RGBf32,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {     65,     17,       2,    NVCV_IMAGE_FORMAT_RGBf32,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {     65,     17,       2,    NVCV_IMAGE_FORMAT_RGBf32,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {     31,      7,       2,    NVCV_IMAGE_FORMAT_RGBAf32,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {     31,      7,       2,    NVCV_IMAGE_FORMAT_RGBAf32,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {     33,      9,       2,    NVCV_IMAGE_FORMAT_RGBAf32,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {     33,      9,       2,    NVCV_IMAGE_FORMAT_RGBAf32,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {     65,     17,       2,    NVCV_IMAGE_FORMAT_RGBAf32,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {     65,     17,       2,    NVCV_IMAGE_FORMAT_RGBAf32,     3,  0.5f, NVCV_BORDER_REFLECT101},
    // Constant RGB tiles need interior rows and columns plus partial right/bottom halos.
    {    257,     33,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {    257,     33,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    {    258,     34,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {    258,     34,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    // Cover no-interior images and complete four-output spans next to constant-border tails.
    {      1,      1,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {      1,      1,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    {      1,      3,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {      1,      3,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    {      3,      1,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {      3,      1,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    {    129,      3,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {    129,      3,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    {    257,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_CONSTANT},
    {    257,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_CONSTANT},
    // Exercise the tiled byte paths at small images, full tiles, and partial tiles.
    {      1,      1,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_REPLICATE},
    {    129,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_REPLICATE},
    {    127,     15,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_REFLECT},
    {      1,      1,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_REFLECT},
    {    128,     16,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {    129,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {      1,      1,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_WRAP},
    {    127,     15,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_WRAP},
    {    127,     15,       2,   NVCV_IMAGE_FORMAT_RGBA8,     1,  0.5f, NVCV_BORDER_REPLICATE},
    {      1,      1,       2,   NVCV_IMAGE_FORMAT_RGBA8,     3,  0.5f, NVCV_BORDER_REPLICATE},
    {      1,      1,       2,   NVCV_IMAGE_FORMAT_RGBA8,     1,  0.5f, NVCV_BORDER_REFLECT},
    {    129,     17,       2,   NVCV_IMAGE_FORMAT_RGBA8,     3,  0.5f, NVCV_BORDER_REFLECT},
    {    129,     17,       2,   NVCV_IMAGE_FORMAT_RGBA8,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {    128,     16,       2,   NVCV_IMAGE_FORMAT_RGBA8,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {      1,      1,       2,   NVCV_IMAGE_FORMAT_RGBA8,     1,  0.5f, NVCV_BORDER_WRAP},
    {    129,     17,       2,   NVCV_IMAGE_FORMAT_RGBA8,     3,  0.5f, NVCV_BORDER_WRAP},
    // The second 128-pixel tile becomes fully interior at width 257 and height 33.
    // Width 322 keeps even the smallest randomized VarShape image above that boundary.
    {    256,     32,       2,      NVCV_IMAGE_FORMAT_U8,     1,  0.5f, NVCV_BORDER_REPLICATE},
    {    257,     33,       2,      NVCV_IMAGE_FORMAT_U8,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {    258,     34,       2,      NVCV_IMAGE_FORMAT_U8,     3,  0.5f, NVCV_BORDER_WRAP},
    {    322,     49,       2,      NVCV_IMAGE_FORMAT_U8,     1,  0.5f, NVCV_BORDER_WRAP},
    {    322,     49,       2,      NVCV_IMAGE_FORMAT_U8,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {    322,     49,       2,    NVCV_IMAGE_FORMAT_RGB8,     1,  0.5f, NVCV_BORDER_REPLICATE},
    {    322,     49,       2,    NVCV_IMAGE_FORMAT_RGB8,     3,  0.5f, NVCV_BORDER_REFLECT},
    {    322,     49,       2,   NVCV_IMAGE_FORMAT_RGBA8,     1,  0.5f, NVCV_BORDER_WRAP},
    {    322,     49,       2,   NVCV_IMAGE_FORMAT_RGBA8,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {    322,     49,       2,     NVCV_IMAGE_FORMAT_F32,     1,  0.5f, NVCV_BORDER_REPLICATE},
    {    322,     49,       2,     NVCV_IMAGE_FORMAT_F32,     3,  0.5f, NVCV_BORDER_REFLECT}
});

// clang-format on

static void CheckTensorGoldCase(const GoldParams &params)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const auto &[width, height, batches, imageFormat, ksize, scale, borderMode] = params;
    nvcv::ImageFormat format{imageFormat};
    int3              shape{width, height, batches};

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

    if (test::IsF16Format(format))
        test::FillRandomHalfBytes(inVec, randEng, 0.f, 1.f);
    else if (format == NVCV_IMAGE_FORMAT_F32 || IsFiniteWarpRowFixture(width, height, format, borderMode))
        FillFiniteScalarFloatBytes(inVec, randEng);
    else
        std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Laplacian laplacianOp;

    EXPECT_NO_THROW(laplacianOp(stream, inTensor, outTensor, ksize, scale, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    ASSERT_NO_FATAL_FAILURE(
        CheckLaplacianGold(inVec, inStrides, testVec, outStrides, shape, format, ksize, scale, borderMode));
}

TEST_P(OpLaplacian, correct_output)
{
    ASSERT_NO_FATAL_FAILURE(CheckTensorGoldCase(GetParam()));
}

static void CheckVarShapeGoldCase(const GoldParams &params)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const auto &[width, height, batches, imageFormat, ksize, scale, borderMode] = params;
    nvcv::ImageFormat             format{imageFormat};
    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
    {
        nvcv::Size2D size{udistWidth(rng), udistHeight(rng)};
        // Keep one new finite geometry fixture at the exact warp/block boundary.
        if (i == 0 && IsFiniteWarpRowFixture(width, height, format, borderMode))
            size = nvcv::Size2D{width, height};
        // Pin the first image only in the new constant-RGB block-tail fixtures.
        if (i == 0 && format == NVCV_IMAGE_FORMAT_RGB8 && borderMode == NVCV_BORDER_CONSTANT && width >= 128
            && width <= 131)
            size.w = width;
        imgSrc.emplace_back(size, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        if (test::IsF16Format(format))
            test::FillRandomHalfBytes(srcVec[i], rng, 0.f, 1.f);
        else if (format == NVCV_IMAGE_FORMAT_F32 || IsFiniteWarpRowFixture(width, height, format, borderMode))
            FillFiniteScalarFloatBytes(srcVec[i], rng);
        else
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

    auto ksizeTensor = test::planar::MakePerImageTensor(batches, nvcv::TYPE_S32, ksize);
    auto scaleTensor = test::planar::MakePerImageTensor(batches, nvcv::TYPE_F32, scale);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

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

        ASSERT_NO_FATAL_FAILURE(
            CheckLaplacianGold(srcVec[i], pitches, testVec, pitches, shape, format, ksize, scale, borderMode));
    }
}

TEST_P(OpLaplacian, varshape_correct_output)
{
    ASSERT_NO_FATAL_FAILURE(CheckVarShapeGoldCase(GetParam()));
}

// clang-format off

NVCV_TEST_SUITE_P(OpLaplacianF16, test::ValueList<int, int, int, NVCVImageFormat, int, float, NVCVBorderType>
{
    // width, height, batches,                    format, ksize, scale,           borderMode
    {    176,    113,       1,     NVCV_IMAGE_FORMAT_F16,     1,  1.0f, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,  NVCV_IMAGE_FORMAT_RGBf16,     3,  1.0f, NVCV_BORDER_REPLICATE},
    {     42,     53,       4, NVCV_IMAGE_FORMAT_RGBAf16,     1,  2.0f, NVCV_BORDER_REFLECT},
    {     62,     33,       3,  NVCV_IMAGE_FORMAT_RGBf16,     3,  0.5f, NVCV_BORDER_WRAP},
    {    128,     64,       2,     NVCV_IMAGE_FORMAT_F16,     3,  1.0f, NVCV_BORDER_REFLECT101},
    {      1,      1,       2,  NVCV_IMAGE_FORMAT_RGBf16,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {    127,     15,       2,  NVCV_IMAGE_FORMAT_RGBf16,     1,  0.5f, NVCV_BORDER_REPLICATE},
    {    128,     16,       2,  NVCV_IMAGE_FORMAT_RGBf16,     1,  0.5f, NVCV_BORDER_REFLECT},
    {    129,     17,       2,  NVCV_IMAGE_FORMAT_RGBf16,     1,  0.5f, NVCV_BORDER_WRAP},
    {    129,     17,       2,  NVCV_IMAGE_FORMAT_RGBf16,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {      1,      1,       2,  NVCV_IMAGE_FORMAT_RGBf16,     3,  0.5f, NVCV_BORDER_REFLECT},
    {      1,      1,       2,     NVCV_IMAGE_FORMAT_F16,     1,  0.5f, NVCV_BORDER_REFLECT101},
    {    129,     17,       2,     NVCV_IMAGE_FORMAT_F16,     1,  0.5f, NVCV_BORDER_WRAP},
    {    127,     15,       2,     NVCV_IMAGE_FORMAT_F16,     3,  0.5f, NVCV_BORDER_REPLICATE},
    {    129,     17,       2,     NVCV_IMAGE_FORMAT_F16,     3,  0.5f, NVCV_BORDER_REFLECT},
    {    257,     33,       2,     NVCV_IMAGE_FORMAT_F16,     1,  0.5f, NVCV_BORDER_REPLICATE},
    {    258,     34,       2,     NVCV_IMAGE_FORMAT_F16,     3,  0.5f, NVCV_BORDER_WRAP},
    {    322,     49,       2,     NVCV_IMAGE_FORMAT_F16,     1,  0.5f, NVCV_BORDER_REFLECT},
    {    322,     49,       2,     NVCV_IMAGE_FORMAT_F16,     3,  0.5f, NVCV_BORDER_REFLECT101},
    {    322,     49,       2,  NVCV_IMAGE_FORMAT_RGBf16,     1,  0.5f, NVCV_BORDER_REPLICATE},
    {    322,     49,       2,  NVCV_IMAGE_FORMAT_RGBf16,     3,  0.5f, NVCV_BORDER_WRAP},
});

// clang-format on

TEST_P(OpLaplacianF16, tensor_correct_output)
{
    ASSERT_NO_FATAL_FAILURE(CheckTensorGoldCase(GetParam()));
}

TEST_P(OpLaplacianF16, varshape_correct_output)
{
    ASSERT_NO_FATAL_FAILURE(CheckVarShapeGoldCase(GetParam()));
}

// Laplacian filters each channel independently, so a planar input is filtered plane-by-plane and
// must produce exactly the same pixels as the interleaved path. These tests feed identical data
// through cvcuda::Laplacian in both layouts and require the re-interleaved planar output to match
// the interleaved output bit-for-bit.
// =============================================================================

namespace {

const nvcv::ImageFormat FMT_RGBU16{nvcv::ColorModel::RGB,    nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                   nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYZ1, nvcv::Packing::X16_Y16_Z16};
const nvcv::ImageFormat FMT_RGBU16p{nvcv::ColorModel::RGB,    nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                    nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYZ0, nvcv::Packing::X16,
                                    nvcv::Packing::X16,       nvcv::Packing::X16};

} // namespace

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
    { 37, 29, 3, 1.5f,    NVCV_BORDER_REFLECT, 2,          FMT_RGBU16p,         FMT_RGBU16},
    // F16 parity is bit-exact like the other dtypes: both layouts run the same float-accumulating
    // taps in the same order, so the single half store rounding matches.
    { 64, 48, 1, 1.0f,   NVCV_BORDER_CONSTANT, 2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
    { 50, 40, 3, 2.0f,  NVCV_BORDER_REPLICATE, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
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

TEST(OpLaplacianPlanar, tensor_planar_tiled_nine_point_partial_block_matches_interleaved)
{
    // The 129x17 tensor crosses both tiled launch boundaries while ksize 3 selects the nine-point stencil.
    constexpr int   ksize = 3;
    constexpr float scale = 1.0f;

    test::planar::RunTensorParity(
        nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 129, 17, 129, 17, 1,
        [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Laplacian op;
            EXPECT_NO_THROW(op(s, src, dst, ksize, scale, NVCV_BORDER_REFLECT));
        });
}

TEST(OpLaplacianPlanar, varshape_different_sizes_match_interleaved)
{
    constexpr int numImages   = 2;
    auto          ksizeTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_S32, 3);
    auto          scaleTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_F32, 1.0f);

    const std::vector<nvcv::Size2D> sizes{
        {129, 33},
        { 17,  7}
    };
    test::planar::RunVarShapeParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, sizes, sizes,
                                    [&ksizeTensor, &scaleTensor](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                                 const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::Laplacian op;
                                        EXPECT_NO_THROW(op(s, src, dst, ksizeTensor, scaleTensor, NVCV_BORDER_REFLECT));
                                    });
}

TEST(OpLaplacianPlanar, covers_u16_nine_point_and_rgb8_tiled_edges)
{
    constexpr int   ksize     = 3;
    constexpr float scale     = 1.0f;
    constexpr int   numImages = 2;

    // U16 VarShape uses the generic correlation path, whose parameter accessor selects the
    // nine-point (false) LaplacianStencil arm.
    auto                            ksizeTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_S32, ksize);
    auto                            scaleTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_F32, scale);
    const std::vector<nvcv::Size2D> u16Sizes{
        {33, 17},
        {17,  9}
    };

    test::planar::RunVarShapeParity(FMT_RGBU16p, FMT_RGBU16, u16Sizes, u16Sizes,
                                    [&ksizeTensor, &scaleTensor](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                                 const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::Laplacian op;
                                        EXPECT_NO_THROW(op(s, src, dst, ksizeTensor, scaleTensor, NVCV_BORDER_REFLECT));
                                    });

    // The 257x33 RGB8 planar tensor crosses both tiled launch boundaries while keeping a valid
    // block for the line-297 guard and selecting the same nine-point stencil.
    test::planar::RunTensorParity(
        nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 257, 33, 257, 33, 1,
        [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Laplacian op;
            EXPECT_NO_THROW(op(s, src, dst, ksize, scale, NVCV_BORDER_REFLECT));
        });
}

TEST(OpLaplacianPlanar, generic_nine_point_and_planar_tiled_guard_false_arm)
{
    constexpr int   ksize = 3;
    constexpr float scale = 1.0f;

    // The U16 generic tensor path evaluates the nine-point stencil through LaplacianStencil's
    // false arm, while the RGB8 planar tensor path enters a valid partial tiled block.
    test::planar::RunTensorSingleChannelLayoutParity(
        130, 17, 130, 17, 1, nvcv::TYPE_U16,
        [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        {
            cvcuda::Laplacian op;
            EXPECT_NO_THROW(op(s, src, dst, ksize, scale, NVCV_BORDER_REFLECT));
        });

    test::planar::RunTensorParity(
        nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 130, 17, 130, 17, 1,
        [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Laplacian op;
            EXPECT_NO_THROW(op(s, src, dst, ksize, scale, NVCV_BORDER_REFLECT));
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
    // unsupported data type (64-bit float; F16 is now valid)
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F64, nvcv::FMT_F64, 3, 0.5, NVCV_BORDER_CONSTANT);
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

// Regression: the var-shape channel guard read `channels > 4 || (isPlanar && channels == 2)`, so an
// interleaved 2-channel batch passed validation -- unlike the Tensor overload and unlike Conv2D,
// which both rejected `channels == 2` outright, and unlike the public header, which documents
// channels [1, 3, 4]. The accepted batch then indexed the [dtype][channels] dispatch table at column
// 1, which is null in every row because no 2-component vector type is ever instantiated, and called
// through that null pointer; the NVCV_ASSERT meant to catch it compiles away in release builds. This
// test submits exactly such a batch, so before the fix it faults instead of reporting
// NVCV_ERROR_INVALID_ARGUMENT.
TEST(OpLaplacianVarshape_Negative, rejects_two_channel_interleaved_batch)
{
    test::planar::ExpectVarShapeUniformFormatRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, 2, 2, NVCV_BORDER_CONSTANT,
                                                      InvokeLaplacianVarShapeNegative);
}

TEST(OpLaplacianVarshape_Negative, rejects_scale_tensor_with_wrong_length)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    for (int i = 0; i < 2; ++i)
    {
        srcImages.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
        dstImages.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
    }

    nvcv::ImageBatchVarShape src(2);
    src.pushBack(srcImages.begin(), srcImages.end());
    nvcv::ImageBatchVarShape dst(2);
    dst.pushBack(dstImages.begin(), dstImages.end());

    nvcv::Tensor ksize({{2}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor scale({{1}, "N"}, nvcv::TYPE_F32);

    cvcuda::Laplacian op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, src, dst, ksize, scale, NVCV_BORDER_CONSTANT); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpLaplacian_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaLaplacianCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}

// Both parameter tensors are read through Tensor1DWrap, which reinterprets rather than converts, so
// the dtype has to be rejected on the host.
TEST(OpLaplacianVarshape_Negative, rejects_mistyped_ksize_tensor)
{
    constexpr int numImages = 2;

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    for (int i = 0; i < numImages; ++i)
    {
        srcImages.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
        dstImages.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
    }
    nvcv::ImageBatchVarShape src(numImages);
    nvcv::ImageBatchVarShape dst(numImages);
    src.pushBack(srcImages.begin(), srcImages.end());
    dst.pushBack(dstImages.begin(), dstImages.end());

    nvcv::Tensor mistypedKsize({{numImages}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor scale({{numImages}, "N"}, nvcv::TYPE_F32);

    cvcuda::Laplacian op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, src, dst, mistypedKsize, scale, NVCV_BORDER_CONSTANT); }));
}

namespace {

enum class AlignmentValues
{
    Byte,
    Half,
    Float
};

struct AlignmentCase
{
    nvcv::ImageFormat format;
    AlignmentValues   values;
    int               rowPadding;
    int               samplePadding;
    int               offsetUnit;
};

void FillAlignmentBytes(std::vector<uint8_t> &host, int seed, AlignmentValues kind)
{
    std::default_random_engine rng(seed);
    if (kind == AlignmentValues::Half)
    {
        test::FillRandomHalfBytes(host, rng, 0.f, 1.f);
    }
    else if (kind == AlignmentValues::Float)
    {
        std::uniform_int_distribution values(-127, 127);
        std::vector<float>            finite(host.size() / sizeof(float));
        std::ranges::generate(finite, [&] { return static_cast<float>(values(rng)) / 64.0f; });
        std::memcpy(host.data(), finite.data(), host.size());
    }
    else
    {
        std::uniform_int_distribution values(0, 255);
        std::ranges::generate(host, [&] { return static_cast<uint8_t>(values(rng)); });
    }
}

void CheckHalfOffsetReference(const std::vector<uint8_t> &actual, std::vector<uint8_t> &reference, int offset)
{
    if (offset == 0)
        reference = actual;
    else
        EXPECT_EQ(actual, reference);
}

void CheckTensorAlignmentOutput(const AlignmentCase &fixture, const std::vector<uint8_t> &inputHost, long3 inputStrides,
                                const nvcv::TensorDataStridedCuda &outputData, long3 outputStrides, int ksize,
                                int offset, std::vector<uint8_t> &reference)
{
    constexpr int        width       = 257;
    constexpr int        height      = 33;
    constexpr int        batches     = 2;
    const size_t         outputBytes = outputStrides.x * batches;
    std::vector<uint8_t> actual(outputBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), outputData.basePtr(), outputBytes, cudaMemcpyDeviceToHost));
    ASSERT_NO_FATAL_FAILURE(CheckLaplacianGold(inputHost, inputStrides, actual, outputStrides,
                                               int3{width, height, batches}, fixture.format, ksize, 0.5f,
                                               NVCV_BORDER_REFLECT101));
    if (fixture.values == AlignmentValues::Half)
    {
        ASSERT_NO_FATAL_FAILURE(CheckHalfOffsetReference(actual, reference, offset));
    }
}

void CheckTensorAlignment(const AlignmentCase &fixture)
{
    constexpr int                       width        = 257;
    constexpr int                       height       = 33;
    constexpr int                       batches      = 2;
    const int                           channels     = fixture.format.numChannels();
    const int                           pixelBytes   = fixture.format.planePixelStrideBytes(0);
    const int                           scalarBytes  = pixelBytes / channels;
    const int64_t                       rowStride    = width * pixelBytes + fixture.rowPadding;
    const int64_t                       sampleStride = rowStride * height + fixture.samplePadding;
    const size_t                        inputBytes   = sampleStride * batches;
    std::vector<uint8_t>                inputHost(inputBytes);
    std::array<std::vector<uint8_t>, 2> referenceOutputs;
    FillAlignmentBytes(inputHost, 0, fixture.values);
    for (int alignment = 0; alignment < 4; ++alignment)
    {
        // Half diagnostics historically report byte offsets; byte/RGBA diagnostics report the sweep index.
        const int offset = fixture.values == AlignmentValues::Half ? alignment * 2 : alignment;
        SCOPED_TRACE(offset);
        std::vector<nvcv::Tensor> tensors;
        for (int endpoint = 0; endpoint < 2; ++endpoint)
        {
            const int baseOffset = ((alignment + endpoint) % 4) * fixture.offsetUnit;
            NVCVByte *allocation{};
            ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&allocation), inputBytes + baseOffset));
            nvcv::TensorDataStridedCuda::Buffer buffer{};
            buffer.basePtr    = allocation + baseOffset;
            buffer.strides[0] = sampleStride;
            buffer.strides[1] = rowStride;
            buffer.strides[2] = pixelBytes;
            buffer.strides[3] = scalarBytes;
            tensors.push_back(nvcv::TensorWrapData(
                nvcv::TensorDataStridedCuda{
                    nvcv::TensorShape{{batches, height, width, channels}, "NHWC"},
                    fixture.values == AlignmentValues::Half ? nvcv::TYPE_F16 : nvcv::TYPE_U8, buffer
            },
                nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                                {
                                                    EXPECT_EQ(cudaSuccess, cudaFree(allocation));
                                                }}));
        }
        const auto &input     = tensors[0];
        const auto &output    = tensors[1];
        auto        inputData = input.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(inputData);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(inputData->basePtr(), inputHost.data(), inputBytes, cudaMemcpyHostToDevice));
        auto outputData = output.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(outputData);
        auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*outputData);
        ASSERT_TRUE(access);
        const long3  outputStrides{access->sampleStride(), access->rowStride(), access->colStride()};
        const size_t outputBytes = outputStrides.x * batches;
        for (int ksize : {1, 3})
        {
            SCOPED_TRACE(ksize);
            ASSERT_EQ(cudaSuccess, cudaMemset(outputData->basePtr(), 0, outputBytes));
            cvcuda::Laplacian op;
            ASSERT_NO_THROW(op(nullptr, input, output, ksize, 0.5f, NVCV_BORDER_REFLECT101));
            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
            ASSERT_NO_FATAL_FAILURE(
                CheckTensorAlignmentOutput(fixture, inputHost, long3{sampleStride, rowStride, pixelBytes}, *outputData,
                                           outputStrides, ksize, offset, referenceOutputs[ksize == 1 ? 0 : 1]));
        }
    }
}

struct ImageGoldCase
{
    AlignmentCase alignment;
    bool          wrapped;
    bool          mixedStencils;
};

using ImageHostInputs     = std::vector<std::vector<std::vector<uint8_t>>>;
using ImageGoldReferences = std::array<std::array<std::vector<uint8_t>, 2>, 2>;

nvcv::Size2D GoldImageSize(const ImageGoldCase &fixture, int image)
{
    return fixture.alignment.format.numPlanes() > 1 && image == 1 ? nvcv::Size2D{322, 49}
                                                                  : nvcv::Size2D{257 + image, 33 + image};
}

void MakeGoldImage(const ImageGoldCase &fixture, int image, int offset, std::vector<nvcv::Image> &inputs,
                   std::vector<std::vector<uint8_t>> &hostInputs)
{
    const auto &f          = fixture.alignment;
    const auto  size       = GoldImageSize(fixture, image);
    const int   planes     = f.format.numPlanes();
    const int   pixelBytes = f.format.planePixelStrideBytes(0);
    const bool  planar     = planes > 1;
    if (!fixture.wrapped)
        inputs.emplace_back(size, f.format);
    else
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes = planes;
        std::vector<NVCVByte *> allocations;
        for (int plane = 0; plane < planes; ++plane)
        {
            const int rowStride  = size.w * pixelBytes + (planar ? (plane + 2) * pixelBytes : f.rowPadding);
            const int baseOffset = ((offset + plane + image) % 4) * f.offsetUnit;
            NVCVByte *allocation{};
            ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&allocation), rowStride * size.h + baseOffset));
            allocations.push_back(allocation);
            buffer.planes[plane] = {size.w, size.h, rowStride, allocation + baseOffset};
        }
        inputs.push_back(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{f.format, buffer},
                                             nvcv::ImageDataCleanupCallback{[allocations](const nvcv::ImageData &)
                                                                            {
                                                                                for (auto allocation : allocations)
                                                                                    EXPECT_EQ(cudaSuccess,
                                                                                              cudaFree(allocation));
                                                                            }}));
    }
    auto data = inputs.back().exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_TRUE(data);
    for (int plane = 0; plane < planes; ++plane)
    {
        auto     &host      = hostInputs[plane];
        const int hostPitch = size.w * pixelBytes + (planar ? 0 : f.rowPadding);
        host.resize(hostPitch * size.h);
        FillAlignmentBytes(host, image * planes + plane, f.values);
        // Interleaved fixtures include random row padding; planar fixtures upload packed logical rows.
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(data->plane(plane).basePtr, data->plane(plane).rowStride, host.data(),
                                            hostPitch, hostPitch, size.h, cudaMemcpyHostToDevice));
    }
}

void CheckImageGold(const ImageGoldCase &fixture, const nvcv::Image &output,
                    const std::vector<std::vector<uint8_t>> &hostInputs, int image, int ksize, NVCVBorderType border,
                    int offset, std::vector<uint8_t> &reference)
{
    SCOPED_TRACE(image);
    const auto &f          = fixture.alignment;
    const auto  size       = GoldImageSize(fixture, image);
    const int   planes     = f.format.numPlanes();
    const int   pixelBytes = f.format.planePixelStrideBytes(0);
    auto        format     = f.format;
    if (planes != 1)
        format = f.values == AlignmentValues::Float ? nvcv::FMT_F32 : nvcv::FMT_U8;
    auto data = output.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_TRUE(data);
    for (int plane = 0; plane < planes; ++plane)
    {
        SCOPED_TRACE(plane);
        const int            rowBytes  = size.w * pixelBytes;
        const auto           hostPitch = static_cast<int>(hostInputs[plane].size() / size.h);
        std::vector<uint8_t> actual(rowBytes * size.h);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(actual.data(), rowBytes, data->plane(plane).basePtr,
                                            data->plane(plane).rowStride, rowBytes, size.h, cudaMemcpyDeviceToHost));
        ASSERT_NO_FATAL_FAILURE(CheckLaplacianGold(hostInputs[plane], long3{hostPitch * size.h, hostPitch, pixelBytes},
                                                   actual, long3{rowBytes * size.h, rowBytes, pixelBytes},
                                                   int3{size.w, size.h, 1}, format, ksize, 0.5f, border));
        if (f.values == AlignmentValues::Half)
        {
            ASSERT_NO_FATAL_FAILURE(CheckHalfOffsetReference(actual, reference, offset));
        }
    }
}

void RunImageGoldStencils(const ImageGoldCase &fixture, const nvcv::ImageBatchVarShape &input,
                          const nvcv::ImageBatchVarShape &output, const std::vector<nvcv::Image> &outputs,
                          const ImageHostInputs &hostInputs, NVCVBorderType border, int offset,
                          ImageGoldReferences &references)
{
    SCOPED_TRACE(border);
    for (int ksize : {1, 3})
    {
        SCOPED_TRACE(ksize);
        auto ksizes = test::planar::MakePerImageTensor(2, nvcv::TYPE_S32, ksize);
        auto scales = test::planar::MakePerImageTensor(2, nvcv::TYPE_F32, 0.5f);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        const std::array imageKsizes{ksize, fixture.mixedStencils ? 4 - ksize : ksize};
        if (fixture.mixedStencils)
        {
            ASSERT_NO_FATAL_FAILURE(
                test::planar::UploadTensorValues(ksizes, std::vector<int>{imageKsizes[0], imageKsizes[1]}));
        }
        // Use a nonzero, finite F16/F32 pattern so stale pixels cannot conceal missed writes.
        for (const auto &image : outputs)
        {
            auto data = image.exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_TRUE(data);
            for (int plane = 0; plane < data->numPlanes(); ++plane)
            {
                const auto &p = data->plane(plane);
                ASSERT_EQ(cudaSuccess,
                          cudaMemset2D(p.basePtr, p.rowStride, 0x5A,
                                       p.width * fixture.alignment.format.planePixelStrideBytes(plane), p.height));
            }
        }
        cvcuda::Laplacian op;
        ASSERT_NO_THROW(op(nullptr, input, output, ksizes, scales, border));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        for (int image = 0; image < 2; ++image)
            ASSERT_NO_FATAL_FAILURE(CheckImageGold(fixture, outputs[image], hostInputs[image], image,
                                                   imageKsizes[image], border, offset,
                                                   references[ksize == 1 ? 0 : 1][image]));
    }
}

void CheckImageGoldAtOffset(const ImageGoldCase &fixture, int offset, ImageGoldReferences &references)
{
    const int                planes = fixture.alignment.format.numPlanes();
    std::vector<nvcv::Image> inputs;
    std::vector<nvcv::Image> outputs;
    ImageHostInputs          hostInputs(2, std::vector<std::vector<uint8_t>>(planes));
    for (int image = 0; image < 2; ++image)
    {
        ASSERT_NO_FATAL_FAILURE(MakeGoldImage(fixture, image, offset, inputs, hostInputs[image]));
        std::vector<std::vector<uint8_t>> initialOutput(planes);
        ASSERT_NO_FATAL_FAILURE(MakeGoldImage(fixture, image, (offset + 1) % 4, outputs, initialOutput));
    }
    nvcv::ImageBatchVarShape input(2);
    nvcv::ImageBatchVarShape output(2);
    input.pushBack(inputs.begin(), inputs.end());
    output.pushBack(outputs.begin(), outputs.end());
    if (fixture.wrapped)
        ASSERT_NO_FATAL_FAILURE(RunImageGoldStencils(fixture, input, output, outputs, hostInputs,
                                                     NVCV_BORDER_REFLECT101, offset, references));
    else
        for (NVCVBorderType border :
             {NVCV_BORDER_REPLICATE, NVCV_BORDER_REFLECT, NVCV_BORDER_WRAP, NVCV_BORDER_REFLECT101})
            ASSERT_NO_FATAL_FAILURE(
                RunImageGoldStencils(fixture, input, output, outputs, hostInputs, border, offset, references));
}

void CheckImageAlignment(const ImageGoldCase &fixture)
{
    ImageGoldReferences references;
    for (int offset = 0; offset < (fixture.wrapped ? 4 : 1); ++offset)
    {
        SCOPED_TRACE(fixture.alignment.values == AlignmentValues::Half ? offset * 2 : offset);
        ASSERT_NO_FATAL_FAILURE(CheckImageGoldAtOffset(fixture, offset, references));
    }
}

} // namespace

// Each row retains its own backend, allocation pitch and four base alignments.
// clang-format off
NVCV_TEST_SUITE_P(OpLaplacianAlignment,
    test::ValueList<nvcv::ImageFormat, bool, int, int, int>{
        // format, VarShape, row padding, sample padding, offset unit
        {nvcv::FMT_U8, false, 2, 3, 1}, {nvcv::FMT_U8, true, 2, 3, 1},
        {nvcv::FMT_RGB8, false, 2, 3, 1}, {nvcv::FMT_RGB8, true, 2, 3, 1},
        {nvcv::FMT_RGBA8, false, 4, 12, 4}, {nvcv::FMT_RGBA8, true, 4, 12, 4},
        {nvcv::FMT_F16, false, 4, 2, 2}, {nvcv::FMT_F16, true, 4, 2, 2},
        {nvcv::FMT_RGBf16, false, 2, 2, 2}, {nvcv::FMT_RGBf16, true, 2, 2, 2},
        {nvcv::FMT_F32, true, 4, 0, 4}
    });
NVCV_TEST_SUITE_P(OpLaplacianPlanarGold,
    test::ValueList<nvcv::ImageFormat, bool, bool>{
        // format, explicit plane alignment, mixed per-image stencils
        {nvcv::FMT_RGB8p, false, false}, {nvcv::FMT_RGB8p, true, false},
        {nvcv::FMT_RGBf32p, false, false}, {nvcv::FMT_RGBAf32p, false, false},
        {nvcv::FMT_RGBf32p, true, false}, {nvcv::FMT_RGB8p, true, true}
    });

// clang-format on

TEST_P(OpLaplacianAlignment, matches_cpu_gold)
{
    const auto format = GetParamValue<0>();
    auto       values = AlignmentValues::Byte;
    if (test::IsF16Format(format))
        values = AlignmentValues::Half;
    else if (format == nvcv::FMT_F32)
        values = AlignmentValues::Float;
    const AlignmentCase fixture{format, values, GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>()};
    if (GetParamValue<1>())
        ASSERT_NO_FATAL_FAILURE(CheckImageAlignment({fixture, true, false}));
    else
        ASSERT_NO_FATAL_FAILURE(CheckTensorAlignment(fixture));
}

TEST_P(OpLaplacianPlanarGold, matches_cpu_gold)
{
    const auto format  = GetParamValue<0>();
    const bool isFloat = format == nvcv::FMT_RGBf32p || format == nvcv::FMT_RGBAf32p;
    ASSERT_NO_FATAL_FAILURE(CheckImageAlignment({
        {format, isFloat ? AlignmentValues::Float : AlignmentValues::Byte, 0, 0, isFloat ? 4 : 1},
        GetParamValue<1>(),
        GetParamValue<2>()
    }));
}

// Larger output, equal batch maxima with swapped pairs, and missing output image.
NVCV_TEST_SUITE_P(OpLaplacianVarshape_SizeMismatch, test::ValueList<int, int, int, int, int>{
                                                        {64, 32, 10, 10, 2},
                                                        {10, 10,  8,  8, 2},
                                                        { 8,  8, 10, 10, 1}
});

TEST_P(OpLaplacianVarshape_SizeMismatch, rejects_unpaired_output_images)
{
    for (auto format : {nvcv::FMT_U8, nvcv::FMT_RGB8p})
    {
        SCOPED_TRACE(format);
        nvcv::ImageBatchVarShape input(2);
        nvcv::ImageBatchVarShape output(2);
        input.pushBack(nvcv::Image(nvcv::Size2D{8, 8}, format));
        input.pushBack(nvcv::Image(nvcv::Size2D{10, 10}, format));
        output.pushBack(nvcv::Image(nvcv::Size2D{GetParamValue<0>(), GetParamValue<1>()}, format));
        if (GetParamValue<4>() == 2)
            output.pushBack(nvcv::Image(nvcv::Size2D{GetParamValue<2>(), GetParamValue<3>()}, format));
        auto ksizes = test::planar::MakePerImageTensor(2, nvcv::TYPE_S32, 1);
        auto scales = test::planar::MakePerImageTensor(2, nvcv::TYPE_F32, 1.f);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        cvcuda::Laplacian op;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { op(nullptr, input, output, ksizes, scales, NVCV_BORDER_CONSTANT); }));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    }
}

TEST(OpLaplacian_Negative, rejects_invalid_ksize_through_c_api)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Laplacian op;
    nvcv::Tensor      input  = nvcv::util::CreateTensor(1, 8, 8, nvcv::FMT_U8);
    nvcv::Tensor      output = nvcv::util::CreateTensor(1, 8, 8, nvcv::FMT_U8);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaLaplacianSubmit(op.handle(), stream, input.handle(), output.handle(),
                                                                 0, 1.0f, NVCV_BORDER_CONSTANT));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpLaplacian_Negative, rejects_empty_tensor_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor      input = nvcv::util::CreateTensor(1, 8, 8, nvcv::FMT_U8);
    nvcv::Tensor      output;
    cvcuda::Laplacian op;

    EXPECT_EQ(NVCV_ERROR_INVALID_OPERATION,
              nvcv::ProtectCall([&] { op(stream, input, output, 3, 1.0f, NVCV_BORDER_CONSTANT); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpLaplacianVarshape_Negative, rejects_null_input_handle_through_c_api)
{
    constexpr int numImages = 1;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Laplacian op;

    std::vector<nvcv::Image> dstImages;
    dstImages.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
    nvcv::ImageBatchVarShape dst(numImages);
    dst.pushBack(dstImages.begin(), dstImages.end());

    nvcv::Tensor ksize({{numImages}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor scale({{numImages}, "N"}, nvcv::TYPE_F32);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaLaplacianVarShapeSubmit(op.handle(), stream, nullptr, dst.handle(), ksize.handle(), scale.handle(),
                                            NVCV_BORDER_CONSTANT));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
