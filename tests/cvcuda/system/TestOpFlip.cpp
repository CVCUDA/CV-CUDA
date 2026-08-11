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

#include "Definitions.hpp"
#include "FlipUtils.hpp"
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpFlip.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>
#include <vector>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

// clang-format off

NVCV_TEST_SUITE_P(OpFlip, test::ValueList<int, int, int, NVCVImageFormat, int>
{
    // width, height, batches,                  format, flipCode
    {    176,    113,       1,    NVCV_IMAGE_FORMAT_U8,  0},
    {    123,     66,       5,    NVCV_IMAGE_FORMAT_U8,  1},
    {    123,     33,       3,  NVCV_IMAGE_FORMAT_RGB8, -1},
    {     42,     53,       4, NVCV_IMAGE_FORMAT_RGBA8,  1},
    {     13,     42,       3,  NVCV_IMAGE_FORMAT_RGB8,  0},
    {     62,    111,       4, NVCV_IMAGE_FORMAT_RGBA8, -1},
    // Float3 is intentionally routed through conservative kernels. Cover every flip direction for
    // both tensor and var-shape submissions against the independent CPU reference.
    {     67,     45,       2, NVCV_IMAGE_FORMAT_RGBf32,  1},
    {     70,     43,       3, NVCV_IMAGE_FORMAT_RGBf32,  0},
    {     65,     41,       2, NVCV_IMAGE_FORMAT_RGBf32, -1},
    // Single-channel cases that exercise the wide (VEC=4) vectorized path and its in-register lane
    // reversal: width divisible by 4 (vector body) for each flip code, plus a non-divisible width that
    // falls back to the scalar single-channel tail. U8 and U16 cover both VEC lane widths.
    {    256,     65,       2,    NVCV_IMAGE_FORMAT_U8,  1}, // wide horizontal (lane reversal)
    {    256,     64,       2,    NVCV_IMAGE_FORMAT_U8, -1}, // wide both (lane reversal)
    {    256,     48,       3,    NVCV_IMAGE_FORMAT_U8,  0}, // wide vertical (direct copy)
    {    128,     40,       2,   NVCV_IMAGE_FORMAT_U16, -1}, // wide both, 16-bit lanes
    {    255,     40,       2,    NVCV_IMAGE_FORMAT_U8, -1}  // scalar tail (width % 4 != 0)
});

// clang-format on

TEST_P(OpFlip, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int flipCode = GetParamValue<4>();

    int3 shape{width, height, batches};

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    auto input  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto output = outTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*output);
    ASSERT_TRUE(outAccess);

    long inSampleStride  = inAccess->numRows() * inAccess->rowStride();
    long outSampleStride = outAccess->numRows() * outAccess->rowStride();

    auto inBufSize  = static_cast<size_t>(inSampleStride * inAccess->numSamples());
    auto outBufSize = static_cast<size_t>(outSampleStride * outAccess->numSamples());

    long3 inStrides{inSampleStride, inAccess->rowStride(), inAccess->colStride()};
    long3 outStrides{outSampleStride, outAccess->rowStride(), outAccess->colStride()};

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });
    std::vector<uint8_t> goldVec(outBufSize);
    test::FlipCPU(goldVec, outStrides, inVec, inStrides, shape, format, flipCode);

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Flip flipOp;
    EXPECT_NO_THROW(flipOp(stream, inTensor, outTensor, flipCode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // copy output back to host
    std::vector<uint8_t> testVec(outBufSize);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(testVec, goldVec);
}

TEST_P(OpFlip, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int flipCode = GetParamValue<4>();

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

    // Create flip code tensor
    nvcv::Tensor flip_code({{batches}, "N"}, nvcv::TYPE_S32);
    {
        auto dev = flip_code.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int> vec(batches, flipCode);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::Flip flipOp(batches);

    EXPECT_NO_THROW(flipOp(stream, batchSrc, batchDst, flip_code));

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
        std::vector<uint8_t> goldVec(shape.y * pitches.y);
        test::FlipCPU(goldVec, pitches, srcVec[i], pitches, shape, format, flipCode);

        EXPECT_EQ(testVec, goldVec);
    }
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// Flip remaps each pixel independently of its channel, so a planar input is flipped plane-by-plane
// and must produce exactly the same pixels as the interleaved path. These tests feed identical data
// in both layouts through cvcuda::Flip and require the (re-interleaved) planar output to match the
// interleaved output bit-for-bit, for every dtype and flip code.
// =============================================================================

namespace {

// Flip identical data in interleaved and planar tensor layout; outputs must match bit-for-bit.
// The shared scaffolding (upload/run/download/compare) lives in PlanarParityUtils.hpp; here we only
// bind the Flip call.
void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                               int flipCode, int numImages)
{
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, w, h, w, h, numImages,
        [numImages, flipCode](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Flip op(numImages);
            EXPECT_NO_THROW(op(s, src, dst, flipCode));
        });
}

// Var-shape counterpart of RunPlanarParityTensorCase.
void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                 int flipCode, int numImages)
{
    // Var-shape Flip takes the flip code as a per-image tensor; upload it once (synchronously, so it
    // is ready before the operator runs on the parity helper's stream).
    nvcv::Tensor flip_code({{numImages}, "N"}, nvcv::TYPE_S32);
    {
        auto dev = flip_code.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);
        std::vector<int> vec(numImages, flipCode);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(dev->basePtr(), vec.data(), vec.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    test::planar::RunVarShapeParity(planarFmt, interleavedFmt, w, h, w, h, numImages,
                                    [numImages, &flip_code](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                            const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::Flip op(numImages);
                                        EXPECT_NO_THROW(op(s, src, dst, flip_code));
                                    });
}

} // namespace

// Parameters: width, height, flipCode, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpFlipPlanar,
                  test::ValueList<int, int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113,  1, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, horizontal
    {123,  66,  0, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, vertical
    { 64,  48, -1, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, both
    { 50,  40,  1, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8, horizontal
    {100,  80, -1, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8, both
    { 64,  48,  0, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}, // float planar
    { 67,  45,  1, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32}, // RGB float3, horizontal
    { 70,  43,  0, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32}, // RGB float3, vertical
    { 65,  41, -1, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32}, // RGB float3, both
});

// clang-format on

TEST_P(OpFlipPlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>());
}

TEST_P(OpFlipPlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>(), GetParamValue<3>());
}

// clang-format off
NVCV_TEST_SUITE_P(OpFlip_Negative, nvcv::test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_RGB8, nvcv::FMT_RGBf32},  // data type is different
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p},  // data format is different (interleaved in, planar out)
    {nvcv::FMT_2S16, nvcv::FMT_2S16},  // unsupported two-channel format
    {nvcv::FMT_F16, nvcv::FMT_F16},  // invalid data type,
});

// clang-format on

TEST_P(OpFlip_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inputFmt  = GetParamValue<0>();
    nvcv::ImageFormat outputFmt = GetParamValue<1>();
    int               flipCode  = 0;

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(2, 24, 24, inputFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(2, 24, 24, outputFmt);

    // run operator
    cvcuda::Flip flipOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &inTensor, &outTensor, &flipCode]
                                                             { flipOp(stream, inTensor, outTensor, flipCode); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpFlip_Negative, varshape_op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inputFmt  = GetParamValue<0>();
    nvcv::ImageFormat outputFmt = GetParamValue<1>();

    int width    = 24;
    int height   = 24;
    int batches  = 3;
    int flipCode = 0;

    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inputFmt);
    }
    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), outputFmt);
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create flip code tensor
    nvcv::Tensor flip_code({{batches}, "N"}, nvcv::TYPE_S32);
    {
        auto dev = flip_code.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int> vec(batches, flipCode);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::Flip flipOp(batches);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &batchSrc, &batchDst, &flip_code]
                                                             { flipOp(stream, batchSrc, batchDst, flip_code); }));
}

TEST(OpFlip_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt      = nvcv::FMT_RGB8;
    int               flipCode = 0;
    int               width    = 24;
    int               height   = 24;
    int               batches  = 3;

    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_U8,          fmt},
        {         fmt, nvcv::FMT_U8}
    };

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        std::vector<nvcv::Image> imgSrc;
        for (int i = 0; i < batches - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        nvcv::ImageBatchVarShape batchSrc(batches);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        // Create output varshape
        std::vector<nvcv::Image> imgDst;
        for (int i = 0; i < batches - 1; ++i)
        {
            imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
        }
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);
        nvcv::ImageBatchVarShape batchDst(batches);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Create flip code tensor
        nvcv::Tensor flip_code({{batches}, "N"}, nvcv::TYPE_S32);
        {
            auto dev = flip_code.exportData<nvcv::TensorDataStridedCuda>();
            ASSERT_NE(dev, nullptr);

            std::vector<int> vec(batches, flipCode);

            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice, stream));
        }

        // Run operator
        cvcuda::Flip flipOp(batches);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &batchSrc, &batchDst, &flip_code]
                                                                 { flipOp(stream, batchSrc, batchDst, flip_code); }));
    }
}

TEST(OpFlip_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaFlipCreate(nullptr, 2), NVCV_ERROR_INVALID_ARGUMENT);
}
