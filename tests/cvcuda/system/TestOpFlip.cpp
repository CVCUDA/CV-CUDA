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
    {    255,     40,       2,    NVCV_IMAGE_FORMAT_U8, -1}, // scalar tail (width % 4 != 0)
    // F16 routes through the 16-bit integer kernels (pure data movement, bit-exact); cover C1
    // wide + scalar-tail paths and the C3/C4 interleaved kernels for each flip direction.
    {    128,     40,       2,    NVCV_IMAGE_FORMAT_F16, -1}, // wide both, 16-bit lanes
    {    255,     40,       2,    NVCV_IMAGE_FORMAT_F16,  0}, // scalar tail (width % 4 != 0)
    {     67,     45,       2, NVCV_IMAGE_FORMAT_RGBf16,  1},
    {     70,     43,       3, NVCV_IMAGE_FORMAT_RGBf16,  0},
    {     62,    111,       4, NVCV_IMAGE_FORMAT_RGBAf16, -1},
    // F32 and S32 single channel had no independent-reference coverage at all, even though both
    // route through the wide vectorized single-channel kernel. Cover the vector body for every
    // flip direction plus the scalar tail.
    {     64,     33,       2,   NVCV_IMAGE_FORMAT_F32,  1},
    {     64,     32,       2,   NVCV_IMAGE_FORMAT_F32,  0},
    {     68,     31,       3,   NVCV_IMAGE_FORMAT_F32, -1},
    {     63,     30,       2,   NVCV_IMAGE_FORMAT_F32, -1},
    {     64,     29,       2,   NVCV_IMAGE_FORMAT_S32,  1},
    {     67,     28,       2,   NVCV_IMAGE_FORMAT_S32,  0},
    // RGBAf32 (the 128-bit interleaved kernel) likewise had no independent-reference coverage.
    {     37,     27,       2, NVCV_IMAGE_FORMAT_RGBAf32,  1},
    {     36,     26,       3, NVCV_IMAGE_FORMAT_RGBAf32,  0},
    {     35,     25,       2, NVCV_IMAGE_FORMAT_RGBAf32, -1},
    // A 128-bit access covers 16 elements of an 8-bit type and 8 of a 16-bit type, so widths that
    // are multiples of 4 but not of 8, and of 8 but not of 16, are the boundaries where a widened
    // vector body can silently drop or duplicate its tail. The pre-existing single-channel widths
    // were all multiples of 16, which cannot separate those cases.
    {     40,     24,       2,    NVCV_IMAGE_FORMAT_U8,  1},
    {     68,     23,       3,    NVCV_IMAGE_FORMAT_U8, -1},
    {     72,     22,       2,    NVCV_IMAGE_FORMAT_U8,  0},
    {     68,     21,       2,   NVCV_IMAGE_FORMAT_U16,  1},
    {     72,     20,       2,   NVCV_IMAGE_FORMAT_U16, -1},
    {     68,     19,       2,   NVCV_IMAGE_FORMAT_F16, -1},
    // Interleaved C3/C4 at widths that are exact multiples of the pixel group a 128-bit access
    // spans (16 pixels for 8-bit, 8 for 16-bit, 4 for 32-bit) and at widths that are not, so both
    // a vectorized body and its scalar tail are checked in every direction.
    {     64,     18,       2,  NVCV_IMAGE_FORMAT_RGB8,  1},
    {     48,     17,       3,  NVCV_IMAGE_FORMAT_RGB8,  0},
    {     33,     16,       2,  NVCV_IMAGE_FORMAT_RGB8, -1},
    {     64,     15,       2, NVCV_IMAGE_FORMAT_RGBA8,  0},
    {     36,     14,       3, NVCV_IMAGE_FORMAT_RGBA8, -1},
    {     48,     13,       2, NVCV_IMAGE_FORMAT_RGBf32,  1},
    {     36,     12,       2, NVCV_IMAGE_FORMAT_RGBf32,  0},
    {     48,     11,       2, NVCV_IMAGE_FORMAT_RGBf16, -1},
    {     35,     10,       2, NVCV_IMAGE_FORMAT_RGBAf16,  1}
});

// Degenerate extents are tensor-only: the var-shape suite jitters each image to 0.8-1.1x the
// listed size, which cannot represent a 1-pixel extent. A single row makes vertical flip the
// identity and a single column does the same for horizontal flip, and both collapse a
// vectorized launch to one partial group.
NVCV_TEST_SUITE_P(OpFlipDegenerate, test::ValueList<int, int, int, NVCVImageFormat, int>
{
    // width, height, batches,                  format, flipCode
    {      1,     37,       2,    NVCV_IMAGE_FORMAT_U8,  1},
    {      1,     37,       2,    NVCV_IMAGE_FORMAT_U8,  0},
    {      1,     37,       2,    NVCV_IMAGE_FORMAT_U8, -1},
    {     37,      1,       2,    NVCV_IMAGE_FORMAT_U8,  1},
    {     37,      1,       2,    NVCV_IMAGE_FORMAT_U8,  0},
    {     37,      1,       2,    NVCV_IMAGE_FORMAT_U8, -1},
    {      1,      1,       1,    NVCV_IMAGE_FORMAT_U8, -1},
    {      1,     16,       2,  NVCV_IMAGE_FORMAT_RGB8, -1},
    {     16,      1,       2,  NVCV_IMAGE_FORMAT_RGB8,  1},
    {      1,     16,       2, NVCV_IMAGE_FORMAT_RGBA8,  1},
    {      1,     16,       2, NVCV_IMAGE_FORMAT_RGBf32, -1},
    {     16,      1,       2,   NVCV_IMAGE_FORMAT_F32,  1},
    {      1,     16,       2,   NVCV_IMAGE_FORMAT_F32, -1},
    {      1,     16,       2,   NVCV_IMAGE_FORMAT_F16,  0}
});

// clang-format on

// Bit-exact comparison of one tensor submission against the independent host reference. Shared by
// the main and the degenerate-extent suites so both check values the same way.
static void ExpectTensorFlipMatchesReference(int width, int height, int batches, nvcv::ImageFormat format, int flipCode)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

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

TEST_P(OpFlip, correct_output)
{
    ExpectTensorFlipMatchesReference(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                     nvcv::ImageFormat{GetParamValue<3>()}, GetParamValue<4>());
}

TEST_P(OpFlipDegenerate, correct_output)
{
    ExpectTensorFlipMatchesReference(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                     nvcv::ImageFormat{GetParamValue<3>()}, GetParamValue<4>());
}

NVCV_TEST_SUITE_P(OpFlipRGB8Carrier, test::ValueList<int, int, int, bool>{
                                         {689, 0, 0,  true},
                                         {689, 5, 0,  true},
                                         {689, 0, 1,  true},
                                         { 17, 0, 0,  true},
                                         {  5, 0, 0,  true},
                                         {  1, 5, 0,  true},
                                         {689, 0, 0, false},
                                         {  1, 5, 0, false}
});

TEST_P(OpFlipRGB8Carrier, tensor_matches_reference_and_preserves_padding)
{
    const int     width              = GetParamValue<0>();
    const int     baseOffset         = GetParamValue<1>();
    const int     strideOffset       = GetParamValue<2>();
    const bool    batched            = GetParamValue<3>();
    const int     samples            = batched ? 3 : 1;
    constexpr int height             = 7;
    const int     rowBytes           = width * 3;
    const int     inputRowStride     = rowBytes + 7;
    const int     outputRowStride    = ((rowBytes + 15) / 16 + 1) * 16 + strideOffset;
    const int     inputSampleStride  = inputRowStride * height + 13;
    const int     outputSampleStride = outputRowStride * height + 32;
    const int     inputBytes         = baseOffset + inputSampleStride * samples;
    const int     outputBytes        = baseOffset + outputSampleStride * samples;

    auto makeTensor = [=](NVCVByte *allocation, int rowStride, int sampleStride)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr                  = allocation + baseOffset;
        buffer.strides[0]               = sampleStride;
        buffer.strides[batched ? 1 : 0] = rowStride;
        buffer.strides[batched ? 2 : 1] = 3;
        buffer.strides[batched ? 3 : 2] = 1;
        const auto tensorShape = batched ? nvcv::TensorShape{{samples, height, width, 3}, "NHWC"}
                                         : nvcv::TensorShape{{height, width, 3}, "HWC"};
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{tensorShape, nvcv::TYPE_U8, buffer},
                                    nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                                                    {
                                                                        EXPECT_EQ(cudaSuccess, cudaFree(allocation));
                                                                    }});
    };

    NVCVByte *inputAllocation{};
    NVCVByte *outputAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&inputAllocation), inputBytes));
    nvcv::Tensor input = makeTensor(inputAllocation, inputRowStride, inputSampleStride);
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&outputAllocation), outputBytes));
    nvcv::Tensor output = makeTensor(outputAllocation, outputRowStride, outputSampleStride);

    std::vector<uint8_t> inputHost(inputBytes);
    for (size_t index = 0; index < inputHost.size(); ++index)
    {
        inputHost[index] = static_cast<uint8_t>((index * 37 + index / 251) % 256);
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputAllocation, inputHost.data(), inputBytes, cudaMemcpyHostToDevice));

    std::vector<uint8_t> inputPixels(inputHost.begin() + baseOffset, inputHost.end());
    const long3          inputStrides{inputSampleStride, inputRowStride, 3};
    const long3          outputStrides{outputSampleStride, outputRowStride, 3};
    const int3           shape{width, height, samples};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::Flip flipOp;
    for (int flipCode : {1, -1, 0})
    {
        SCOPED_TRACE(flipCode);
        std::vector<uint8_t> expectedPixels(outputBytes - baseOffset, 0xA5);
        test::FlipCPU(expectedPixels, outputStrides, inputPixels, inputStrides, shape, nvcv::FMT_RGB8, flipCode);
        std::vector<uint8_t> expected(baseOffset, 0xA5);
        expected.insert(expected.end(), expectedPixels.begin(), expectedPixels.end());

        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(outputAllocation, 0xA5, outputBytes, stream));
        flipOp(stream, input, output, flipCode);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        std::vector<uint8_t> actual(outputBytes);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), outputAllocation, outputBytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(actual, expected);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> inputAfter(inputBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputAfter.data(), inputAllocation, inputBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(inputAfter, inputHost);
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

TEST(OpFlip, varshape_s16_planar_interleaved_parity)
{
    const nvcv::ImageFormat interleavedFmt{nvcv::ColorModel::RGB,  nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                           nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZ1, nvcv::Packing::X16_Y16_Z16};
    const nvcv::ImageFormat planarFmt{nvcv::ColorModel::RGB,  nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                      nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZ0, nvcv::Packing::X16,
                                      nvcv::Packing::X16,     nvcv::Packing::X16};

    constexpr int numImages = 2;
    constexpr int flipCode  = -1;
    nvcv::Tensor  flip_code = test::planar::MakePerImageTensor<int>(numImages, nvcv::TYPE_S32, flipCode);

    test::planar::RunVarShapeParity(planarFmt, interleavedFmt, 37, 29, 37, 29, numImages,
                                    [&flip_code](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                 const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::Flip op(numImages);
                                        EXPECT_NO_THROW(op(stream, src, dst, flip_code));
                                    });
}

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
    { 67,  45,  1, 2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16}, // RGB f16, horizontal
    { 70,  43,  0, 2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16}, // RGB f16, vertical
    { 50,  40, -1, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16}, // RGBA f16, both
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

TEST(OpFlip, tensor_s32_padded_row_scalar_fallback)
{
    constexpr int samples         = 2;
    constexpr int width           = 8;
    constexpr int height          = 3;
    constexpr int rowElements     = width + 1;
    constexpr int sampleElements  = rowElements * height + 1;
    constexpr int allocationBytes = sampleElements * samples * sizeof(int);
    constexpr int flipCode        = -1;

    auto makeTensor = [=](NVCVByte *allocation)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = allocation;
        buffer.strides[0] = (sampleElements * sizeof(int));
        buffer.strides[1] = (rowElements * sizeof(int));
        buffer.strides[2] = sizeof(int);
        buffer.strides[3] = sizeof(int);
        return nvcv::TensorWrapData(
            nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{samples, height, width, 1}, "NHWC"},
                nvcv::TYPE_S32, buffer
        },
            nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                            {
                                                EXPECT_EQ(cudaSuccess, cudaFree(allocation));
                                            }});
    };

    NVCVByte *inputAllocation{};
    NVCVByte *outputAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&inputAllocation), allocationBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&outputAllocation), allocationBytes));

    nvcv::Tensor input  = makeTensor(inputAllocation);
    nvcv::Tensor output = makeTensor(outputAllocation);

    std::vector<int> inputHost(sampleElements * samples);
    for (int b = 0; b < samples; ++b)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                inputHost[b * sampleElements + y * rowElements + x] = 1000 * b + 100 * y + x;
            }
        }
    }
    std::vector<int> outputHost(inputHost.size(), -1);

    auto inputData  = input.exportData<nvcv::TensorDataStridedCuda>();
    auto outputData = output.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inputData && outputData);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputData->basePtr(), inputHost.data(), allocationBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(outputData->basePtr(), outputHost.data(), allocationBytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Flip flipOp;
    EXPECT_NO_THROW(flipOp(stream, input, output, flipCode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(outputHost.data(), outputData->basePtr(), allocationBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<int> expected(outputHost.size(), -1);
    for (int b = 0; b < samples; ++b)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                expected[b * sampleElements + y * rowElements + x]
                    = inputHost[b * sampleElements + (height - 1 - y) * rowElements + (width - 1 - x)];
            }
        }
    }

    EXPECT_EQ(outputHost, expected);
}

// clang-format off
NVCV_TEST_SUITE_P(OpFlip_Negative, nvcv::test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_RGB8, nvcv::FMT_RGBf32},  // data type is different
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p},  // data format is different (interleaved in, planar out)
    {nvcv::FMT_2S16, nvcv::FMT_2S16},  // unsupported two-channel format
    {nvcv::FMT_F64, nvcv::FMT_F64},  // unsupported data type (64-bit float)
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

// Flip's public contract (OpFlip.h, "Input/Output dependency") requires the output to agree with
// the input on Number, Channels, Width and Height, but nothing enforced it: not the legacy
// implementation and not its first non-legacy port. Both derived the launch geometry and the pixel
// type from a single side and then addressed the other with it, so a disagreement addressed outside
// an allocation instead of being rejected:
//   * the kernels compute the *source* coordinate from the *destination* geometry,
//     src(H-1-y, W-1-x) with H, W and the sample count taken from the output, and bounds-check
//     against the output only -- a larger output reads past the end of the input;
//   * the pixel type comes from the *input* channel count -- a narrower output is written with the
//     input's wider pixel type over the output's narrower column stride, past the end of the output.
// Either way the caller saw NVCV_SUCCESS and a silently corrupt result.
TEST(OpFlip_Negative, tensor_output_shape_must_match_input)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt      = nvcv::FMT_RGB8;
    int               flipCode = -1;

    // inBatches, inW, inH, inFmt, outBatches, outW, outH, outFmt
    std::vector<std::tuple<int, int, int, nvcv::ImageFormat, int, int, int, nvcv::ImageFormat>> testSet{
        {2, 24, 24, fmt, 2, 32, 24,          fmt}, // wider output: reads past the input rows
        {2, 24, 24, fmt, 2, 24, 32,          fmt}, // taller output: reads past the input image
        {2, 24, 24, fmt, 3, 24, 24,          fmt}, // extra sample: reads past the input batch
        {2, 24, 24, fmt, 2, 24, 24, nvcv::FMT_U8}  // narrower output: writes past the output rows
    };

    int caseIdx = 0;
    for (const auto &[inN, inW, inH, inFmt, outN, outW, outH, outFmt] : testSet)
    {
        SCOPED_TRACE(caseIdx++);

        nvcv::Tensor inTensor  = nvcv::util::CreateTensor(inN, inW, inH, inFmt);
        nvcv::Tensor outTensor = nvcv::util::CreateTensor(outN, outW, outH, outFmt);

        cvcuda::Flip flipOp;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &inTensor, &outTensor, &flipCode]
                                                                 { flipOp(stream, inTensor, outTensor, flipCode); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Var-shape counterpart of tensor_output_shape_must_match_input. Here the geometry is per image, so
// a batch whose formats and image count agree can still pair a small input image with a larger
// output one; the kernel reads src(dstH-1-y, dstW-1-x) and runs off the end of that one input
// image. The batch below is uniform except for its last pair, so the mismatch is genuinely
// per-image and cannot be caught by the existing unique-format check.
TEST(OpFlip_Negative, varshape_output_size_must_match_input)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt      = nvcv::FMT_RGB8;
    int               flipCode = -1;
    int               batches  = 3;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches - 1; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{24, 24}, fmt);
        imgDst.emplace_back(imgSrc[i].size(), fmt);
    }
    imgSrc.emplace_back(nvcv::Size2D{24, 24}, fmt);
    imgDst.emplace_back(nvcv::Size2D{32, 32}, fmt);

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor flip_code = test::planar::MakePerImageTensor<int>(batches, nvcv::TYPE_S32, flipCode);

    cvcuda::Flip flipOp(batches);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &batchSrc, &batchDst, &flip_code]
                                                             { flipOp(stream, batchSrc, batchDst, flip_code); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpFlip_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaFlipCreate(nullptr, 2), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpFlip_Negative, tensor_rejects_unsupported_layout)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor input(
        {
            {2, 24, 24, 3},
            "ABCD"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor output(input.shape(), input.dtype());
    int          flipCode = 0;

    cvcuda::Flip flipOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &input, &output, &flipCode]
                                                             { flipOp(stream, input, output, flipCode); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// The interleaved tensor kernel puts numSamples in grid.z. A batch above the CUDA limit must
// fail with INVALID_ARGUMENT rather than a device-side CUDA configuration error.
TEST(OpFlip_Negative, tensor_numSamples_exceeds_grid_z_limit)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // 65536 samples of 1×1 U8 — minimal allocation, purely exercises the host-side guard.
    constexpr int kLimit    = 65536;
    nvcv::Tensor  inTensor  = nvcv::util::CreateTensor(kLimit, 1, 1, nvcv::FMT_U8);
    nvcv::Tensor  outTensor = nvcv::util::CreateTensor(kLimit, 1, 1, nvcv::FMT_U8);
    int           flipCode  = 0;

    cvcuda::Flip flipOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &inTensor, &outTensor, &flipCode]
                                                             { flipOp(stream, inTensor, outTensor, flipCode); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// flipCode must be an int32 tensor; other element types must be rejected before the
// Tensor1DWrap<int> cast misreads device memory.
TEST(OpFlip_Negative, varshape_flipcode_dtype_must_be_s32)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int batches = 2;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
        imgDst.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
    }
    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Use F32 instead of S32 — should be rejected.
    nvcv::Tensor flip_code({{batches}, "N"}, nvcv::TYPE_F32);

    cvcuda::Flip flipOp(batches);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &batchSrc, &batchDst, &flip_code]
                                                             { flipOp(stream, batchSrc, batchDst, flip_code); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// flipCode must have at least numImages elements; a shorter tensor must be rejected before
// the Tensor1DWrap<int> reads out of bounds on the device.
TEST(OpFlip_Negative, varshape_flipcode_too_short)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int batches  = 3;
    constexpr int flipCode = 0;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
        imgDst.emplace_back(nvcv::Size2D{8, 8}, nvcv::FMT_U8);
    }
    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Provide only 1 element for a batch of 3 images.
    nvcv::Tensor flip_code = test::planar::MakePerImageTensor<int>(1, nvcv::TYPE_S32, flipCode);

    cvcuda::Flip flipOp(batches);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&flipOp, &stream, &batchSrc, &batchDst, &flip_code]
                                                             { flipOp(stream, batchSrc, batchDst, flip_code); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
