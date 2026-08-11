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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace gt   = ::testing;
namespace test = nvcv::test;

template<typename DT_DEST>
static void setGoldBuffer(std::vector<DT_DEST> &vect, DT_DEST val, int width, int height, int rowStride, int imgStride,
                          int numImages)
{
    for (int img = 0; img < numImages; img++)
    {
        int imgStart = imgStride * img;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                vect[imgStart + x] = val;
            }
            imgStart += rowStride;
        }
    }
}

template<typename DT_SOURCE, typename DT_DEST>
void testConvertTo(nvcv::ImageFormat fmtIn, nvcv::ImageFormat fmtOut, int batch, int width, int height, double alpha,
                   double beta, DT_SOURCE setVal, DT_DEST expVal, NVCVRoundMode roundMode = NVCV_ROUND_NEAREST)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batch, width, height, fmtOut);
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batch, width, height, fmtIn);

    auto inData  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    auto inSampleStride  = static_cast<size_t>(inAccess->numRows() * inAccess->rowStride());
    auto outSampleStride = static_cast<size_t>(outAccess->numRows() * outAccess->rowStride());
    auto numSamples      = static_cast<size_t>(inAccess->numSamples());

    size_t inBufSizeElements  = (inSampleStride / sizeof(DT_SOURCE)) * numSamples;
    size_t outBufSizeElements = (outSampleStride / sizeof(DT_DEST)) * numSamples;
    size_t inBufSizeBytes     = inSampleStride * numSamples;
    size_t outBufSizeBytes    = outSampleStride * numSamples;

    std::vector<DT_SOURCE> srcVec(inBufSizeElements, setVal);
    std::vector<DT_DEST>   goldVec(outBufSizeElements);
    std::vector<DT_DEST>   testVec(outBufSizeElements);

    setGoldBuffer<DT_DEST>(goldVec, expVal, width * outAccess->numChannels(), height,
                           static_cast<int>(outAccess->rowStride() / sizeof(DT_DEST)),
                           static_cast<int>(outSampleStride / sizeof(DT_DEST)), batch);

    // Copy input data to the GPU
    EXPECT_EQ(cudaSuccess,
              cudaMemcpyAsync(inData->basePtr(), srcVec.data(), inBufSizeBytes, cudaMemcpyHostToDevice, stream));
    EXPECT_EQ(cudaSuccess, cudaMemsetAsync(outData->basePtr(), 0x0, outBufSizeBytes, stream));

    // run operator
    cvcuda::ConvertTo convertToOp;

    EXPECT_NO_THROW(convertToOp(stream, imgIn, imgOut, alpha, beta, roundMode));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSizeBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    EXPECT_EQ(goldVec, testVec);
}

// clang-format off
NVCV_TEST_SUITE_P(OpConvertTo, test::ValueList<int, int, double, double, int>
{
         //   width,     height,       alpha,          beta,   batch
         {       5,        5,          1.0,           1.0,       1 },
         {       5,        5,          2.1,           2.0,       5 },
         {       1,        1,          2.1,           -1.0,      1 }
});

// clang-format on

TEST_P(OpConvertTo, OpConvertTo_RGBA8toRGBA8)
{
    using fromType = uint8_t;
    using toType   = uint8_t;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBA8toRGBAf32)
{
    using fromType = uint8_t;
    using toType   = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBA8, nvcv::FMT_RGBAf32, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBAf32toRGBA8)
{
    using fromType = float;
    using toType   = uint8_t;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBAf32, nvcv::FMT_RGBA8, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBAf32toRGBAf32)
{
    using fromType = float;
    using toType   = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf32, batch, width, height, alpha, beta, val,
                                    valExp);
}

TEST_P(OpConvertTo, OpConvertTo_U8toU16)
{
    using fromType = uint8_t;
    using toType   = uint16_t;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_U8, nvcv::FMT_U16, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_2S16to2F32)
{
    using fromType = uint16_t;
    using toType   = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_2S16, nvcv::FMT_2F32, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGB8toRGBf32)
{
    using fromType = uint8_t;
    using toType   = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGB8, nvcv::FMT_RGBf32, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBf32toRGBf32)
{
    using fromType = float;
    using toType   = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBf32, nvcv::FMT_RGBf32, batch, width, height, alpha, beta, val, valExp);
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// ConvertTo applies the same scalar alpha/beta to every element regardless of channel, so a planar
// input is converted plane-by-plane and must produce exactly the same pixels as the interleaved
// path. These tests feed identical data in both layouts through cvcuda::ConvertTo and require the
// (re-interleaved) planar output to match the interleaved output bit-for-bit.
// =============================================================================

namespace {

// Reorder an interleaved (HWC) element vector into planar (CHW) element order. Applied to both the
// input (to seed the planar tensor) and the interleaved output (to compare against the planar
// output), so a single direction suffices.
template<typename DT>
std::vector<DT> InterleavedToPlanar(const std::vector<DT> &hwc, int w, int h, int channels)
{
    std::vector<DT> chw(hwc.size());
    const int       hw = w * h;
    for (int p = 0; p < hw; ++p)
    {
        for (int c = 0; c < channels; ++c)
        {
            chw[c * hw + p] = hwc[p * channels + c];
        }
    }
    return chw;
}

// Deterministic, finite source values valid for every supported input dtype (max 250 fits u8/u16
// and is exactly representable in float; avoids the float NaN/Inf that random bytes would produce).
template<typename DT>
std::vector<DT> MakeDeterministic(int count, int seed)
{
    std::vector<DT> v(count);
    for (int i = 0; i < count; ++i)
    {
        v[i] = static_cast<DT>((i * 7 + seed * 31 + 13) % 251);
    }
    return v;
}

// Convert identical data in interleaved and planar layout, require the outputs to match exactly.
// Seeds/reads the tensors with util::{Set,Get}ImageTensorFromVector, which transparently handle the
// planar layout (interleaved vectors are HWC-ordered, planar vectors CHW-ordered).
template<typename SrcT, typename DstT>
void RunConvertToPlanarParityCase(nvcv::ImageFormat interleavedInFmt, nvcv::ImageFormat planarInFmt,
                                  nvcv::ImageFormat interleavedOutFmt, nvcv::ImageFormat planarOutFmt, int w, int h,
                                  double alpha, double beta, int numImages)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels = planarInFmt.numChannels();

    nvcv::Tensor srcI = nvcv::util::CreateTensor(numImages, w, h, interleavedInFmt);
    nvcv::Tensor dstI = nvcv::util::CreateTensor(numImages, w, h, interleavedOutFmt);
    nvcv::Tensor srcP = nvcv::util::CreateTensor(numImages, w, h, planarInFmt);
    nvcv::Tensor dstP = nvcv::util::CreateTensor(numImages, w, h, planarOutFmt);

    auto srcIData = srcI.exportData<nvcv::TensorDataStridedCuda>();
    auto dstIData = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPData = srcP.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData = dstP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcIData && dstIData && srcPData && dstPData);
    ASSERT_TRUE(srcPData->layout() == nvcv::TENSOR_NCHW || srcPData->layout() == nvcv::TENSOR_CHW);

    for (int i = 0; i < numImages; ++i)
    {
        std::vector<SrcT> hwc = MakeDeterministic<SrcT>(w * h * channels, i);
        std::vector<SrcT> chw = InterleavedToPlanar(hwc, w, h, channels);
        nvcv::util::SetImageTensorFromVector<SrcT>(*srcIData, hwc, i);
        nvcv::util::SetImageTensorFromVector<SrcT>(*srcPData, chw, i);
    }

    cvcuda::ConvertTo op;
    EXPECT_NO_THROW(op(stream, srcI, dstI, alpha, beta));
    EXPECT_NO_THROW(op(stream, srcP, dstP, alpha, beta));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<DstT> outI;
        std::vector<DstT> outP;
        nvcv::util::GetImageVectorFromTensor<DstT>(*dstIData, i, outI);
        nvcv::util::GetImageVectorFromTensor<DstT>(*dstPData, i, outP);
        EXPECT_EQ(InterleavedToPlanar(outI, w, h, channels), outP);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

TEST(OpConvertToPlanar, rgb8_to_rgbf32_matches_interleaved)
{
    RunConvertToPlanarParityCase<uint8_t, float>(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p,
                                                 33, 17, 2.1, 2.0, 1);
}

TEST(OpConvertToPlanar, rgba8_to_rgba8_matches_interleaved)
{
    RunConvertToPlanarParityCase<uint8_t, uint8_t>(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p,
                                                   16, 16, 1.0, 5.0, 1);
}

TEST(OpConvertToPlanar, rgbf32_to_rgb8_matches_interleaved)
{
    RunConvertToPlanarParityCase<float, uint8_t>(nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p, nvcv::FMT_RGB8, nvcv::FMT_RGB8p,
                                                 20, 12, 0.5, 1.0, 1);
}

TEST(OpConvertToPlanar, rgb8_to_rgbf32_batched_matches_interleaved)
{
    RunConvertToPlanarParityCase<uint8_t, float>(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p,
                                                 24, 24, 1.5, -3.0, 4);
}

// Rounding mode: NEAREST (historical default) vs TRUNCATE (toward zero), affecting float-to-integer
// outputs only. Cases pin values whose two roundings differ, incl. a negative, and check the default.

TEST(OpConvertTo_Round, f32_to_s32_positive)
{
    // 2.7 -> nearest 3, truncate 2.
    testConvertTo<float, int32_t>(nvcv::FMT_F32, nvcv::FMT_S32, 1, 5, 5, 1.0, 0.0, 2.7f, 3); // default == NEAREST
    testConvertTo<float, int32_t>(nvcv::FMT_F32, nvcv::FMT_S32, 1, 5, 5, 1.0, 0.0, 2.7f, 3, NVCV_ROUND_NEAREST);
    testConvertTo<float, int32_t>(nvcv::FMT_F32, nvcv::FMT_S32, 1, 5, 5, 1.0, 0.0, 2.7f, 2, NVCV_ROUND_TRUNCATE);
}

TEST(OpConvertTo_Round, f32_to_s32_negative)
{
    // -2.7 -> nearest -3, truncate (toward zero) -2.
    testConvertTo<float, int32_t>(nvcv::FMT_F32, nvcv::FMT_S32, 1, 5, 5, 1.0, 0.0, -2.7f, -3, NVCV_ROUND_NEAREST);
    testConvertTo<float, int32_t>(nvcv::FMT_F32, nvcv::FMT_S32, 1, 5, 5, 1.0, 0.0, -2.7f, -2, NVCV_ROUND_TRUNCATE);
}

TEST(OpConvertTo_Round, float_output_unaffected)
{
    // Float output: the rounding mode is meaningless; TRUNCATE must not alter the fractional result.
    testConvertTo<float, float>(nvcv::FMT_F32, nvcv::FMT_F32, 1, 5, 5, 1.0, 0.0, 2.7f, 2.7f, NVCV_ROUND_TRUNCATE);
}

// clang-format off

NVCV_TEST_SUITE_P(OpConvertTo_Negative, nvcv::test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, int, int, int, int, int, int>{
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8, 24, 24, 24, 24, 3, 3}, // mismatched layout (planar in, interleaved out)
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F32, 24, 24, 24, 24, 3, 3}, // invalid input data type
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F32, nvcv::FMT_F16, 24, 24, 24, 24, 3, 3}, // invalid output data type
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F32, nvcv::FMT_F32, 25, 24, 24, 24, 3, 3}, // width is different
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F32, nvcv::FMT_F32, 24, 25, 24, 24, 3, 3}, // height is different
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F32, nvcv::FMT_F32, 24, 24, 24, 24, 4, 3}, // batch number is different
});

// clang-format on

TEST_P(OpConvertTo_Negative, op)
{
    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat outputFmt          = GetParamValue<2>();
    int               inputWidth         = GetParamValue<3>();
    int               inputHeight        = GetParamValue<4>();
    int               outputWidth        = GetParamValue<5>();
    int               outputHeight       = GetParamValue<6>();
    int               inputBatch         = GetParamValue<7>();
    int               outputBatch        = GetParamValue<8>();

    double alpha = 1.0;
    double beta  = 0.0;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut = nvcv::util::CreateTensor(outputBatch, outputWidth, outputHeight, outputFmt);
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inputBatch, inputWidth, inputHeight, inputFmt);

    // run operator
    cvcuda::ConvertTo convertToOp;
    EXPECT_EQ(expectedReturnCode, nvcv::ProtectCall([&convertToOp, &stream, &imgIn, &imgOut, &alpha, &beta]
                                                    { convertToOp(stream, imgIn, imgOut, alpha, beta); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpConvertTo_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaConvertToCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
