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
#include "HalfTestUtils.hpp"

#include <common/DeviceBuffer.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cuda_fp16.h>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
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

// F16-output variant of testConvertTo. The gold expVal is computed in double like every other
// row, but the kernel evaluates alpha * x + beta in float and rounds once on the half store, so
// a bit-exact vector compare against a double-derived half is not applicable; per the
// HalfTestUtils.hpp policy a single mul-add chain with one final rounding justifies kUlps = 1.
template<typename DT_SOURCE>
void testConvertToF16Output(nvcv::ImageFormat fmtIn, nvcv::ImageFormat fmtOut, int batch, int width, int height,
                            double alpha, double beta, DT_SOURCE setVal, float expVal)
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
    size_t outBufSizeElements = (outSampleStride / sizeof(__half)) * numSamples;
    size_t inBufSizeBytes     = inSampleStride * numSamples;
    size_t outBufSizeBytes    = outSampleStride * numSamples;

    std::vector<DT_SOURCE> srcVec(inBufSizeElements, setVal);
    std::vector<float>     goldVec(outBufSizeElements); // FP32 reference; padding stays 0 as memset below
    std::vector<__half>    testVec(outBufSizeElements);

    setGoldBuffer<float>(goldVec, expVal, width * outAccess->numChannels(), height,
                         static_cast<int>(outAccess->rowStride() / sizeof(__half)),
                         static_cast<int>(outSampleStride / sizeof(__half)), batch);

    EXPECT_EQ(cudaSuccess,
              cudaMemcpyAsync(inData->basePtr(), srcVec.data(), inBufSizeBytes, cudaMemcpyHostToDevice, stream));
    EXPECT_EQ(cudaSuccess, cudaMemsetAsync(outData->basePtr(), 0x0, outBufSizeBytes, stream));

    cvcuda::ConvertTo convertToOp;

    EXPECT_NO_THROW(convertToOp(stream, imgIn, imgOut, alpha, beta, NVCV_ROUND_NEAREST));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSizeBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    test::ExpectNearHalfUlps(goldVec, testVec, 1.f);
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

// F16 pairs, both directions. F16-input rows with wider outputs reuse the exact comparison: the
// kernel widens the half to float, so with the integer-valued test input it evaluates the same
// float mul-add as the u8/f32 rows above (F16 -> integer additionally rounds to nearest in the
// saturating cast on both host and device). F16-output rows go through testConvertToF16Output.

TEST_P(OpConvertTo, OpConvertTo_RGBAf16toRGBAf32)
{
    using fromType = __half;
    using toType   = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBAf16, nvcv::FMT_RGBAf32, batch, width, height, alpha, beta, val,
                                    valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBf16toRGB8)
{
    using fromType = __half;
    using toType   = uint8_t;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBf16, nvcv::FMT_RGB8, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBAf32toRGBAf16)
{
    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    float      val    = 0x10;
    const auto valExp = static_cast<float>(alpha * val + beta);

    testConvertToF16Output<float>(nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf16, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGB8toRGBf16)
{
    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    uint8_t    val    = 0x10;
    const auto valExp = static_cast<float>(alpha * val + beta);

    testConvertToF16Output<uint8_t>(nvcv::FMT_RGB8, nvcv::FMT_RGBf16, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_F16toF16)
{
    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    __half     val    = 0x10;
    const auto valExp = static_cast<float>(alpha * val + beta);

    testConvertToF16Output<__half>(nvcv::FMT_F16, nvcv::FMT_F16, batch, width, height, alpha, beta, val, valExp);
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

// F16 parity stays bit-exact: interleaved (half3) and planar (flattened single-channel half)
// kernels evaluate the same alpha * x + beta in float and round to half at the same single point.
// Widths divisible by 4 additionally route the planar view through the wide half4 path.

TEST(OpConvertToPlanar, rgb8_to_rgbf16_batched_matches_interleaved)
{
    RunConvertToPlanarParityCase<uint8_t, __half>(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGBf16, nvcv::FMT_RGBf16p,
                                                  24, 16, 1.5, -3.0, 2);
}

TEST(OpConvertToPlanar, rgbf16_to_rgbf32_matches_interleaved)
{
    RunConvertToPlanarParityCase<__half, float>(nvcv::FMT_RGBf16, nvcv::FMT_RGBf16p, nvcv::FMT_RGBf32,
                                                nvcv::FMT_RGBf32p, 20, 12, 0.5, 1.0, 2);
}

// =============================================================================
// Per-element gold-reference coverage
//
// The parameterized cases above write one constant to every element, so they cannot detect index
// arithmetic that permutes, drops or duplicates elements inside a row, and their widths (5, 5, 1)
// never reach the vectorized dispatch arms. These cases give every element a distinct value, seed the
// destination with a sentinel, and check the whole raw destination buffer against an independent CPU
// reference. Each case is run at a width whose element count per row is a multiple of the widest
// vector arm and at one that is not, so the vector body and the scalar tail are both covered, and the
// untouched row padding proves nothing is written past the last valid column.
// =============================================================================

namespace {

// Independent CPU reference for one element. This re-implements the operator's documented contract,
// out = saturate(round(alpha * in + beta)), from nearbyint plus an explicit clamp instead of calling
// nvcv::cuda::SaturateCast, so the kernel and the gold do not share an implementation. Integer
// destinations round to nearest with ties to even, matching the device cvt.rni.sat instruction;
// the TRUNCATE modes pre-round toward zero, exactly as the operator documents. The WRAP modes replace
// the clamp with the destination type's own modular reduction, except that an unsigned destination
// still floors at zero.
template<typename SrcT, typename DstT>
DstT ConvertToGoldElement(SrcT src, float alpha, float beta, NVCVRoundMode roundMode)
{
    const bool truncate = roundMode == NVCV_ROUND_TRUNCATE || roundMode == NVCV_ROUND_TRUNCATE_WRAP;
    const bool wrap     = roundMode == NVCV_ROUND_NEAREST_WRAP || roundMode == NVCV_ROUND_TRUNCATE_WRAP;

    const float work = alpha * static_cast<float>(src) + beta;

    if constexpr (std::is_integral_v<DstT>)
    {
        const double rounded = std::nearbyint(static_cast<double>(truncate ? std::trunc(work) : work));

        if (wrap)
        {
            constexpr double kPeriod = static_cast<double>(std::numeric_limits<DstT>::max())
                                     - static_cast<double>(std::numeric_limits<DstT>::lowest()) + 1.0;
            if constexpr (std::is_unsigned_v<DstT>)
            {
                return rounded < 0.0 ? DstT{0} : static_cast<DstT>(rounded - kPeriod * std::floor(rounded / kPeriod));
            }
            else
            {
                const auto lo = static_cast<double>(std::numeric_limits<DstT>::lowest());
                return static_cast<DstT>(lo + (rounded - lo) - kPeriod * std::floor((rounded - lo) / kPeriod));
            }
        }

        if (rounded <= static_cast<double>(std::numeric_limits<DstT>::lowest()))
        {
            return std::numeric_limits<DstT>::lowest();
        }
        if (rounded >= static_cast<double>(std::numeric_limits<DstT>::max()))
        {
            return std::numeric_limits<DstT>::max();
        }
        return static_cast<DstT>(rounded);
    }
    else
    {
        return static_cast<DstT>(work);
    }
}

// Distinct, exactly-representable source values. The cycle lengths are odd and coprime with every
// tested row length, so neighbouring rows and channels never repeat a value at the same position;
// float sources are quarter-integers spanning negatives so signed handling is covered.
template<typename SrcT>
SrcT GoldSourceValue(int64_t i)
{
    if constexpr (std::is_floating_point_v<SrcT>)
    {
        return static_cast<SrcT>((static_cast<double>(i % 509) - 254.0) * 0.25);
    }
    else if constexpr (std::is_signed_v<SrcT>)
    {
        return static_cast<SrcT>((i % 501) - 250);
    }
    else
    {
        return static_cast<SrcT>(i % 251);
    }
}

// Flattened description of a tensor's element storage, valid for interleaved (HWC/NHWC) and planar
// (CHW/NCHW) layouts alike: the elements of one line are contiguous, and line y of plane p of sample n
// starts at n * sampleStride + p * planeStride + y * rowStride bytes.
struct GoldElementLayout
{
    int64_t sampleStride;
    int64_t planeStride;
    int64_t rowStride;
    int64_t elemSize;
    int64_t numSamples;
    int     numPlanes;
    int     numRows;
    int     elemsPerLine;

    int64_t offset(int64_t element) const
    {
        const int64_t row   = element / elemsPerLine;
        const int64_t plane = row / numRows;
        return (plane / numPlanes) * sampleStride + (plane % numPlanes) * planeStride + (row % numRows) * rowStride
             + (element % elemsPerLine) * elemSize;
    }
};

GoldElementLayout MakeGoldElementLayout(const nvcv::TensorDataStridedCuda &data)
{
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(data);
    if (!access)
    {
        // Returning rather than only recording the failure: every field below dereferences `access`,
        // so continuing would crash the test process instead of failing the case.
        ADD_FAILURE() << "TensorDataAccessStridedImagePlanar::Create failed";
        return {};
    }

    const bool planar = data.layout() == nvcv::TENSOR_NCHW || data.layout() == nvcv::TENSOR_CHW;

    GoldElementLayout g;
    g.elemSize     = data.dtype().strideBytes();
    g.numSamples   = access->numSamples();
    g.numRows      = access->numRows();
    g.rowStride    = access->rowStride();
    g.sampleStride = access->sampleStride();
    g.numPlanes    = planar ? access->numChannels() : 1;
    g.planeStride  = planar ? access->chStride() : 0;
    g.elemsPerLine = planar ? access->numCols() : access->numCols() * access->numChannels();

    // The flattened description assumes a line's elements are contiguous; nvcv-allocated tensors are.
    EXPECT_EQ(planar ? g.elemSize : access->numChannels() * g.elemSize, access->colStride());
    if (!planar)
    {
        EXPECT_EQ(g.elemSize, access->chStride());
    }
    return g;
}

constexpr uint8_t kGoldSentinel = 0xA5;

// Convert a tensor whose every element holds a distinct value and check the entire destination
// buffer: valid elements against the independent reference, padding against the sentinel.
template<typename SrcT, typename DstT>
void RunConvertToGoldCase(nvcv::ImageFormat inFmt, nvcv::ImageFormat outFmt, int batch, int width, int height,
                          double alpha, double beta, NVCVRoundMode roundMode = NVCV_ROUND_NEAREST,
                          double tolerance = 0.0)
{
    SCOPED_TRACE(testing::Message() << "batch=" << batch << " width=" << width << " height=" << height
                                    << " alpha=" << alpha << " beta=" << beta << " roundMode=" << roundMode);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src = nvcv::util::CreateTensor(batch, width, height, inFmt);
    nvcv::Tensor dst = nvcv::util::CreateTensor(batch, width, height, outFmt);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);

    const auto sg = MakeGoldElementLayout(*srcData);
    const auto dg = MakeGoldElementLayout(*dstData);

    const auto srcBytes = static_cast<size_t>(srcData->stride(0) * srcData->shape(0));
    const auto dstBytes = static_cast<size_t>(dstData->stride(0) * dstData->shape(0));

    std::vector<uint8_t> hostSrc(srcBytes, 0);
    std::vector<uint8_t> hostGold(dstBytes, kGoldSentinel);
    std::vector<uint8_t> hostDst(dstBytes, 0);
    std::vector<uint8_t> isElement(dstBytes, 0);

    const auto alphaF = static_cast<float>(alpha);
    const auto betaF  = static_cast<float>(beta);

    int64_t counter = 0;
    for (; counter < sg.numSamples * sg.numPlanes * sg.numRows * sg.elemsPerLine; ++counter)
    {
        const SrcT sv = GoldSourceValue<SrcT>(counter);
        const DstT dv = ConvertToGoldElement<SrcT, DstT>(sv, alphaF, betaF, roundMode);
        std::memcpy(&hostSrc[sg.offset(counter)], &sv, sizeof(SrcT));
        std::memcpy(&hostGold[dg.offset(counter)], &dv, sizeof(DstT));
        std::memset(&isElement[dg.offset(counter)], 1, sizeof(DstT));
    }

    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(srcData->basePtr(), hostSrc.data(), srcBytes, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), kGoldSentinel, dstBytes, stream));

    cvcuda::ConvertTo op;
    EXPECT_NO_THROW(op(stream, src, dst, alpha, beta, roundMode));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hostDst.data(), dstData->basePtr(), dstBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Valid elements: bit-exact for integer destinations (alpha and beta are dyadic in those cases, so
    // alpha * src + beta is exact and device FMA contraction cannot change the result), within a tight
    // tolerance for float destinations, where contraction makes a bit-exact host comparison invalid.
    size_t  mismatches = 0;
    int64_t checked    = 0;
    for (; checked < dg.numSamples * dg.numPlanes * dg.numRows * dg.elemsPerLine; ++checked)
    {
        DstT got;
        DstT want;
        std::memcpy(&got, &hostDst[dg.offset(checked)], sizeof(DstT));
        std::memcpy(&want, &hostGold[dg.offset(checked)], sizeof(DstT));
        if (const bool ok = std::is_integral_v<DstT>
                              ? (got == want)
                              : (std::abs(static_cast<double>(got) - static_cast<double>(want)) <= tolerance);
            ok)
        {
            continue;
        }
        ++mismatches;
        if (mismatches > 8)
        {
            continue;
        }
        SCOPED_TRACE(testing::Message() << "element " << checked);
        if constexpr (std::is_integral_v<DstT>)
        {
            EXPECT_EQ(want, got);
        }
        else
        {
            EXPECT_NEAR(static_cast<double>(want), static_cast<double>(got), tolerance);
        }
    }

    // Padding: every byte outside a valid element must still hold the sentinel.
    size_t padMismatches = 0;
    for (size_t b = 0; b < dstBytes; ++b)
    {
        if (isElement[b] || hostDst[b] == kGoldSentinel)
        {
            continue;
        }
        ++padMismatches;
        if (padMismatches <= 8)
        {
            ADD_FAILURE() << "destination padding byte " << b << " was overwritten with "
                          << static_cast<int>(hostDst[b]);
        }
    }

    // Guard against the checks above silently covering nothing.
    EXPECT_EQ(counter, checked);
    EXPECT_GT(checked, 0);
    EXPECT_EQ(0u, mismatches);
    EXPECT_EQ(0u, padMismatches);
}

void RunPlanarU8F32BulkGoldCase(int batch, int channels, int width, int height, int sourceOffset, int destinationOffset,
                                int rowPadding, int planePadding, double alpha, double beta,
                                NVCVRoundMode roundMode = NVCV_ROUND_NEAREST)
{
    SCOPED_TRACE(testing::Message() << "N=" << batch << " C=" << channels << " W=" << width << " H=" << height
                                    << " sourceOffset=" << sourceOffset << " destinationOffset=" << destinationOffset
                                    << " rowPadding=" << rowPadding << " planePadding=" << planePadding
                                    << " alpha=" << alpha << " beta=" << beta << " roundMode=" << roundMode);
    const int sourceRow          = width + rowPadding;
    const int destinationRow     = (width + rowPadding) * sizeof(float);
    const int sourcePlane        = sourceRow * height + planePadding;
    const int destinationPlane   = destinationRow * height + planePadding * sizeof(float);
    const int sourceBytes        = sourceOffset + batch * channels * sourcePlane + 32;
    const int destinationBytes   = destinationOffset + batch * channels * destinationPlane + 32;
    auto      sourceStorage      = nvcv::util::CreateTensor(1, sourceBytes, 1, nvcv::FMT_U8);
    auto      destinationStorage = nvcv::util::CreateTensor(1, destinationBytes, 1, nvcv::FMT_U8);
    auto      sourceData         = sourceStorage.exportData<nvcv::TensorDataStridedCuda>();
    auto      destinationData    = destinationStorage.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(sourceData && destinationData);
    auto wrap = [&](nvcv::Byte *pointer, nvcv::DataType dtype, int rowStride, int planeStride)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = reinterpret_cast<NVCVByte *>(pointer);
        buffer.strides[0] = channels * planeStride;
        buffer.strides[1] = planeStride;
        buffer.strides[2] = rowStride;
        buffer.strides[3] = dtype.strideBytes();
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{batch, channels, height, width}, "NCHW"},
            dtype, buffer
        });
    };
    auto source = wrap(sourceData->basePtr() + sourceOffset, nvcv::TYPE_U8, sourceRow, sourcePlane);
    auto destination
        = wrap(destinationData->basePtr() + destinationOffset, nvcv::TYPE_F32, destinationRow, destinationPlane);
    std::vector<uint8_t> hostSource(sourceBytes, kGoldSentinel);
    std::vector<uint8_t> expected(destinationBytes, kGoldSentinel);
    std::vector<uint8_t> actual(destinationBytes);
    for (int plane = 0; plane < batch * channels; ++plane)
    {
        for (int row = 0; row < height; ++row)
        {
            for (int column = 0; column < width; ++column)
            {
                const uint8_t value = ((plane * height + row) * width + column) * 73 + 19;
                hostSource[sourceOffset + plane * sourcePlane + row * sourceRow + column] = value;
                const float gold
                    = std::fma(static_cast<float>(alpha), static_cast<float>(value), static_cast<float>(beta));
                std::memcpy(expected.data() + destinationOffset + plane * destinationPlane + row * destinationRow
                                + column * sizeof(float),
                            &gold, sizeof(gold));
            }
        }
    }
    test::StreamGuard streamGuard;
    cudaStream_t      stream = streamGuard.get();
    ASSERT_NE(nullptr, stream);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(sourceData->basePtr(), hostSource.data(), sourceBytes, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(destinationData->basePtr(), kGoldSentinel, destinationBytes, stream));
    cvcuda::ConvertTo op;
    EXPECT_NO_THROW(op(stream, source, destination, alpha, beta, roundMode));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), destinationData->basePtr(), destinationBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
    std::vector<uint8_t> sourceAfter(sourceBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(sourceAfter.data(), sourceData->basePtr(), sourceBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(hostSource, sourceAfter);
}

} // namespace

TEST(OpConvertToGold, planar_u8_to_f32_bulk_tiles)
{
    for (int channels : {1, 2, 3, 4})
    {
        for (auto mode : {NVCV_ROUND_NEAREST, NVCV_ROUND_TRUNCATE, NVCV_ROUND_NEAREST_WRAP, NVCV_ROUND_TRUNCATE_WRAP})
        {
            RunPlanarU8F32BulkGoldCase(2, channels, 256, 8, 0, 0, 0, 0, 0.123, 0.456, mode);
            RunPlanarU8F32BulkGoldCase(2, channels, 256, 8, 0, 0, 0, 0, -1.75, 300.5, mode);
            RunPlanarU8F32BulkGoldCase(2, channels, 256, 8, 0, 0, 0, 0, 0.0, -4.0, mode);
        }
    }
    RunPlanarU8F32BulkGoldCase(1, 3, 127, 4096, 0, 0, 0, 0, 0.123, 0.456);
}

TEST(OpConvertToGold, planar_u8_to_f32_bulk_fallbacks)
{
    for (int width : {1, 4095, 4096, 4097}) RunPlanarU8F32BulkGoldCase(1, 3, width, 1, 0, 0, 0, 0, 0.123, 0.456);
    RunPlanarU8F32BulkGoldCase(2, 3, 256, 8, 1, 0, 0, 0, 0.123, 0.456);
    RunPlanarU8F32BulkGoldCase(2, 3, 256, 8, 0, 4, 0, 0, 0.123, 0.456);
    RunPlanarU8F32BulkGoldCase(2, 3, 256, 8, 0, 0, 17, 0, 0.123, 0.456);
    RunPlanarU8F32BulkGoldCase(2, 3, 256, 8, 0, 0, 0, 19, 0.123, 0.456);
}

TEST(OpConvertToGold, classical_rgb_u8_layout_and_destination_controls)
{
    for (int width : {4095, 4096, 4097})
    {
        for (auto mode : {NVCV_ROUND_NEAREST, NVCV_ROUND_TRUNCATE, NVCV_ROUND_NEAREST_WRAP, NVCV_ROUND_TRUNCATE_WRAP})
        {
            RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, width, 3, 1.5, -4.25, mode);
            RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, 2, width, 3, 1.5, -4.25, mode);
            RunConvertToGoldCase<uint8_t, float>(nvcv::FMT_RGB8, nvcv::FMT_RGBf32, 2, width, 3, 1.5, -4.25, mode);
            RunConvertToGoldCase<uint8_t, float>(nvcv::FMT_RGB8p, nvcv::FMT_RGBf32p, 2, width, 3, 1.5, -4.25, mode);
        }
    }
}

TEST(OpConvertToGold, planar_f32_in_place_preserved)
{
    nvcv::Tensor tensor{
        {{2, 3, 8, 256}, "NCHW"},
        nvcv::TYPE_F32
    };
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(data);
    const auto           layout = MakeGoldElementLayout(*data);
    const auto           bytes  = data->stride(0) * data->shape(0);
    std::vector<uint8_t> input(bytes, kGoldSentinel);
    std::vector<uint8_t> expected(bytes, kGoldSentinel);
    std::vector<uint8_t> actual(bytes);
    for (int64_t element = 0; element < 2 * 3 * 8 * 256; ++element)
    {
        const float value = GoldSourceValue<float>(element);
        const float gold  = std::fma(0.5f, value, -4.0f);
        std::memcpy(input.data() + layout.offset(element), &value, sizeof(value));
        std::memcpy(expected.data() + layout.offset(element), &gold, sizeof(gold));
    }
    test::StreamGuard streamGuard;
    cudaStream_t      stream = streamGuard.get();
    ASSERT_NE(nullptr, stream);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(data->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice, stream));
    cvcuda::ConvertTo op;
    EXPECT_NO_THROW(op(stream, tensor, tensor, 0.5, -4.0));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), data->basePtr(), bytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
}

// Interleaved: one case per (source, destination) pair on the optimization-target surface, at a width
// whose row element count is a multiple of 16 (the widest vector group) and at one that is not, so the
// vector body and the scalar tail are both covered. Dyadic alpha/beta keep integer destinations
// bit-exact -- alpha * src + beta is then exact in float, so FMA contraction cannot change it -- while
// still spreading the results across the destination range.

TEST(OpConvertToGold, interleaved_u8_to_u8)
{
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_U8, nvcv::FMT_U8, 3, 64, 5, 1.0, -4.0);
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_U8, nvcv::FMT_U8, 2, 37, 3, 1.0, -4.0);
}

TEST(OpConvertToGold, interleaved_f32_to_u8)
{
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 3, 64, 5, 2.0, 128.25);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 2, 37, 3, 2.0, 128.25);
}

TEST(OpConvertToGold, interleaved_f32_to_f32)
{
    RunConvertToGoldCase<float, float>(nvcv::FMT_F32, nvcv::FMT_F32, 3, 64, 5, 0.123, 0.456, NVCV_ROUND_NEAREST, 1e-4);
    RunConvertToGoldCase<float, float>(nvcv::FMT_F32, nvcv::FMT_F32, 2, 37, 3, 0.123, 0.456, NVCV_ROUND_NEAREST, 1e-4);
}

TEST(OpConvertToGold, interleaved_rgb8_to_rgb8)
{
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGB8, nvcv::FMT_RGB8, 3, 16, 5, 1.0, -4.0);
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, 17, 3, 1.0, -4.0);
}

TEST(OpConvertToGold, interleaved_rgb8_to_rgbf32)
{
    RunConvertToGoldCase<uint8_t, float>(nvcv::FMT_RGB8, nvcv::FMT_RGBf32, 3, 256, 5, 1.0, 0.0);
    RunConvertToGoldCase<uint8_t, float>(nvcv::FMT_RGB8, nvcv::FMT_RGBf32, 2, 257, 3, 0.5, -4.0);
}

TEST(OpConvertToGold, interleaved_rgbf32_to_rgb8)
{
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32, nvcv::FMT_RGB8, 3, 16, 5, 2.0, 128.25);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32, nvcv::FMT_RGB8, 2, 17, 3, 2.0, 128.25);
}

TEST(OpConvertToGold, interleaved_rgbf32_to_rgbf32)
{
    RunConvertToGoldCase<float, float>(nvcv::FMT_RGBf32, nvcv::FMT_RGBf32, 2, 16, 5, 0.123, 0.456, NVCV_ROUND_NEAREST,
                                       1e-4);
    RunConvertToGoldCase<float, float>(nvcv::FMT_RGBf32, nvcv::FMT_RGBf32, 2, 17, 3, 0.123, 0.456, NVCV_ROUND_NEAREST,
                                       1e-4);
}

TEST(OpConvertToGold, interleaved_rgba8_to_rgba8)
{
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, 3, 16, 5, 1.0, -4.0);
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, 2, 19, 3, 1.0, -4.0);
}

// Saturation: alpha/beta chosen so a large share of the elements clamp at both ends of the
// destination range, covering the saturating-cast branches the spread cases above mostly avoid.

TEST(OpConvertToGold, interleaved_saturates_both_ends)
{
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_U8, nvcv::FMT_U8, 2, 64, 5, 2.0, -128.0);
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, 17, 3, 2.0, -128.0);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 2, 64, 5, 8.0, -64.0);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32, nvcv::FMT_RGB8, 2, 17, 3, 8.0, -64.0);
}

// Planar: the same reference applied to the native NCHW/CHW path, which the parity cases above only
// check against the interleaved result.

TEST(OpConvertToGold, planar_rgb8_to_rgb8)
{
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, 3, 64, 5, 1.0, -4.0);
    RunConvertToGoldCase<uint8_t, uint8_t>(nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, 2, 37, 3, 1.0, -4.0);
}

TEST(OpConvertToGold, planar_rgbf32_to_rgb8)
{
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32p, nvcv::FMT_RGB8p, 3, 64, 5, 2.0, 128.25);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32p, nvcv::FMT_RGB8p, 2, 37, 3, 2.0, 128.25);
}

TEST(OpConvertToGold, planar_rgbaf32_to_rgbaf32)
{
    RunConvertToGoldCase<float, float>(nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32p, 2, 64, 5, 0.123, 0.456,
                                       NVCV_ROUND_NEAREST, 1e-4);
    RunConvertToGoldCase<float, float>(nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32p, 2, 37, 3, 0.123, 0.456,
                                       NVCV_ROUND_NEAREST, 1e-4);
}

// TRUNCATE across a vectorized interleaved width: the rounding mode must reach every element of a
// vector group, not only the first, and must not be applied to a float destination.

TEST(OpConvertToGold, interleaved_truncate_reaches_every_element)
{
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 2, 64, 5, 2.0, 128.25, NVCV_ROUND_TRUNCATE);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32, nvcv::FMT_RGB8, 2, 16, 5, 2.0, 128.25, NVCV_ROUND_TRUNCATE);
    RunConvertToGoldCase<float, float>(nvcv::FMT_RGBf32, nvcv::FMT_RGBf32, 2, 16, 5, 0.123, 0.456, NVCV_ROUND_TRUNCATE,
                                       1e-4);
}

// The wrapping round modes across a vectorized width. The existing OpConvertTo_Wrap cases pin single
// values on 5x5 tensors, which never reach a vector arm; these drive every element of a full group
// past both ends of the destination period, so a group whose lanes were reduced with the wrong
// period -- or not at all -- fails. alpha and beta are dyadic, so the pre-wrap value is exact in
// float and the modular reduction is unambiguous.

TEST(OpConvertToGold, interleaved_wrap_reaches_every_element)
{
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 2, 64, 5, 8.0, 0.25, NVCV_ROUND_NEAREST_WRAP);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 2, 37, 3, 8.0, 0.25, NVCV_ROUND_TRUNCATE_WRAP);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32, nvcv::FMT_RGB8, 2, 16, 5, 8.0, 0.25,
                                         NVCV_ROUND_NEAREST_WRAP);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32, nvcv::FMT_RGB8, 2, 17, 3, 8.0, 0.25,
                                         NVCV_ROUND_TRUNCATE_WRAP);
}

TEST(OpConvertToGold, planar_wrap_reaches_every_element)
{
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32p, nvcv::FMT_RGB8p, 2, 64, 5, 8.0, 0.25,
                                         NVCV_ROUND_NEAREST_WRAP);
    RunConvertToGoldCase<float, uint8_t>(nvcv::FMT_RGBf32p, nvcv::FMT_RGB8p, 2, 37, 3, 8.0, 0.25,
                                         NVCV_ROUND_TRUNCATE_WRAP);
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
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U32, nvcv::FMT_F32, 24, 24, 24, 24, 3, 3}, // unsupported input data type (32-bit unsigned; F16 is now valid)
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F32, nvcv::FMT_U32, 24, 24, 24, 24, 3, 3}, // unsupported output data type (32-bit unsigned)
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

TEST(OpConvertTo_Negative, invalid_round_mode)
{
    // An out-of-range enumerator value must be rejected, not silently fallen through to NEAREST.
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    nvcv::Tensor      imgIn  = nvcv::util::CreateTensor(1, 5, 5, nvcv::FMT_F32);
    nvcv::Tensor      imgOut = nvcv::util::CreateTensor(1, 5, 5, nvcv::FMT_S32);
    cvcuda::ConvertTo op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, imgIn, imgOut, 1.0, 0.0, static_cast<NVCVRoundMode>(99)); }));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpConvertTo_Negative, rejects_complex_data_type)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor      imgIn  = nvcv::util::CreateTensor(1, 4, 4, nvcv::FMT_C64);
    nvcv::Tensor      imgOut = nvcv::util::CreateTensor(1, 4, 4, nvcv::FMT_F32);
    cvcuda::ConvertTo op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, imgIn, imgOut, 1.0, 0.0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpConvertTo_Negative, rejects_mixed_width_tensor_dtype)
{
    const nvcv::DataType mixedWidth{nvcv::DataKind::UNSIGNED, static_cast<nvcv::Packing>(NVCV_PACKING_X32_Y24b8)};
    nvcv::Tensor         imgIn(
                {
                    {1, 5, 5, 2},
                    "NHWC"
    },
                mixedWidth);
    nvcv::Tensor imgOut(
        {
            {1, 5, 5, 2},
            "NHWC"
    },
        mixedWidth);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::ConvertTo op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, imgIn, imgOut, 1.0, 0.0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpConvertTo_Negative, rejects_more_than_four_channels)
{
    const nvcv::TensorShape shape{
        {1, 4, 4, 5},
        nvcv::TENSOR_NHWC
    };
    nvcv::Tensor imgIn(shape, nvcv::TYPE_U8);
    nvcv::Tensor imgOut(shape, nvcv::TYPE_U8);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::ConvertTo op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, imgIn, imgOut, 1.0, 0.0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// =============================================================================
// Wrapping round modes: NEAREST_WRAP and TRUNCATE_WRAP
//
// Reduction is modulo 2^(8*sizeof(T)).  Unsigned negatives map to zero.
// Signed values wrap symmetrically into [-span/2, span/2).
// After wrapping, SaturateCast is a no-op.
// =============================================================================

// Verify that the saturating modes are bit-identical before and after the addition of the wrap modes.
TEST(OpConvertTo_Wrap, saturating_modes_unchanged)
{
    // Values that would change if NEAREST were silently replaced with NEAREST_WRAP.
    testConvertTo<float, uint16_t>(nvcv::FMT_F32, nvcv::FMT_U16, 1, 4, 4, 1.0, 0.0, 92414.59f,
                                   std::numeric_limits<uint16_t>::max(), NVCV_ROUND_NEAREST);
    testConvertTo<float, uint16_t>(nvcv::FMT_F32, nvcv::FMT_U16, 1, 4, 4, 1.0, 0.0, -5000.0f, uint16_t{0},
                                   NVCV_ROUND_NEAREST);
}

// --- uint8 (unsigned 8-bit) --- // NOSONAR: section header, not commented-out code.

TEST(OpConvertTo_Wrap, u8_in_range_nearest)
{
    // 100 is already in [0, 255]: result is 100 for both modes.
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, 100.0f, uint8_t{100},
                                  NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, u8_overflow_nearest)
{
    // 300 = 256 + 44 → nearest→wrap gives 44.
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, 300.0f, uint8_t{44},
                                  NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, u8_full_period_nearest)
{
    // 256 is exactly one period past 0 → 0.
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, 256.0f, uint8_t{0},
                                  NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, u8_negative_maps_to_zero)
{
    // Unsigned asymmetry: negative → 0.
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, -50.0f, uint8_t{0},
                                  NVCV_ROUND_NEAREST_WRAP);
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, -50.0f, uint8_t{0},
                                  NVCV_ROUND_TRUNCATE_WRAP);
}

TEST(OpConvertTo_Wrap, u8_overflow_truncate)
{
    // 300.9 truncated → 300, wrapped → 300 - 256 = 44.
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, 300.9f, uint8_t{44},
                                  NVCV_ROUND_TRUNCATE_WRAP);
}

TEST(OpConvertTo_Wrap, u8_ties_nearest_vs_truncate)
{
    // 2.5 → nearest (ties to even) → 2; truncate → 2. Both give 2.
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, 2.5f, uint8_t{2},
                                  NVCV_ROUND_NEAREST_WRAP);
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, 2.5f, uint8_t{2},
                                  NVCV_ROUND_TRUNCATE_WRAP);
    // 3.5 → nearest (ties to even) → 4; truncate → 3.
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, 3.5f, uint8_t{4},
                                  NVCV_ROUND_NEAREST_WRAP);
    testConvertTo<float, uint8_t>(nvcv::FMT_F32, nvcv::FMT_U8, 1, 4, 4, 1.0, 0.0, 3.5f, uint8_t{3},
                                  NVCV_ROUND_TRUNCATE_WRAP);
}

// --- int8 (signed 8-bit) --- // NOSONAR: section header, not commented-out code.

TEST(OpConvertTo_Wrap, s8_in_range)
{
    testConvertTo<float, int8_t>(nvcv::FMT_F32, nvcv::FMT_S8, 1, 4, 4, 1.0, 0.0, -50.0f, int8_t{-50},
                                 NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, s8_positive_overflow)
{
    // 200 mod 256 = 200; 200 >= 128 → 200 - 256 = -56.
    testConvertTo<float, int8_t>(nvcv::FMT_F32, nvcv::FMT_S8, 1, 4, 4, 1.0, 0.0, 200.0f, int8_t{-56},
                                 NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, s8_negative_underflow)
{
    // -200 mod 256 (via trunc): trunc(-200/256)=0, w=-200; -200 < -128 → -200+256=56.
    testConvertTo<float, int8_t>(nvcv::FMT_F32, nvcv::FMT_S8, 1, 4, 4, 1.0, 0.0, -200.0f, int8_t{56},
                                 NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, s8_boundary_128)
{
    // 128 → 128 - 256 = -128 (signed min).
    testConvertTo<float, int8_t>(nvcv::FMT_F32, nvcv::FMT_S8, 1, 4, 4, 1.0, 0.0, 128.0f, int8_t{-128},
                                 NVCV_ROUND_NEAREST_WRAP);
}

// --- uint16 (unsigned 16-bit) — must match !1662 byte-for-byte ---

TEST(OpConvertTo_Wrap, u16_overflow_truncate)
{
    testConvertTo<float, uint16_t>(nvcv::FMT_F32, nvcv::FMT_U16, 1, 4, 4, 1.0, 0.0, 92414.59f, uint16_t{26878},
                                   NVCV_ROUND_TRUNCATE_WRAP);
}

TEST(OpConvertTo_Wrap, u16_large_positive_truncate)
{
    testConvertTo<float, uint16_t>(nvcv::FMT_F32, nvcv::FMT_U16, 1, 4, 4, 1.0, 0.0, 70000.0f, uint16_t{4464},
                                   NVCV_ROUND_TRUNCATE_WRAP);
}

TEST(OpConvertTo_Wrap, u16_negative_maps_to_zero)
{
    testConvertTo<float, uint16_t>(nvcv::FMT_F32, nvcv::FMT_U16, 1, 4, 4, 1.0, 0.0, -5000.0f, uint16_t{0},
                                   NVCV_ROUND_TRUNCATE_WRAP);
    testConvertTo<float, uint16_t>(nvcv::FMT_F32, nvcv::FMT_U16, 1, 4, 4, 1.0, 0.0, -70000.0f, uint16_t{0},
                                   NVCV_ROUND_TRUNCATE_WRAP);
}

TEST(OpConvertTo_Wrap, u16_near_max_truncate)
{
    // 65535.9 truncated → 65535, in-range → 65535.
    testConvertTo<float, uint16_t>(nvcv::FMT_F32, nvcv::FMT_U16, 1, 4, 4, 1.0, 0.0, 65535.9f, uint16_t{65535},
                                   NVCV_ROUND_TRUNCATE_WRAP);
}

TEST(OpConvertTo_Wrap, u16_exact_period_truncate)
{
    // 65536 = one full period → 0.
    testConvertTo<float, uint16_t>(nvcv::FMT_F32, nvcv::FMT_U16, 1, 4, 4, 1.0, 0.0, 65536.0f, uint16_t{0},
                                   NVCV_ROUND_TRUNCATE_WRAP);
}

// --- int16 (signed 16-bit) — must match !1662 byte-for-byte ---

TEST(OpConvertTo_Wrap, s16_large_positive_nearest)
{
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, 92414.59f, int16_t{26879},
                                  NVCV_ROUND_NEAREST_WRAP);
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, 70000.0f, int16_t{4464},
                                  NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, s16_negative_nearest)
{
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, -5000.0f, int16_t{-5000},
                                  NVCV_ROUND_NEAREST_WRAP);
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, -70000.0f, int16_t{-4464},
                                  NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, s16_boundary_32768_nearest)
{
    // 32768 wraps to -32768 (signed min).
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, 32768.0f, int16_t{-32768},
                                  NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, s16_ties_to_even_nearest)
{
    // 3.5 → nearest ties-to-even → 4; 2.5 → 2.
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, 3.5f, int16_t{4},
                                  NVCV_ROUND_NEAREST_WRAP);
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, 2.5f, int16_t{2},
                                  NVCV_ROUND_NEAREST_WRAP);
}

TEST(OpConvertTo_Wrap, s16_ties_truncate)
{
    // 3.5 → truncate → 3; 2.5 → 2.
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, 3.5f, int16_t{3},
                                  NVCV_ROUND_TRUNCATE_WRAP);
    testConvertTo<float, int16_t>(nvcv::FMT_F32, nvcv::FMT_S16, 1, 4, 4, 1.0, 0.0, 2.5f, int16_t{2},
                                  NVCV_ROUND_TRUNCATE_WRAP);
}

// --- int32 (signed 32-bit): wrap modes must be rejected ---
// The float32 working arithmetic cannot represent the 2^31 wrap period, so modular
// reduction would silently produce wrong results. The API rejects these combinations
// before any kernel is launched.

TEST(OpConvertTo_Wrap, s32_wrap_rejected)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    nvcv::Tensor      imgIn  = nvcv::util::CreateTensor(1, 4, 4, nvcv::FMT_F32);
    nvcv::Tensor      imgOut = nvcv::util::CreateTensor(1, 4, 4, nvcv::FMT_S32);
    cvcuda::ConvertTo op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, imgIn, imgOut, 1.0, 0.0, NVCV_ROUND_NEAREST_WRAP); }));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, imgIn, imgOut, 1.0, 0.0, NVCV_ROUND_TRUNCATE_WRAP); }));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Float output: wrap modes must be inert (same result as saturating counterpart).
TEST(OpConvertTo_Wrap, float_output_wrap_is_inert)
{
    testConvertTo<float, float>(nvcv::FMT_F32, nvcv::FMT_F32, 1, 4, 4, 1.0, 0.0, 2.7f, 2.7f, NVCV_ROUND_NEAREST_WRAP);
    testConvertTo<float, float>(nvcv::FMT_F32, nvcv::FMT_F32, 1, 4, 4, 1.0, 0.0, 2.7f, 2.7f, NVCV_ROUND_TRUNCATE_WRAP);
}
