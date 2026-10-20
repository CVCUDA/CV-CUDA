/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ElementwiseOpHarness.hpp"
#include "PlanarParityUtils.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpInvert.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

namespace {

// Independent CPU gold reference: out = bound - in per element, where bound is the dtype maximum
// (255 / 65535) for unsigned integers and 1.0 for float. This mirrors the documented oracle
// (torchvision.transforms.v2.functional.invert / OpenCV cv::bitwise_not) and is intentionally
// computed independently of the kernel so the bit-exact EXPECT_EQ is a real regression check.
//
// goldFor(DT{}) yields the per-element reference for a given element type; one expression covers
// every declared dtype.
const auto goldFor = []<typename DT>(DT)
{
    return [](DT v)
    {
        if constexpr (std::is_integral_v<DT> && std::is_signed_v<DT>)
        {
            if (v < 0)
            {
                return std::numeric_limits<DT>::max();
            }
        }
        return static_cast<DT>(ew::Bound<DT>() - v);
    };
};

// Run the operator. A single generic invoker serves Tensor, ImageBatchVarShape, and the parity
// harness (Invert takes no extra parameters).
const auto invoke = [](cudaStream_t s, const auto &in, auto &out)
{
    cvcuda::Invert op;
    op(s, in, out);
};

} // namespace

// Tensor correctness: bit-exact vs the CPU gold, over the declared dtype × channel matrix --------
// clang-format off
NVCV_TEST_SUITE_P(OpInvert, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    //   width, height, batch,            format          (dtype / channels)
    {       66,     55,     1,   nvcv::FMT_U8     }, // u8  / 1ch
    {      123,     67,     3,   nvcv::FMT_RGB8   }, // u8  / 3ch
    {       42,     53,     4,   nvcv::FMT_RGBA8  }, // u8  / 4ch
    {       80,     40,     2,   nvcv::FMT_U16    }, // u16 / 1ch
    {       17,     19,     1,   nvcv::FMT_F32    }, // f32 / 1ch
    {      101,     33,     2,   nvcv::FMT_RGBf32 }, // f32 / 3ch
    {       64,     48,     3,   nvcv::FMT_RGBAf32}, // f32 / 4ch
});

// clang-format on
TEST_P(OpInvert, tensor_correct_output)
{
    // Invert has no dtype-dependent parameter, so the invoke factory returns the same invoker for
    // every element type.
    ew::RunTensorCorrectDispatch(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                 nvcv::ImageFormat{GetParamValue<3>()}, goldFor, [](auto) { return invoke; });
}

// VarShape correctness: bit-exact vs the CPU gold ------------------------------------------------
TEST(OpInvert, varshape_correct_output)
{
    ew::RunVarShapeCorrect([](uint8_t v) { return static_cast<uint8_t>(ew::Bound<uint8_t>() - v); }, invoke);
}

// Caller-owned RGB8 rows with an odd base offset and odd row stride alternate 4-byte alignment.
// Width 23 leaves a 3-pixel tail, exercising the packed kernel's wide and scalar row arms.
TEST(OpInvert, varshape_rgb8_alignment_fallback_correct_output)
{
    constexpr int     kBatch      = 2;
    constexpr int     kWidth      = 23;
    constexpr int     kHeight     = 5;
    constexpr int64_t kRowBytes   = static_cast<int64_t>(kWidth) * 3;
    constexpr int64_t kRowStride  = kRowBytes + 2;
    constexpr int64_t kImageBytes = kRowStride * kHeight;
    constexpr int64_t kBaseOffset = 1;
    constexpr auto    kBytes      = static_cast<size_t>(kBaseOffset + kBatch * kImageBytes);

    auto wrapImage = [=](NVCVByte *base)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes           = 1;
        buffer.planes[0].width     = kWidth;
        buffer.planes[0].height    = kHeight;
        buffer.planes[0].rowStride = kRowStride;
        buffer.planes[0].basePtr   = base;
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8, buffer});
    };

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kBytes));

    std::vector<uint8_t> srcBytes(kBytes);
    for (size_t i = 0; i < kBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    const std::vector<uint8_t> dstSentinel(kBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), kBytes, cudaMemcpyHostToDevice));

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    for (int i = 0; i < kBatch; ++i)
    {
        srcImages.push_back(wrapImage(srcPtr + kBaseOffset + i * kImageBytes));
        dstImages.push_back(wrapImage(dstPtr + kBaseOffset + i * kImageBytes));
    }

    nvcv::ImageBatchVarShape srcBatch(kBatch);
    nvcv::ImageBatchVarShape dstBatch(kBatch);
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, srcBatch, dstBatch));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    for (int i = 0; i < kBatch; ++i)
    {
        for (int y = 0; y < kHeight; ++y)
        {
            const auto row = static_cast<size_t>(kBaseOffset + i * kImageBytes + y * kRowStride);
            for (int64_t x = 0; x < kRowBytes; ++x)
            {
                ASSERT_EQ(static_cast<uint8_t>(255 - srcBytes[row + x]), dstBytes[row + x])
                    << "image " << i << " row " << y << " byte " << x;
            }
            for (int64_t x = kRowBytes; x < kRowStride; ++x)
            {
                ASSERT_EQ(0xA5, dstBytes[row + x]) << "row padding overwritten";
            }
        }
    }
}

// Aligned and one-byte-offset RGB8 rows in one batch cover both packed four-pixel groups and the
// scalar tail while keeping each aligned row's address stable across its padded row stride.
TEST(OpInvert, varshape_rgb8_wide_and_tail_correct_output)
{
    constexpr int                        kBatch            = 2;
    constexpr int                        kWidth            = 9;
    constexpr int                        kHeight           = 5;
    constexpr int64_t                    kRowBytes         = static_cast<int64_t>(kWidth) * 3;
    constexpr int64_t                    kRowStride        = 28;
    constexpr int64_t                    kImageBytes       = kRowStride * kHeight;
    constexpr size_t                     kMisalignedOffset = 1;
    constexpr std::array<size_t, kBatch> kImageOffsets     = {0, static_cast<size_t>(kImageBytes) + kMisalignedOffset};
    constexpr size_t                     kBytes = static_cast<size_t>(kImageBytes) * kBatch + kMisalignedOffset;

    auto wrapImage = [=](NVCVByte *base)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes           = 1;
        buffer.planes[0].width     = kWidth;
        buffer.planes[0].height    = kHeight;
        buffer.planes[0].rowStride = kRowStride;
        buffer.planes[0].basePtr   = base;
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8, buffer});
    };

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kBytes));

    std::vector<uint8_t> srcBytes(kBytes);
    for (size_t i = 0; i < kBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    const std::vector<uint8_t> dstSentinel(kBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), kBytes, cudaMemcpyHostToDevice));

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    for (const auto offset : kImageOffsets)
    {
        srcImages.push_back(wrapImage(srcPtr + offset));
        dstImages.push_back(wrapImage(dstPtr + offset));
    }

    nvcv::ImageBatchVarShape srcBatch(kBatch);
    nvcv::ImageBatchVarShape dstBatch(kBatch);
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, srcBatch, dstBatch));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    for (int i = 0; i < kBatch; ++i)
    {
        for (int y = 0; y < kHeight; ++y)
        {
            const size_t row = kImageOffsets[i] + static_cast<size_t>(y * kRowStride);
            for (int64_t x = 0; x < kRowBytes; ++x)
            {
                ASSERT_EQ(static_cast<uint8_t>(255 - srcBytes[row + x]), dstBytes[row + x])
                    << "image " << i << " row " << y << " byte " << x;
            }
            for (int64_t x = kRowBytes; x < kRowStride; ++x)
            {
                ASSERT_EQ(0xA5, dstBytes[row + x]) << "row padding overwritten";
            }
        }
    }
}

// A 517-pixel RGB8 row spans two packed x blocks: its first block uses every inner group, and the
// second block ends with a one-pixel scalar tail. An odd row stride alternates aligned and
// misaligned rows, while the second short image exercises the per-image width bound.
TEST(OpInvert, varshape_rgb8_multi_group_and_tail_correct_output)
{
    constexpr int                         kBatch      = 2;
    constexpr std::array<int, kBatch>     kWidths     = {517, 7};
    constexpr std::array<int, kBatch>     kHeights    = {5, 5};
    constexpr std::array<int32_t, kBatch> kRowStrides = {1553, 23};
    constexpr std::array<size_t, kBatch>  kImageBytes
        = {static_cast<size_t>(kRowStrides[0] * kHeights[0]), static_cast<size_t>(kRowStrides[1] * kHeights[1])};
    constexpr size_t kTotalBytes = kImageBytes[0] + kImageBytes[1];

    auto wrapImage = [](NVCVByte *base, int width, int height, int32_t rowStride)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes           = 1;
        buffer.planes[0].width     = width;
        buffer.planes[0].height    = height;
        buffer.planes[0].rowStride = rowStride;
        buffer.planes[0].basePtr   = base;
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8, buffer});
    };

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kTotalBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kTotalBytes));

    std::vector<uint8_t> srcBytes(kTotalBytes);
    for (size_t i = 0; i < kTotalBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    const std::vector<uint8_t> dstSentinel(kTotalBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kTotalBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), kTotalBytes, cudaMemcpyHostToDevice));

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    size_t                   imageOffset = 0;
    for (int i = 0; i < kBatch; ++i)
    {
        srcImages.push_back(wrapImage(srcPtr + imageOffset, kWidths[i], kHeights[i], kRowStrides[i]));
        dstImages.push_back(wrapImage(dstPtr + imageOffset, kWidths[i], kHeights[i], kRowStrides[i]));
        imageOffset += kImageBytes[i];
    }

    nvcv::ImageBatchVarShape srcBatch(kBatch);
    nvcv::ImageBatchVarShape dstBatch(kBatch);
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, srcBatch, dstBatch));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kTotalBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kTotalBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    imageOffset = 0;
    for (int i = 0; i < kBatch; ++i)
    {
        const size_t rowBytes = static_cast<size_t>(kWidths[i]) * 3;
        for (int y = 0; y < kHeights[i]; ++y)
        {
            const size_t row = imageOffset + static_cast<size_t>(y) * kRowStrides[i];
            for (size_t x = 0; x < rowBytes; ++x)
            {
                ASSERT_EQ(static_cast<uint8_t>(255 - srcBytes[row + x]), dstBytes[row + x])
                    << "image " << i << " row " << y << " byte " << x;
            }
            for (size_t x = rowBytes; x < static_cast<size_t>(kRowStrides[i]); ++x)
            {
                ASSERT_EQ(0xA5, dstBytes[row + x]) << "row padding overwritten";
            }
        }
        imageOffset += kImageBytes[i];
    }
}

// Keep one RGB8 VarShape image fully aligned and give the other two images opposite source/destination
// base alignments. Width 9 retains a packed group and a scalar tail for each row.
TEST(OpInvert, varshape_rgb8_mixed_src_dst_alignment_correct_output)
{
    constexpr int                        kBatch      = 3;
    constexpr int                        kWidth      = 9;
    constexpr int                        kHeight     = 5;
    constexpr int64_t                    kRowBytes   = static_cast<int64_t>(kWidth) * 3;
    constexpr int64_t                    kRowStride  = kRowBytes + 1;
    constexpr auto                       kImageBytes = static_cast<size_t>(kRowStride * kHeight);
    constexpr std::array<size_t, kBatch> kSrcOffsets = {0, kImageBytes, 2 * kImageBytes + 1};
    constexpr std::array<size_t, kBatch> kDstOffsets = {0, kImageBytes + 1, 2 * kImageBytes + 2};
    constexpr size_t                     kBytes      = static_cast<size_t>(kBatch) * kImageBytes + 2;

    auto wrapImage = [=](NVCVByte *base, size_t offset)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes           = 1;
        buffer.planes[0].width     = kWidth;
        buffer.planes[0].height    = kHeight;
        buffer.planes[0].rowStride = kRowStride;
        buffer.planes[0].basePtr   = base + offset;
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8, buffer});
    };

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kBytes));

    std::vector<uint8_t> srcBytes(kBytes);
    for (size_t i = 0; i < kBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 37 + 11) % 251);
    }
    const std::vector<uint8_t> dstSentinel(kBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), kBytes, cudaMemcpyHostToDevice));

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    for (int i = 0; i < kBatch; ++i)
    {
        srcImages.push_back(wrapImage(srcPtr, kSrcOffsets[i]));
        dstImages.push_back(wrapImage(dstPtr, kDstOffsets[i]));
    }

    nvcv::ImageBatchVarShape srcBatch(kBatch);
    nvcv::ImageBatchVarShape dstBatch(kBatch);
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, srcBatch, dstBatch));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    for (int i = 0; i < kBatch; ++i)
    {
        for (int y = 0; y < kHeight; ++y)
        {
            const size_t srcRow = kSrcOffsets[i] + static_cast<size_t>(y * kRowStride);
            const size_t dstRow = kDstOffsets[i] + static_cast<size_t>(y * kRowStride);
            for (int64_t x = 0; x < kRowBytes; ++x)
            {
                ASSERT_EQ(static_cast<uint8_t>(255 - srcBytes[srcRow + x]), dstBytes[dstRow + x])
                    << "image " << i << " row " << y << " byte " << x;
            }
            for (int64_t x = kRowBytes; x < kRowStride; ++x)
            {
                ASSERT_EQ(0xA5, dstBytes[dstRow + x]) << "row padding overwritten";
            }
        }
    }
}

NVCV_TEST_SUITE_P(OpInvertVarShapeTyped, test::ValueList<nvcv::ImageFormat>{nvcv::FMT_U16, nvcv::FMT_F32});

TEST_P(OpInvertVarShapeTyped, correct_output)
{
    nvcv::ImageFormat fmt = GetParam();
    if (ew::BaseKind(fmt) == 2)
    {
        ew::RunVarShapeCorrectTyped<float>(fmt, goldFor(float{}), invoke);
    }
    else
    {
        ew::RunVarShapeCorrectTyped<uint16_t>(fmt, goldFor(uint16_t{}), invoke);
    }
}

// F16 correctness: the kernel computes 1 - x in native half, the reference in FP32 on the
// half-quantized input; kUlps = 1 because the kernel is a single correctly-rounded subtract (one
// rounding step; see HalfTestUtils.hpp). The FP32 gold for out = 1 - x on half input.
constexpr float kInvertF16Ulps = 1.f;

const auto goldInvertF32 = [](float v)
{
    return 1.0f - v;
};

// The single-channel FMT_F16 rows matter: 1-channel F16 must take the scalar kernel that the
// wide-load (ushort4) exclusion in OpInvert.cu falls back to, in both Tensor and VarShape paths.
// clang-format off
NVCV_TEST_SUITE_P(OpInvertF16, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    //   width, height, batch,            format          (dtype / channels)
    {       61,     23,     2,   nvcv::FMT_F16    }, // f16 / 1ch
    {       55,     33,     2,   nvcv::FMT_RGBf16 }, // f16 / 3ch
    {       23,     54,     1,   nvcv::FMT_RGBAf16}, // f16 / 4ch
});

// clang-format on
TEST_P(OpInvertF16, tensor_correct_output)
{
    ew::RunTensorCorrectF16(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                            nvcv::ImageFormat{GetParamValue<3>()}, goldInvertF32, invoke, kInvertF16Ulps);
}

NVCV_TEST_SUITE_P(OpInvertF16VarShape, test::ValueList<nvcv::ImageFormat>{nvcv::FMT_F16, nvcv::FMT_RGBf16});

TEST_P(OpInvertF16VarShape, correct_output)
{
    ew::RunVarShapeCorrectF16(GetParam(), goldInvertF32, invoke, kInvertF16Ulps);
}

// Planar ≡ interleaved parity (fake-planar): native planar output must be byte-for-byte identical
// to the interleaved result on the same data. Uses the shared PlanarParityUtils scaffolding. -----
// clang-format off
NVCV_TEST_SUITE_P(OpInvertPlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {177, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 65,  48, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {101,  80, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    { 66,  49, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

NVCV_TEST_SUITE_P(OpInvertPlanarVarShape,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // exact width exercises the pure wide arm
    {177, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // odd width exercises the outlined scalar tail
    {101,  80, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    { 66,  49, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

// clang-format on
TEST_P(OpInvertPlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invoke(s, src, dst)); });
}

TEST_P(OpInvertPlanarVarShape, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invoke(s, src, dst)); });
}

// Padded-stride tensors: the dense flat fast path must reject padded rows and fall back to the
// layout kernels, which must stay bit-exact vs the CPU gold and must not touch the padding. ------
namespace {

// Check every payload lane equals bound - src (the inverse) and every padding byte is still the
// 0xA5 sentinel, i.e. the fallback kernel wrote the payload correctly and never touched the padding.
template<typename BT>
void VerifyPaddedTensor(const std::vector<uint8_t> &srcBytes, const std::vector<uint8_t> &dstBytes, int batch,
                        int rowsPerSample, int64_t sampleStride, int64_t rowStride, int64_t rowBytes)
{
    for (int n = 0; n < batch; ++n)
    {
        for (int r = 0; r < rowsPerSample; ++r)
        {
            const size_t rowOff = n * sampleStride + r * rowStride;
            for (size_t e = 0; e < rowBytes / sizeof(BT); ++e)
            {
                // memcpy the lanes out of the byte buffers: rowOff need not satisfy alignof(BT)
                // and the payload is raw bytes, so a typed reinterpret_cast read would be UB.
                BT in{};
                BT out{};
                std::memcpy(&in, srcBytes.data() + rowOff + e * sizeof(BT), sizeof(BT));
                std::memcpy(&out, dstBytes.data() + rowOff + e * sizeof(BT), sizeof(BT));
                ASSERT_EQ(goldFor(BT{})(in), out)
                    << "payload mismatch at sample " << n << " row " << r << " elem " << e;
            }
            for (int64_t b = rowBytes; b < rowStride; ++b)
            {
                ASSERT_EQ(0xA5, dstBytes[rowOff + b]) << "padding overwritten at sample " << n << " row " << r;
            }
        }
    }
}

template<typename BT>
void RunPaddedTensor(nvcv::ImageFormat fmt, int width, int height, int batch, int padBytes)
{
    const bool    planar    = fmt.numPlanes() > 1;
    const int     channels  = fmt.numChannels();
    const int64_t elemBytes = sizeof(BT);
    // Packed row (per plane for planar formats) plus deliberate padding.
    const int64_t rowBytes     = width * elemBytes * (planar ? 1 : channels);
    const int64_t rowStride    = rowBytes + padBytes;
    const int64_t planeStride  = rowStride * height;
    const int64_t sampleStride = planeStride * (planar ? channels : 1);
    const size_t  totalBytes   = static_cast<size_t>(sampleStride) * batch;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, totalBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, totalBytes));

    auto wrapTensor = [&](NVCVByte *ptr)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr = ptr;
        if (planar)
        {
            buffer.strides[0] = sampleStride;
            buffer.strides[1] = planeStride;
            buffer.strides[2] = rowStride;
            buffer.strides[3] = elemBytes;
            return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{batch, channels, height, width}, "NCHW"},
                nvcv::DataType{            fmt.planeDataType(0)       },
                buffer
            });
        }
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = rowStride;
        buffer.strides[2] = channels * elemBytes;
        buffer.strides[3] = elemBytes;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{   {batch, height, width, channels}, "NHWC"},
            nvcv::DataType{fmt.planeDataType(0).channelType(0)       },
            buffer
        });
    };

    nvcv::Tensor src = wrapTensor(srcPtr);
    nvcv::Tensor dst = wrapTensor(dstPtr);

    // Fill the whole src allocation (payload + padding) with a deterministic byte pattern and the
    // dst allocation with a sentinel, so untouched padding can be verified after the run.
    std::vector<uint8_t> srcBytes(totalBytes);
    for (size_t i = 0; i < totalBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    if constexpr (std::is_floating_point_v<BT>)
    {
        // Overwrite payload lanes with valid floats (deterministic in [0, 1]). memcpy each value in
        // rather than a typed store so the write stays well-defined on the uint8_t backing storage.
        for (size_t i = 0; i < totalBytes / sizeof(BT); ++i)
        {
            const BT v = static_cast<BT>((i * 37 + 11) % 1000) / BT{1000};
            std::memcpy(srcBytes.data() + i * sizeof(BT), &v, sizeof(BT));
        }
    }
    const std::vector<uint8_t> dstSentinel(totalBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), totalBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), totalBytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(totalBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, totalBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    const int rowsPerSample = height * (planar ? channels : 1);
    VerifyPaddedTensor<BT>(srcBytes, dstBytes, batch, rowsPerSample, sampleStride, rowStride, rowBytes);
}

} // namespace

TEST(OpInvert, tensor_padded_strides_correct_output)
{
    // Interleaved fallbacks (scalar uchar3/uchar4/float3 kernels).
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGB8, 41, 13, 2, 4);
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGBA8, 22, 9, 2, 8);
    RunPaddedTensor<float>(nvcv::FMT_RGBf32, 21, 7, 1, 12);
    // Single-channel interleaved: 8-byte-aligned pad keeps the ushort4 vector path (with padding),
    // odd pad forces the scalar tensor kernel.
    RunPaddedTensor<uint16_t>(nvcv::FMT_U16, 30, 11, 2, 4);
    RunPaddedTensor<uint16_t>(nvcv::FMT_U16, 30, 11, 2, 6);
    // Planar u8 (41-byte packed rows): pad 3 gives 4-byte-aligned strides and keeps the vectorized
    // planar kernel, pad 2 gives odd strides and forces the scalar planar kernel.
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGB8p, 41, 13, 2, 3);
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGB8p, 41, 13, 2, 2);
    // Planar f32 (84-byte packed rows): pad 12 gives 16-byte-aligned strides and keeps the
    // vectorized planar kernel, pad 8 forces its scalar fallback.
    RunPaddedTensor<float>(nvcv::FMT_RGBf32p, 21, 7, 2, 12);
    RunPaddedTensor<float>(nvcv::FMT_RGBf32p, 21, 7, 2, 8);
}

// Signed integer and 64-bit float dtypes: call RunTensorCorrect directly since
// RunTensorCorrectDispatch only covers u8/u16/f32 (used by tensor_correct_output above).
TEST(OpInvert, tensor_correct_output_signed_and_f64)
{
    ew::RunTensorCorrect<short>(66, 55, 1, nvcv::FMT_S16, goldFor(short{}), invoke);
    ew::RunTensorCorrect<int>(47, 31, 2, nvcv::FMT_S32, goldFor(int{}), invoke);
    // S64 1ch — no predefined FMT_S64 constant; construct from the image-format macro
    ew::RunTensorCorrect<long long>(
        33, 27, 1, nvcv::ImageFormat{NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X64)},
        goldFor(static_cast<long long>(0)), invoke);
    ew::RunTensorCorrect<double>(41, 19, 2, nvcv::FMT_F64, goldFor(double{}), invoke);
}

TEST(OpInvert, signed16_packed_saturation_exhaustive)
{
    constexpr int      width  = 256;
    constexpr int      height = 256;
    std::vector<short> input(width * height);
    std::vector<short> expected(input.size());
    std::vector<short> actual(input.size());
    for (int i = 0; i < width * height; ++i)
    {
        const int value = i + std::numeric_limits<short>::lowest();
        input[i]        = static_cast<short>(value);
        expected[i]     = static_cast<short>(std::min<int>(32767, 32767 - value));
    }
    nvcv::Tensor src     = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_S16);
    nvcv::Tensor dst     = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_S16);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess && dstAccess);
    constexpr size_t rowBytes = width * sizeof(short);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->basePtr(), srcAccess->rowStride(), input.data(), rowBytes, rowBytes,
                                        height, cudaMemcpyHostToDevice));
    invoke(nullptr, src, dst);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(actual.data(), rowBytes, dstData->basePtr(), dstAccess->rowStride(), rowBytes,
                                        height, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
}

static void ExpectInvertF64(double input, double actual)
{
    const double expected = 1. - input;
    if (std::isnan(expected))
        EXPECT_TRUE(std::isnan(actual));
    else
        EXPECT_EQ(0, std::memcmp(&expected, &actual, sizeof(actual)));
}

static void RunInvertF64TensorCase(const std::vector<double> &values, int channels, bool planar)
{
    const int width = static_cast<int>(values.size()) | 1;

    SCOPED_TRACE(testing::Message() << channels << ":" << planar);
    const nvcv::TensorShape shape = planar ? nvcv::TensorShape{{1, channels, 1, width}, "NCHW"}
                                      : nvcv::TensorShape{{1, 1, width, channels}, "NHWC"};
    nvcv::Tensor src(shape, nvcv::TYPE_F64);
    nvcv::Tensor dst(shape, nvcv::TYPE_F64);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto         access  = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(access);
    const size_t         bytes = planar ? access->planeStride() * channels : access->rowStride();
    std::vector<uint8_t> input(bytes, 0);
    std::vector<uint8_t> output(bytes);
    const auto           offset = [planar, &access, channels](int x, int c)
    {
        return planar ? c * access->planeStride() + x * sizeof(double) : (x * channels + c) * sizeof(double);
    };
    for (int c = 0; c < channels; ++c)
        for (int x = 0; x < width; ++x)
            std::memcpy(input.data() + offset(x, c), &values[(x + c) % values.size()], sizeof(double));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice));
    ASSERT_NO_THROW(invoke(nullptr, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), dstData->basePtr(), bytes, cudaMemcpyDeviceToHost));
    for (int c = 0; c < channels; ++c)
        for (int x = 0; x < width; ++x)
        {
            double actual;
            std::memcpy(&actual, output.data() + offset(x, c), sizeof(actual));
            ExpectInvertF64(values[(x + c) % values.size()], actual);
        }
}

TEST(OpInvert, f64_exact_half_interval_and_native_fallback)
{
    const double        inf = std::numeric_limits<double>::infinity();
    std::vector<double> values{-inf,
                               -1.,
                               -0.,
                               0.,
                               std::numeric_limits<double>::denorm_min(),
                               0.25,
                               0.5,
                               1.,
                               std::nextafter(1., inf),
                               2.,
                               std::numeric_limits<double>::max(),
                               inf,
                               std::numeric_limits<double>::quiet_NaN()};
    for (int exponent = 1; exponent <= 53; ++exponent)
    {
        const double value = 1. - std::ldexp(1., -exponent);
        values.push_back(std::nextafter(value, 0.));
        values.push_back(value);
        values.push_back(std::nextafter(value, inf));
    }
    const int width = static_cast<int>(values.size()) | 1;

    const auto cases = test::Combine(test::ValueList<int>{1, 3, 4}, test::ValueList<bool>{false, true});
    for (const auto &[channels, planar] : cases)
        ASSERT_NO_FATAL_FAILURE(RunInvertF64TensorCase(values, channels, planar));

    nvcv::Image         srcImage({width, 1}, nvcv::FMT_F64);
    nvcv::Image         dstImage({width, 1}, nvcv::FMT_F64);
    auto                srcData = srcImage.exportData<nvcv::ImageDataStridedCuda>();
    auto                dstData = dstImage.exportData<nvcv::ImageDataStridedCuda>();
    std::vector<double> input(width);
    std::vector<double> output(width);
    for (int x = 0; x < width; ++x) input[x] = values[x % values.size()];
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(srcData->plane(0).basePtr, input.data(), width * sizeof(double), cudaMemcpyHostToDevice));
    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);
    src.pushBack(srcImage);
    dst.pushBack(dstImage);
    ASSERT_NO_THROW(invoke(nullptr, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(output.data(), dstData->plane(0).basePtr, width * sizeof(double), cudaMemcpyDeviceToHost));
    for (int x = 0; x < width; ++x) ExpectInvertF64(input[x], output[x]);
}

TEST(OpInvert, signed16_vector_body_tails_and_padding)
{
    for (auto fmt : {nvcv::FMT_S16, nvcv::FMT_RGBf16.dataKind(nvcv::DataKind::SIGNED),
                     nvcv::FMT_RGBAf16.dataKind(nvcv::DataKind::SIGNED)})
    {
        ew::RunTensorCorrect<short>(256, 128, 2, fmt, goldFor(short{}), invoke);
        ew::RunTensorCorrect<short>(131, 17, 2, fmt, goldFor(short{}), invoke);
        ew::RunVarShapeCorrectTyped<short>(fmt, goldFor(short{}), invoke);
    }
    for (auto fmt :
         {nvcv::FMT_S16, nvcv::FMT_RGBf16.dataKind(nvcv::DataKind::SIGNED),
          nvcv::FMT_RGBAf16.dataKind(nvcv::DataKind::SIGNED), nvcv::FMT_RGBf16p.dataKind(nvcv::DataKind::SIGNED),
          nvcv::FMT_RGBAf16p.dataKind(nvcv::DataKind::SIGNED)})
    {
        for (int pad : {0, 2, 10, 16})
        {
            RunPaddedTensor<short>(fmt, 131, 17, 2, pad);
        }
    }
}

TEST(OpInvert, double_vector_body_tails_and_padding)
{
    const nvcv::ImageFormat rgb{NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZ1, ASSOCIATED, X64_Y64_Z64)};
    const nvcv::ImageFormat rgba{
        NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZW, ASSOCIATED, X64_Y64_Z64_W64)};
    const nvcv::ImageFormat rgbp{
        NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, FLOAT, XYZ0, ASSOCIATED, X64, X64, X64)};
    const nvcv::ImageFormat rgbap{
        NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, FLOAT, XYZW, ASSOCIATED, X64, X64, X64, X64)};
    for (auto fmt : {nvcv::FMT_F64, rgb, rgba})
    {
        ew::RunTensorCorrect<double>(256, 17, 2, fmt, goldFor(double{}), invoke);
        ew::RunTensorCorrect<double>(131, 17, 2, fmt, goldFor(double{}), invoke);
        ew::RunVarShapeCorrectTyped<double>(fmt, goldFor(double{}), invoke);
    }
    for (auto fmt : {nvcv::FMT_F64, rgb, rgba, rgbp, rgbap})
    {
        for (int pad : {0, 8, 24})
        {
            RunPaddedTensor<double>(fmt, 131, 17, 2, pad);
        }
    }
}

TEST(OpInvert, dense_u8_tensor_tail_correct_output)
{
    // A 15-byte packed row declines the element-stream carrier while the 45-byte tensor remains
    // eligible for the aligned dense path and leaves a five-byte tail.
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGB8, 5, 3, 1, 0);
}

// A packed planar tensor with a 2-byte base offset declines both wide paths while the scalar
// planar wrapper remains valid.
TEST(OpInvert, tensor_planar_u8_packed_unaligned_scalar_correct_output)
{
    constexpr int    kChannels        = 3;
    constexpr int    kWidth           = 16;
    constexpr int    kHeight          = 5;
    constexpr size_t kBaseOffset      = 2;
    constexpr size_t kRowBytes        = kWidth;
    constexpr size_t kPlaneBytes      = kRowBytes * kHeight;
    constexpr size_t kPayloadBytes    = kPlaneBytes * kChannels;
    constexpr size_t kAllocationBytes = kBaseOffset + kPayloadBytes;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kAllocationBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kAllocationBytes));

    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    srcBuffer.basePtr    = srcPtr + kBaseOffset;
    srcBuffer.strides[0] = kPayloadBytes;
    srcBuffer.strides[1] = kPlaneBytes;
    srcBuffer.strides[2] = kRowBytes;
    srcBuffer.strides[3] = 1;
    auto dstBuffer       = srcBuffer;
    dstBuffer.basePtr    = dstPtr + kBaseOffset;

    const nvcv::TensorShape shape{
        {1, kChannels, kHeight, kWidth},
        "NCHW"
    };
    nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, srcBuffer});
    nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, dstBuffer});

    std::vector<uint8_t> srcBytes(kAllocationBytes);
    for (size_t i = 0; i < srcBytes.size(); ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 29 + 13) % 251);
    }
    const std::vector<uint8_t> dstSentinel(kAllocationBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kAllocationBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), kAllocationBytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kAllocationBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kAllocationBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    ASSERT_EQ(0xA5, dstBytes[0]);
    ASSERT_EQ(0xA5, dstBytes[1]);
    for (size_t i = 0; i < kPayloadBytes; ++i)
    {
        ASSERT_EQ(static_cast<uint8_t>(255 - srcBytes[kBaseOffset + i]), dstBytes[kBaseOffset + i])
            << "payload byte " << i;
    }
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpInvert_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_RGB8,  nvcv::FMT_RGB8p}, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,  nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpInvert_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invoke);
}

// 2-channel input (outside the 1/3/4 matrix) is rejected.
TEST(OpInvert_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, invoke, 16, 16);
}

// grid.y launch bound ---------------------------------------------------------------------------
namespace {

// A rank-4 NHWC U8 tensor of `height` single-pixel rows, `rowStride` bytes apart, wrapped over a
// caller-owned allocation. This is what ValidateTensorBufferStrided admits for a non-packed row
// axis: any stride >= the 1-byte minimum pitch, including a stride that leaves the rows padded.
nvcv::Tensor WrapTallU8Tensor(NVCVByte *base, int64_t height, int64_t rowStride)
{
    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr    = base;
    buffer.strides[0] = height * rowStride;
    buffer.strides[1] = rowStride;
    buffer.strides[2] = 1;
    buffer.strides[3] = 1;
    return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{1, height, 1, 1}, "NHWC"},
        nvcv::DataType{    nvcv::TYPE_U8       },
        buffer
    });
}

} // namespace

// Regression (bug, not hardening): rows map straight onto grid.y, which is 16-bit, and nothing
// bounded it. With a padded row stride every fast path declines -- the element stream because
// 1000000 regions exceed 65535, the dense path because the row is not packed -- and the tensor
// kernels then launched grid.y = 250000, which the driver rejects with
// cudaErrorInvalidConfiguration. No path can serve this tensor, so it is now an API error.
TEST(OpInvert_Negative, tensor_height_exceeds_grid_y_limit)
{
    constexpr int64_t kHeight    = 1000000; // 250000 blocks of 4 rows, well past the 65535 cap
    constexpr int64_t kRowStride = 4;       // padded, so no row-folding path can take it
    constexpr size_t  kBytes     = static_cast<size_t>(kHeight) * kRowStride;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kBytes));

    nvcv::Tensor src = WrapTallU8Tensor(srcPtr, kHeight, kRowStride);
    nvcv::Tensor dst = WrapTallU8Tensor(dstPtr, kHeight, kRowStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { invoke(stream, src, dst); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

// Companion to the above, and the reason the bound cannot be checked up front: the same height
// with a *packed* row is folded into a single contiguous run by the element stream, launches a
// legal grid, and must keep working. A blanket height rejection would break it.
TEST(OpInvert, tensor_tall_packed_rows_still_run)
{
    constexpr int64_t kHeight = 1000000; // above the scalar kernels' 262140-row ceiling
    constexpr auto    kBytes  = static_cast<size_t>(kHeight);

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kBytes));

    std::vector<uint8_t> srcBytes(kBytes);
    for (size_t i = 0; i < kBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, kBytes));

    nvcv::Tensor src = WrapTallU8Tensor(srcPtr, kHeight, 1);
    nvcv::Tensor dst = WrapTallU8Tensor(dstPtr, kHeight, 1);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    std::vector<uint8_t> expected(kBytes);
    for (size_t i = 0; i < kBytes; ++i)
    {
        expected[i] = static_cast<uint8_t>(255 - srcBytes[i]);
    }
    EXPECT_EQ(expected, dstBytes);
}

// Wrapped (caller-owned) memory: alignment and stride contracts --------------------------------
namespace {

// Wrap `kPlanes` U8 planes of `kWidth` x `kHeight` at `rowStride` bytes per row over caller-owned
// device memory. ImageWrapData requires only a non-null plane pointer and a row stride of at least
// one packed row (nvcv::priv::ImageWrapData::doValidateData), so both the plane base and the row
// stride below are deliberately odd. Their changing alignment exercises both kernel arms.
constexpr int     kMisWidth      = 65;
constexpr int     kMisHeight     = 5;
constexpr int     kMisPlanes     = 3;
constexpr int64_t kMisRowStride  = kMisWidth + 2; // 67: odd, and > the packed row
constexpr int64_t kMisPlaneBytes = kMisRowStride * kMisHeight;
constexpr int64_t kMisImageBytes = kMisPlaneBytes * kMisPlanes;
constexpr int64_t kMisBaseOffset = 1; // odd plane base pointer

// cudaMalloc returns bases aligned to at least 256 bytes, so these offsets determine uchar4 alignment.
static_assert(kMisWidth % 4 != 0, "ragged width must exercise a scalar tail");
static_assert(kMisRowStride > kMisWidth, "wrapped rows must be padded");
static_assert(kMisBaseOffset % alignof(uchar4) != 0, "the first row must exercise the scalar arm");
static_assert((kMisBaseOffset + kMisRowStride) % alignof(uchar4) == 0, "the second row must exercise the wide arm");

nvcv::Image WrapMisalignedPlanarImage(NVCVByte *base)
{
    nvcv::ImageDataStridedCuda::Buffer buf{};
    buf.numPlanes = kMisPlanes;
    for (int p = 0; p < kMisPlanes; ++p)
    {
        buf.planes[p].width     = kMisWidth;
        buf.planes[p].height    = kMisHeight;
        buf.planes[p].rowStride = kMisRowStride;
        buf.planes[p].basePtr   = base + p * kMisPlaneBytes;
    }
    return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8p, buf});
}

} // namespace

// Regression (bug, not hardening): the vectorized 1-byte planar var-shape kernel used to issue an
// unguarded uchar4 load at `src.ptr(batch, plane, y, cx)`. With an odd plane base and an odd row
// stride -- which ImageWrapData accepts -- some rows are misaligned, and the launch failed with a
// sticky cudaErrorMisalignedAddress. The kernel now tests each row's alignment, using its wide arm
// for aligned rows and its scalar arm otherwise. Posterize and Solarize shared the defect through
// the same shared kernel.
TEST(OpInvert, varshape_planar_wrapped_misaligned_images_correct_output)
{
    constexpr int    kBatch = 2;
    constexpr size_t kBytes = static_cast<size_t>(kMisBaseOffset) + kMisImageBytes * kBatch;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kBytes));

    std::vector<uint8_t> srcBytes(kBytes);
    for (size_t i = 0; i < kBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    const std::vector<uint8_t> dstSentinel(kBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), kBytes, cudaMemcpyHostToDevice));

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    for (int i = 0; i < kBatch; ++i)
    {
        srcImages.push_back(WrapMisalignedPlanarImage(srcPtr + kMisBaseOffset + i * kMisImageBytes));
        dstImages.push_back(WrapMisalignedPlanarImage(dstPtr + kMisBaseOffset + i * kMisImageBytes));
    }

    nvcv::ImageBatchVarShape srcBatch(kBatch);
    nvcv::ImageBatchVarShape dstBatch(kBatch);
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, srcBatch, dstBatch));
    // The pre-fix failure mode is here, not in the launch: a misaligned wide load faults
    // asynchronously and surfaces at the synchronize.
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    auto verifyRow = [&srcBytes, &dstBytes](int i, int p, int y)
    {
        const size_t row
            = static_cast<size_t>(kMisBaseOffset) + i * kMisImageBytes + p * kMisPlaneBytes + y * kMisRowStride;
        for (int x = 0; x < kMisWidth; ++x)
        {
            ASSERT_EQ(static_cast<uint8_t>(255 - srcBytes[row + x]), dstBytes[row + x])
                << "image " << i << " plane " << p << " row " << y << " col " << x;
        }
        for (int64_t b = kMisWidth; b < kMisRowStride; ++b)
        {
            ASSERT_EQ(0xA5, dstBytes[row + b]) << "row padding overwritten";
        }
    };

    for (int i = 0; i < kBatch; ++i)
    {
        for (int p = 0; p < kMisPlanes; ++p)
        {
            for (int y = 0; y < kMisHeight; ++y)
            {
                verifyRow(i, p, y);
            }
        }
    }
}

// Regression (bug, not hardening): the 32-bit-stride span check used to read only
// sampleStride() * numSamples(), and TensorDataAccessStridedImagePlanar::sampleStride() is 0 for a
// layout with no sample axis, so the check was identically 0 for HWC and CHW. A rank-3 HWC tensor
// whose rows span more than INT32_MAX bytes was accepted and then wrapped its int32 offsets,
// addressing memory outside the tensor. It is now an overflow error.
//
// The tensor is wrapped over a token allocation on purpose: the rejection happens in host-side
// validation, before any kernel is launched, so no memory is ever addressed. Allocating the 2.5 GB
// the shape describes would make the test unrunnable on small devices for no added coverage.
TEST(OpInvert_Negative, tensor_hwc_row_span_exceeds_int32)
{
    constexpr int64_t kSide = 50000; // 50000 * 50000 == 2.5e9 > INT32_MAX

    NVCVByte *ptr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&ptr, 1024));

    auto wrap = [&]
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr;
        buffer.strides[0] = kSide; // row stride
        buffer.strides[1] = 1;     // column stride (packed)
        buffer.strides[2] = 1;     // channel stride
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{kSide, kSide, 1}, "HWC"},
            nvcv::DataType{    nvcv::TYPE_U8      },
            buffer
        });
    };

    nvcv::Tensor src = wrap();
    nvcv::Tensor dst = wrap();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&] { invoke(stream, src, dst); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    ASSERT_EQ(cudaSuccess, cudaFree(ptr));
}

// Both rejections below are made by same_shape::ValidateSrcDstTensors, which is now the single
// place four operators' tensor contract is decided; the bodies live in the shared harness and
// TestOpPosterize/TestOpSolarize/TestOpAdjustSharpness assert the same two through their own entry
// points. See ElementwiseOpHarness.hpp for why each shape is reachable and what it used to do.
TEST(OpInvert_Negative, tensor_padded_column_stride)
{
    ew::ExpectPaddedColumnStrideTensorRejected(invoke);
}

TEST(OpInvert_Negative, tensor_multi_channel_dtype)
{
    ew::ExpectMultiChannelDtypeTensorRejected(invoke);
}

// The complement of the int32 span check: an axis of extent 1 is only ever indexed at 0, so its
// stride is never added to any address, and nvcv::priv::ValidateTensorBufferStrided leaves such a
// stride entirely unconstrained. Counting it at full stride rejected this NCHW {1, 1, H, W} tensor
// with ERROR_OVERFLOW even though every address it forms is tiny. It must be accepted, and
// transformed correctly.
TEST(OpInvert, tensor_unused_extent1_stride_not_rejected)
{
    constexpr int64_t kSide       = 4;
    constexpr int64_t kRowStride  = kSide;
    constexpr int64_t kHugeStride = static_cast<int64_t>(4) * 1000 * 1000 * 1000; // > INT32_MAX, on an axis of extent 1
    constexpr size_t  kBytes      = static_cast<size_t>(kSide) * kRowStride;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kBytes));

    std::vector<uint8_t> srcBytes(kBytes);
    for (size_t i = 0; i < kBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, kBytes));

    auto wrap = [](NVCVByte *ptr)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr;
        buffer.strides[0] = kHugeStride; // N, extent 1: unconstrained and never added
        buffer.strides[1] = kHugeStride; // C, extent 1: likewise
        buffer.strides[2] = kRowStride;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, 1, kSide, kSide}, "NCHW"},
            nvcv::DataType{       nvcv::TYPE_U8       },
            buffer
        });
    };

    nvcv::Tensor src = wrap(srcPtr);
    nvcv::Tensor dst = wrap(dstPtr);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    std::vector<uint8_t> expected(kBytes);
    for (size_t i = 0; i < kBytes; ++i)
    {
        expected[i] = static_cast<uint8_t>(255 - srcBytes[i]);
    }
    EXPECT_EQ(expected, dstBytes);
}

// Negative signed inputs saturate rather than overflowing the subtraction.
TEST(OpInvert, signed_extrema_tensor_and_varshape)
{
    const auto run = []<typename T>(T, nvcv::ImageFormat format)
    {
        const T              hi = std::numeric_limits<T>::max();
        const std::vector<T> input{std::numeric_limits<T>::lowest(), T{-1}, T{0}, hi};
        const std::vector<T> expected{hi, hi, hi, T{0}};
        const size_t         bytes   = input.size() * sizeof(T);
        nvcv::Tensor         src     = nvcv::util::CreateTensor(1, 4, 1, format);
        nvcv::Tensor         dst     = nvcv::util::CreateTensor(1, 4, 1, format);
        auto                 srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        auto                 dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(srcData && dstData);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice));
        invoke(nullptr, src, dst);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
        std::vector<T> actual(input.size());
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstData->basePtr(), bytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected, actual);

        nvcv::Image srcImage({4, 1}, format);
        nvcv::Image dstImage({4, 1}, format);
        auto        srcImageData = srcImage.exportData<nvcv::ImageDataStridedCuda>();
        auto        dstImageData = dstImage.exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(srcImageData && dstImageData);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(srcImageData->plane(0).basePtr, input.data(), bytes, cudaMemcpyHostToDevice));
        nvcv::ImageBatchVarShape srcBatch(1);
        nvcv::ImageBatchVarShape dstBatch(1);
        srcBatch.pushBack(srcImage);
        dstBatch.pushBack(dstImage);
        invoke(nullptr, srcBatch, dstBatch);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(actual.data(), dstImageData->plane(0).basePtr, bytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected, actual);
    };
    run(int16_t{}, nvcv::FMT_S16);
    run(int32_t{}, nvcv::FMT_S32);
    run(int64_t{}, nvcv::ImageFormat{NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X64)});
}

static void RunInvertHalf4AlignmentCase(bool varShape, int srcOffset, int dstOffset)
{
    constexpr int kWidth    = 37;
    constexpr int kHeight   = 5;
    constexpr int kBatch    = 2;
    constexpr int kChannels = 4;
    constexpr int kRowBytes = kWidth * kChannels * sizeof(__half);

    SCOPED_TRACE(testing::Message() << "varShape=" << varShape << " srcOffset=" << srcOffset
                                    << " dstOffset=" << dstOffset);
    const int            srcStride = kRowBytes + 16 + srcOffset;
    const int            dstStride = kRowBytes + 16 + dstOffset;
    const int            srcSample = srcStride * kHeight;
    const int            dstSample = dstStride * kHeight;
    std::vector<uint8_t> input(srcSample * kBatch + 8, 0x5A);
    std::vector<uint8_t> expected(dstSample * kBatch + 8, 0xA5);
    for (int n = 0; n < kBatch; ++n)
    {
        for (int y = 0; y < kHeight; ++y)
        {
            for (int x = 0; x < kWidth * kChannels; ++x)
            {
                const float  value = float((n * 19 + y * 7 + x) % 33 - 16) / 8.f;
                const __half in    = __float2half_rn(value);
                const __half gold  = __float2half_rn(1.f - value);
                std::memcpy(input.data() + srcOffset + n * srcSample + y * srcStride + x * sizeof(in), &in, sizeof(in));
                std::memcpy(expected.data() + dstOffset + n * dstSample + y * dstStride + x * sizeof(gold), &gold,
                            sizeof(gold));
            }
        }
    }
    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, input.size()));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, expected.size()));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), input.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, expected.size()));
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    if (varShape)
    {
        auto wrapImage = [](NVCVByte *ptr, int stride)
        {
            nvcv::ImageDataStridedCuda::Buffer buffer{};
            buffer.numPlanes           = 1;
            buffer.planes[0].width     = kWidth;
            buffer.planes[0].height    = kHeight;
            buffer.planes[0].rowStride = stride;
            buffer.planes[0].basePtr   = ptr;
            return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGBAf16, buffer});
        };
        nvcv::ImageBatchVarShape src(kBatch);
        nvcv::ImageBatchVarShape dst(kBatch);
        for (int n = 0; n < kBatch; ++n)
        {
            src.pushBack(wrapImage(srcPtr + srcOffset + n * srcSample, srcStride));
            dst.pushBack(wrapImage(dstPtr + dstOffset + n * dstSample, dstStride));
        }
        EXPECT_NO_THROW(invoke(stream, src, dst));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }
    else
    {
        auto wrapTensor = [](NVCVByte *ptr, int stride, int sample)
        {
            nvcv::TensorDataStridedCuda::Buffer buffer{};
            buffer.basePtr    = ptr;
            buffer.strides[0] = sample;
            buffer.strides[1] = stride;
            buffer.strides[2] = kChannels * sizeof(__half);
            buffer.strides[3] = sizeof(__half);
            return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{kBatch, kHeight, kWidth, kChannels}, "NHWC"},
                nvcv::TYPE_F16, buffer
            });
        };
        auto src = wrapTensor(srcPtr + srcOffset, srcStride, srcSample);
        auto dst = wrapTensor(dstPtr + dstOffset, dstStride, dstSample);
        EXPECT_NO_THROW(invoke(stream, src, dst));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    std::vector<uint8_t> actual(expected.size());
    std::vector<uint8_t> unchanged(input.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstPtr, actual.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(unchanged.data(), srcPtr, unchanged.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
    EXPECT_EQ(expected, actual);
    EXPECT_EQ(input, unchanged);
}

TEST(OpInvert, half4_wrapped_alignment_pairs)
{
    const auto cases
        = test::Combine(test::ValueList<bool>{false, true}, test::ValueList<int>{0, 2}, test::ValueList<int>{0, 2});
    for (const auto &[varShape, srcOffset, dstOffset] : cases)
        ASSERT_NO_FATAL_FAILURE(RunInvertHalf4AlignmentCase(varShape, srcOffset, dstOffset));
}

// A rank-3 HWC tensor has one implicit sample and therefore a zero sample stride. An odd base
// offset declines the dense and wide-load paths while the scalar RGB8 tensor wrapper remains valid.
TEST(OpInvert, tensor_hwc_u8_sample_stride_zero_correct_output)
{
    constexpr int    kWidth           = 5;
    constexpr int    kHeight          = 3;
    constexpr int    kChannels        = 3;
    constexpr size_t kBaseOffset      = 1;
    constexpr size_t kRowBytes        = static_cast<size_t>(kWidth) * kChannels;
    constexpr size_t kPayloadBytes    = kRowBytes * kHeight;
    constexpr size_t kAllocationBytes = kBaseOffset + kPayloadBytes;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kAllocationBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kAllocationBytes));

    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    srcBuffer.basePtr    = srcPtr + kBaseOffset;
    srcBuffer.strides[0] = kRowBytes;
    srcBuffer.strides[1] = kChannels;
    srcBuffer.strides[2] = 1;
    auto dstBuffer       = srcBuffer;
    dstBuffer.basePtr    = dstPtr + kBaseOffset;

    const nvcv::TensorShape shape{
        {kHeight, kWidth, kChannels},
        "HWC"
    };
    nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, srcBuffer});
    nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, dstBuffer});

    std::vector<uint8_t> srcBytes(kAllocationBytes);
    for (size_t i = 0; i < srcBytes.size(); ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 37 + 11) % 251);
    }
    const std::vector<uint8_t> dstSentinel(kAllocationBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), kAllocationBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), kAllocationBytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kAllocationBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kAllocationBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    ASSERT_EQ(0xA5, dstBytes[0]);
    for (size_t i = 0; i < kPayloadBytes; ++i)
    {
        ASSERT_EQ(static_cast<uint8_t>(255 - srcBytes[kBaseOffset + i]), dstBytes[kBaseOffset + i])
            << "payload byte " << i;
    }
}

// A U16 VarShape image that needs one more than the 65535 legal grid.y blocks must fall through
// the serial vector path and be rejected by the scalar launch guard before any kernel is submitted.
TEST(OpInvert_Negative, varshape_u16_height_exceeds_grid_y_limit)
{
    constexpr int kHeight = 4 * 65535 + 1;

    nvcv::Image              srcImage({1, kHeight}, nvcv::FMT_U16);
    nvcv::Image              dstImage({1, kHeight}, nvcv::FMT_U16);
    nvcv::ImageBatchVarShape srcBatch(1);
    nvcv::ImageBatchVarShape dstBatch(1);
    srcBatch.pushBack(srcImage);
    dstBatch.pushBack(dstImage);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { invoke(stream, srcBatch, dstBatch); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpInvert_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaInvertCreate(nullptr));
}
