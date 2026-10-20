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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpPosterize.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

namespace {

// Independent CPU gold: keep the top `bits` bits of each channel value, zero the rest:
//   out = in & ~((1 << (W - bits)) - 1), where W is the type bit width. Mirrors the documented
// oracle (torchvision/PIL posterize, generalized to 16-bit) and is computed independently of the
// kernel so the bit-exact EXPECT_EQ is a real regression check.
template<typename T>
T PosterizeScalarGold(T v, int bits)
{
    constexpr auto W = static_cast<int>(sizeof(T) * 8);
    uint32_t       mask;
    if (bits <= 0)
    {
        mask = 0u;
    }
    else if (bits >= W)
    {
        mask = ~0u;
    }
    else
    {
        mask = ~((1u << (W - bits)) - 1u);
    }
    return static_cast<T>(v & static_cast<T>(mask));
}

// Generic invoker pinning a given bit count (Posterize is integer-only, so the same bits value is
// used for u8 and u16).
auto invokeBits(int bits)
{
    return [bits](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::Posterize op;
        op(s, in, out, bits);
    };
}

template<typename BT>
BT PosterizePaddedValue(uint64_t ordinal, bool firstRow)
{
    if (firstRow && ordinal == 0)
    {
        return static_cast<BT>(0);
    }
    if (firstRow && ordinal == 1)
    {
        return ew::Bound<BT>();
    }
    return static_cast<BT>(ordinal * 37U + 11U);
}

template<typename BT>
void RunPosterizePaddedTensor(nvcv::ImageFormat fmt, int width, int height, int batch, int padBytes, int bits)
{
    const bool    planar        = fmt.numPlanes() > 1;
    const int     channels      = fmt.numChannels();
    const int64_t elemBytes     = sizeof(BT);
    const int64_t rowElems      = static_cast<int64_t>(width) * (planar ? 1 : channels);
    const int64_t rowBytes      = rowElems * elemBytes;
    const int64_t rowStride     = rowBytes + padBytes;
    const int64_t planeStride   = rowStride * height;
    const int64_t sampleStride  = planeStride * (planar ? channels : 1);
    const int     rowsPerSample = height * (planar ? channels : 1);
    const size_t  totalBytes    = static_cast<size_t>(sampleStride) * batch;

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

    std::vector<uint8_t> srcBytes(totalBytes, 0x3D);
    for (int row = 0; row < batch * rowsPerSample; ++row)
    {
        const int  n         = row / rowsPerSample;
        const int  r         = row % rowsPerSample;
        const auto rowOffset = static_cast<size_t>(n * sampleStride + r * rowStride);
        for (int64_t e = 0; e < rowElems; ++e)
        {
            const uint64_t ordinal = (static_cast<uint64_t>(n) * rowsPerSample + r) * rowElems + e;
            const auto     value   = PosterizePaddedValue<BT>(ordinal, row == 0);
            std::memcpy(srcBytes.data() + rowOffset + static_cast<size_t>(e * elemBytes), &value, elemBytes);
        }
    }
    const std::vector<uint8_t> dstSentinel(totalBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), totalBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), totalBytes, cudaMemcpyHostToDevice));

    nvcv::Tensor src = wrapTensor(srcPtr);
    nvcv::Tensor dst = wrapTensor(dstPtr);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invokeBits(bits)(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(totalBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, totalBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    for (int row = 0; row < batch * rowsPerSample; ++row)
    {
        const int  n         = row / rowsPerSample;
        const int  r         = row % rowsPerSample;
        const auto rowOffset = static_cast<size_t>(n * sampleStride + r * rowStride);
        for (int64_t e = 0; e < rowElems; ++e)
        {
            BT           in{};
            BT           out{};
            const size_t offset = rowOffset + static_cast<size_t>(e * elemBytes);
            std::memcpy(&in, srcBytes.data() + offset, elemBytes);
            std::memcpy(&out, dstBytes.data() + offset, elemBytes);
            ASSERT_EQ(PosterizeScalarGold<BT>(in, bits), out)
                << "payload mismatch at sample " << n << " row " << r << " element " << e;
        }
        for (int64_t b = rowBytes; b < rowStride; ++b)
        {
            ASSERT_EQ(0xA5, dstBytes[rowOffset + static_cast<size_t>(b)])
                << "padding overwritten at sample " << n << " row " << r;
        }
    }
}

} // namespace

// Tensor correctness over the declared dtype × channel matrix (u8 / u16 only) -------------------
// clang-format off
NVCV_TEST_SUITE_P(OpPosterize, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    //   width, height, batch,          format        (dtype / channels)
    {       66,     55,     1,   nvcv::FMT_U8    }, // u8  / 1ch
    {      123,     67,     3,   nvcv::FMT_RGB8  }, // u8  / 3ch
    {       42,     53,     4,   nvcv::FMT_RGBA8 }, // u8  / 4ch
    {       80,     40,     2,   nvcv::FMT_U16   }, // u16 / 1ch
});

// clang-format on
TEST_P(OpPosterize, tensor_correct_output)
{
    const int               width  = GetParamValue<0>();
    const int               height = GetParamValue<1>();
    const int               batch  = GetParamValue<2>();
    const nvcv::ImageFormat fmt{GetParamValue<3>()};
    const int               bits = 4;

    // Posterize is integer-only, so dispatch directly over u8 / u16 (no f32 branch).
    if (ew::BaseKind(fmt) == 1)
    {
        ew::RunTensorCorrect<uint16_t>(
            width, height, batch, fmt, [](uint16_t v) { return PosterizeScalarGold<uint16_t>(v, 4); },
            invokeBits(bits));
    }
    else
    {
        ew::RunTensorCorrect<uint8_t>(
            width, height, batch, fmt, [](uint8_t v) { return PosterizeScalarGold<uint8_t>(v, 4); }, invokeBits(bits));
    }
}

// VarShape correctness: bit-exact vs the CPU gold ------------------------------------------------
TEST(OpPosterize, varshape_correct_output)
{
    ew::RunVarShapeCorrect([](uint8_t v) { return PosterizeScalarGold<uint8_t>(v, 3); }, invokeBits(3));
}

TEST(OpPosterize, varshape_u16_correct_output)
{
    ew::RunVarShapeCorrectTyped<uint16_t>(
        nvcv::FMT_U16, [](uint16_t v) { return PosterizeScalarGold<uint16_t>(v, 9); }, invokeBits(9));
}

// Planar ≡ interleaved parity (fake-planar) -----------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpPosterizePlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 64,  48, 1, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
});

NVCV_TEST_SUITE_P(OpPosterizePlanarVarShape,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    {177, 113, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    {178, 113, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    {1031, 5, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
});

// clang-format on
TEST_P(OpPosterizePlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeBits(3)(s, src, dst)); });
}

TEST_P(OpPosterizePlanarVarShape, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeBits(3)(s, src, dst)); });
}

namespace {

constexpr int     kWrappedWidth      = 65;
constexpr int     kWrappedHeight     = 5;
constexpr int     kWrappedPlanes     = 3;
constexpr int64_t kWrappedRowStride  = kWrappedWidth + 2;
constexpr int64_t kWrappedPlaneBytes = kWrappedRowStride * kWrappedHeight;
constexpr int64_t kWrappedImageBytes = kWrappedPlaneBytes * kWrappedPlanes;
constexpr int64_t kWrappedBaseOffset = 1;

static_assert(kWrappedWidth % 4 != 0);
static_assert(kWrappedBaseOffset % alignof(uchar4) != 0);
static_assert((kWrappedBaseOffset + kWrappedRowStride) % alignof(uchar4) == 0);

nvcv::Image WrapPlanarImage(NVCVByte *base)
{
    nvcv::ImageDataStridedCuda::Buffer buf{};
    buf.numPlanes = kWrappedPlanes;
    for (int p = 0; p < kWrappedPlanes; ++p)
    {
        buf.planes[p].width     = kWrappedWidth;
        buf.planes[p].height    = kWrappedHeight;
        buf.planes[p].rowStride = kWrappedRowStride;
        buf.planes[p].basePtr   = base + p * kWrappedPlaneBytes;
    }
    return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8p, buf});
}

} // namespace

TEST(OpPosterize, varshape_planar_wrapped_misaligned_images_correct_output)
{
    constexpr int    kBatch = 2;
    constexpr int    kBits  = 3;
    constexpr size_t kBytes = static_cast<size_t>(kWrappedBaseOffset) + kWrappedImageBytes * kBatch;

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
        srcImages.push_back(WrapPlanarImage(srcPtr + kWrappedBaseOffset + i * kWrappedImageBytes));
        dstImages.push_back(WrapPlanarImage(dstPtr + kWrappedBaseOffset + i * kWrappedImageBytes));
    }

    nvcv::ImageBatchVarShape srcBatch(kBatch);
    nvcv::ImageBatchVarShape dstBatch(kBatch);
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invokeBits(kBits)(stream, srcBatch, dstBatch));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, kBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    constexpr int kPixelsPerPlane = kWrappedHeight * kWrappedWidth;
    constexpr int kPixelsPerImage = kWrappedPlanes * kPixelsPerPlane;
    for (int index = 0; index < kBatch * kPixelsPerImage; ++index)
    {
        const int    i   = index / kPixelsPerImage;
        const int    p   = index / kPixelsPerPlane % kWrappedPlanes;
        const int    y   = index / kWrappedWidth % kWrappedHeight;
        const int    x   = index % kWrappedWidth;
        const size_t row = static_cast<size_t>(kWrappedBaseOffset) + i * kWrappedImageBytes + p * kWrappedPlaneBytes
                         + y * kWrappedRowStride;
        EXPECT_EQ(PosterizeScalarGold(srcBytes[row + x], kBits), dstBytes[row + x])
            << "image " << i << " plane " << p << " row " << y << " col " << x;
    }

    constexpr int kPaddingPerRow = kWrappedRowStride - kWrappedWidth;
    constexpr int kRowsPerImage  = kWrappedPlanes * kWrappedHeight;
    for (int index = 0; index < kBatch * kRowsPerImage * kPaddingPerRow; ++index)
    {
        const int    i   = index / (kRowsPerImage * kPaddingPerRow);
        const int    p   = index / (kWrappedHeight * kPaddingPerRow) % kWrappedPlanes;
        const int    y   = index / kPaddingPerRow % kWrappedHeight;
        const int    b   = kWrappedWidth + index % kPaddingPerRow;
        const size_t row = static_cast<size_t>(kWrappedBaseOffset) + i * kWrappedImageBytes + p * kWrappedPlaneBytes
                         + y * kWrappedRowStride;
        EXPECT_EQ(0xA5, dstBytes[row + b]) << "image " << i << " plane " << p << " row " << y << " padding byte " << b;
    }
}

TEST(OpPosterize, varshape_u16_wrapped_misaligned_rows_correct_output)
{
    constexpr int     kWidth      = 65;
    constexpr int     kHeight     = 5;
    constexpr int     kBatch      = 2;
    constexpr int     kBits       = 9;
    constexpr int64_t kRowStride  = kWidth * sizeof(uint16_t) + 4;
    constexpr int64_t kImageBytes = kRowStride * kHeight;
    constexpr int64_t kBaseOffset = 2;
    constexpr size_t  kBytes      = static_cast<size_t>(kBaseOffset) + kImageBytes * kBatch;

    static_assert(kWidth % 4 != 0);
    static_assert(kBaseOffset % alignof(ushort4) != 0);
    static_assert((kBaseOffset + kRowStride) % alignof(ushort4) == 0);

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

    auto wrapImage = [](NVCVByte *base)
    {
        nvcv::ImageDataStridedCuda::Buffer buf{};
        buf.numPlanes           = 1;
        buf.planes[0].width     = kWidth;
        buf.planes[0].height    = kHeight;
        buf.planes[0].rowStride = kRowStride;
        buf.planes[0].basePtr   = base;
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_U16, buf});
    };

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
    EXPECT_NO_THROW(invokeBits(kBits)(stream, srcBatch, dstBatch));
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
            const size_t row = static_cast<size_t>(kBaseOffset) + i * kImageBytes + y * kRowStride;
            for (int x = 0; x < kWidth; ++x)
            {
                uint16_t srcValue;
                uint16_t dstValue;
                std::memcpy(&srcValue, &srcBytes[row + x * sizeof(uint16_t)], sizeof(srcValue));
                std::memcpy(&dstValue, &dstBytes[row + x * sizeof(uint16_t)], sizeof(dstValue));
                EXPECT_EQ(PosterizeScalarGold(srcValue, kBits), dstValue)
                    << "image " << i << " row " << y << " col " << x;
            }
            for (int64_t b = kWidth * sizeof(uint16_t); b < kRowStride; ++b)
            {
                EXPECT_EQ(0xA5, dstBytes[row + b]) << "row padding overwritten";
            }
        }
    }
}

TEST(OpPosterize, padded_tensor_dispatch_and_valid_bit_boundaries)
{
    // Non-16-byte row strides decline the carrier path. These cases then cover the U16 interleaved
    // vector/scalar arms, the multi-channel interleaved scalar arm, and the U8 planar vector/scalar
    // arms while checking the valid bits endpoints and preserving row padding.
    RunPosterizePaddedTensor<uint16_t>(nvcv::FMT_U16, 30, 11, 2, 12, 0); // 72-byte rows: vector path
    RunPosterizePaddedTensor<uint16_t>(nvcv::FMT_U16, 30, 11, 2, 6, 16); // 66-byte rows: scalar path
    RunPosterizePaddedTensor<uint8_t>(nvcv::FMT_RGB8, 41, 13, 2, 2, 4);  // uchar3 scalar path
    RunPosterizePaddedTensor<uint8_t>(nvcv::FMT_RGB8p, 41, 13, 2, 3, 4); // 4-byte-aligned rows: vector path
    RunPosterizePaddedTensor<uint8_t>(nvcv::FMT_RGB8p, 41, 13, 2, 2, 8); // unaligned rows: scalar path
}

TEST(OpPosterize, planar_tensor_and_varshape_vector_dispatch)
{
    // Keep Tensor rows outside the 16-byte carrier alignment while retaining 4-byte Vec4 alignment,
    // then submit a variable-shape planar batch to exercise both planar Vec4 launch gates.
    RunPosterizePaddedTensor<uint8_t>(nvcv::FMT_RGB8p, 37, 9, 1, 3, 4);

    const std::vector<nvcv::Size2D> sizes{
        {37, 9},
        {68, 5}
    };
    test::planar::RunVarShapeParity(
        nvcv::FMT_RGB8p, nvcv::FMT_RGB8, sizes, sizes,
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeBits(4)(s, src, dst)); });
}

TEST(OpPosterize, planar_varshape_plane_grid_fallback)
{
    constexpr int    kChannels   = 3;
    constexpr int    kWidth      = 1;
    constexpr int    kHeight     = 1;
    constexpr int    kBatch      = 65535 / kChannels + 1;
    constexpr size_t kImageBytes = kChannels;
    const size_t     totalBytes  = static_cast<size_t>(kBatch) * kImageBytes;

    // Keep the image count below the standalone grid.z limit while making the planar (image, plane)
    // count 65538. The vectorized VarShape gate must decline that launch, so the scalar fallback is
    // exercised with independent storage for every image.
    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, totalBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, totalBytes));

    std::vector<uint8_t> srcBytes(totalBytes);
    for (size_t i = 0; i < srcBytes.size(); ++i)
    {
        srcBytes[i] = static_cast<uint8_t>(i * 37U + 11U);
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), totalBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, totalBytes));

    auto wrapImage = [=](NVCVByte *base)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes = kChannels;
        for (int plane = 0; plane < kChannels; ++plane)
        {
            buffer.planes[plane].width     = kWidth;
            buffer.planes[plane].height    = kHeight;
            buffer.planes[plane].rowStride = kWidth;
            buffer.planes[plane].basePtr   = base + plane;
        }
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8p, buffer});
    };

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    srcImages.reserve(kBatch);
    dstImages.reserve(kBatch);
    for (int image = 0; image < kBatch; ++image)
    {
        const size_t offset = static_cast<size_t>(image) * kImageBytes;
        srcImages.push_back(wrapImage(srcPtr + offset));
        dstImages.push_back(wrapImage(dstPtr + offset));
    }

    nvcv::ImageBatchVarShape srcBatch(kBatch);
    nvcv::ImageBatchVarShape dstBatch(kBatch);
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invokeBits(4)(stream, srcBatch, dstBatch));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(totalBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, totalBytes, cudaMemcpyDeviceToHost));
    for (int image = 0; image < kBatch; ++image)
    {
        for (int plane = 0; plane < kChannels; ++plane)
        {
            const size_t offset = static_cast<size_t>(image) * kImageBytes + plane;
            EXPECT_EQ(PosterizeScalarGold<uint8_t>(srcBytes[offset], 4), dstBytes[offset])
                << "payload mismatch at image " << image << " plane " << plane;
        }
    }

    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpPosterize_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_S16,   nvcv::FMT_S16  }, // unsupported dtype (signed 16-bit)
    {nvcv::FMT_S32,   nvcv::FMT_S32  }, // unsupported dtype (signed 32-bit)
    {nvcv::FMT_RGB8,  nvcv::FMT_RGB8p}, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,  nvcv::FMT_U8   }, // input/output channel mismatch
});

// clang-format on
TEST_P(OpPosterize_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invokeBits(4));
}

TEST(OpPosterize_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2S16, nvcv::FMT_2S16, invokeBits(4), 16, 16);
}

// bits outside [0, W] must be rejected (u8 has W=8, so 9 and -1 are out of range). A float image
// has no bit layout, so its range is the reference's [0, 8] whatever the type's width.
TEST(OpPosterize_Negative, rejects_bits_out_of_range)
{
    ew::ExpectRejected(nvcv::FMT_U8, nvcv::FMT_U8, invokeBits(9), 16, 16);
    ew::ExpectRejected(nvcv::FMT_U8, nvcv::FMT_U8, invokeBits(-1), 16, 16);
    ew::ExpectRejected(nvcv::FMT_F32, nvcv::FMT_F32, invokeBits(9), 16, 16);
    ew::ExpectRejected(nvcv::FMT_F32, nvcv::FMT_F32, invokeBits(-1), 16, 16);
}

static void RunPosterizeFloatBuckets(cudaStream_t stream, const nvcv::Tensor &inTensor, nvcv::Tensor &outTensor,
                                     const nvcv::TensorDataStridedCuda &outData, const std::vector<float> &input,
                                     int bits)
{
    const size_t count = input.size();
    SCOPED_TRACE(bits);
    ASSERT_NO_THROW(invokeBits(bits)(stream, inTensor, outTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<float> got;
    nvcv::util::GetImageVectorFromTensor<float>(outData, 0, got);
    ASSERT_EQ(count, got.size());

    const auto levels = static_cast<float>(1u << bits);
    for (size_t i = 0; i < count; ++i)
    {
        float q = std::floor(input[i] * levels);
        q       = std::clamp(q, 0.0f, levels - 1.0f);
        EXPECT_EQ(q / levels, got[i]) << "at " << i;
    }
}

TEST(OpPosterize, float_quantizes_into_level_buckets)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int      kWidth  = 9;
    constexpr int      kHeight = 5;
    const size_t       count   = static_cast<size_t>(kWidth) * kHeight;
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i)
    {
        input[i] = static_cast<float>(i) / static_cast<float>(count - 1);
    }

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(1, kWidth, kHeight, nvcv::FMT_F32);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(1, kWidth, kHeight, nvcv::FMT_F32);
    auto         inData    = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto         outData   = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);
    ASSERT_NO_THROW(nvcv::util::SetImageTensorFromVector<float>(*inData, input, 0));

    for (int bits = 0; bits <= 8; ++bits)
    {
        ASSERT_NO_FATAL_FAILURE(RunPosterizeFloatBuckets(stream, inTensor, outTensor, *outData, input, bits));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// The two rejections same_shape::ValidateSrcDstTensors owns. They are asserted per entry point
// rather than only where the fix was written, because that one validator now decides the tensor
// contract of four operators. Bodies and rationale: ElementwiseOpHarness.hpp.
TEST(OpPosterize_Negative, tensor_padded_column_stride)
{
    ew::ExpectPaddedColumnStrideTensorRejected(invokeBits(4));
}

TEST(OpPosterize_Negative, tensor_multi_channel_dtype)
{
    ew::ExpectMultiChannelDtypeTensorRejected(invokeBits(4));
}

namespace {

template<typename T>
std::vector<T> PosterizeFloatInputs(int bits)
{
    std::vector<T> values;
    for (double v : std::initializer_list<double>{-INFINITY, -2.0, -0.0, 0.0, 1.0, 2.0, INFINITY, NAN})
    {
        values.push_back(static_cast<T>(v));
    }
    const int levels = 1 << bits;
    for (int q = 0; q <= levels; ++q)
    {
        const auto boundary = static_cast<T>(static_cast<double>(q) / levels);
        if constexpr (std::is_same_v<T, __half>)
        {
            uint16_t raw;
            std::memcpy(&raw, &boundary, sizeof(raw));
            for (uint16_t neighbor :
                 {static_cast<uint16_t>(raw == 0 ? 0x8001 : raw - 1), raw, static_cast<uint16_t>(raw + 1)})
            {
                values.push_back(__half{__half_raw{neighbor}});
            }
        }
        else
        {
            values.push_back(std::nextafter(boundary, -std::numeric_limits<T>::infinity()));
            values.push_back(boundary);
            values.push_back(std::nextafter(boundary, std::numeric_limits<T>::infinity()));
        }
    }
    return values;
}

template<typename T>
nvcv::ImageFormat PosterizeFloatFormat(nvcv::DataType dtype, int channels, bool planar)
{
    auto swizzle = nvcv::Swizzle::S_XYZW;
    if (channels == 1)
        swizzle = nvcv::Swizzle::S_X000;
    else if (channels == 3)
        swizzle = nvcv::Swizzle::S_XYZ0;
    if (planar)
    {
        return nvcv::ImageFormat{nvcv::MemLayout::PL,
                                 nvcv::DataKind::FLOAT,
                                 swizzle,
                                 dtype.packing(),
                                 channels > 1 ? dtype.packing() : nvcv::Packing::NONE,
                                 channels > 1 ? dtype.packing() : nvcv::Packing::NONE,
                                 channels == 4 ? dtype.packing() : nvcv::Packing::NONE};
    }
    const std::array<nvcv::Packing, 3> packed3{nvcv::Packing::X16_Y16_Z16, nvcv::Packing::X32_Y32_Z32,
                                               nvcv::Packing::X64_Y64_Z64};
    const std::array<nvcv::Packing, 3> packed4{nvcv::Packing::X16_Y16_Z16_W16, nvcv::Packing::X32_Y32_Z32_W32,
                                               nvcv::Packing::X64_Y64_Z64_W64};
    constexpr auto                     packingIndex = sizeof(T) / 4;
    auto                               packing      = dtype.packing();
    if (channels == 3)
        packing = packed3[packingIndex];
    else if (channels == 4)
        packing = packed4[packingIndex];
    return nvcv::ImageFormat{nvcv::MemLayout::PL, nvcv::DataKind::FLOAT, swizzle, packing};
}

template<typename T>
std::vector<size_t> MakePosterizeFloatGold(int bits, int rowElements, int offset, int64_t rowStride,
                                           std::vector<uint8_t> &input, std::vector<uint8_t> &expected)
{
    const auto          values = PosterizeFloatInputs<T>(bits);
    std::vector<size_t> nanOffsets;
    size_t              index    = 0;
    const size_t        rowCount = (input.size() - offset) / rowStride;
    for (size_t r = 0; r < rowCount; ++r)
    {
        for (int e = 0; e < rowElements; ++e, ++index)
        {
            const auto pos   = offset + r * rowStride + e * sizeof(T);
            const T    value = values[index % values.size()];
            // The oracle uses arithmetic in the original computation domain, independently
            // of the device implementation. Power-of-two division is exact after rounding.
            using AT          = std::conditional_t<std::is_same_v<T, double>, double, float>;
            const auto levels = static_cast<AT>(1 << bits);
            const AT   q      = std::clamp(std::floor(static_cast<AT>(value) * levels), AT{0}, levels - AT{1});
            const auto gold   = static_cast<T>(q / levels);
            std::memcpy(input.data() + pos, &value, sizeof(T));
            std::memcpy(expected.data() + pos, &gold, sizeof(T));
            if (std::isnan(static_cast<double>(value)))
                nanOffsets.push_back(pos);
        }
    }
    return nanOffsets;
}

template<typename T, typename Invoke>
void CheckPosterizeFloatEdges(cudaStream_t stream, Invoke invoke, NVCVByte *dstPtr,
                              const std::vector<size_t> &nanOffsets, std::vector<uint8_t> &expected)
{
    const size_t         bytes = expected.size();
    std::vector<uint8_t> got(bytes);
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, bytes));
    ASSERT_NO_THROW(invoke());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(got.data(), dstPtr, bytes, cudaMemcpyDeviceToHost));
    for (size_t pos : nanOffsets)
    {
        T value;
        std::memcpy(&value, got.data() + pos, sizeof(T));
        ASSERT_TRUE(std::isnan(static_cast<double>(value)));
        std::memcpy(expected.data() + pos, &value, sizeof(T));
    }
    EXPECT_EQ(expected, got); // Includes signed zero, vector tails and untouched row padding.
}

template<typename T>
void RunPosterizeFloatEdges(nvcv::DataType dtype, int channels, bool planar, int packing)
{
    constexpr int batch       = 2;
    constexpr int height      = 5;
    const int     width       = packing == 0 ? 256 : 257;
    const int     rowElements = width * (planar ? 1 : channels);
    const int     rows        = height * (planar ? channels : 1);
    const int64_t rowBytes    = rowElements * sizeof(T);
    int64_t       rowStride   = rowBytes;
    if (packing != 0)
        rowStride = (rowBytes + 15) / 16 * 16 + (packing == 2 ? sizeof(T) : 0);
    const int64_t sampleStride = rowStride * rows;
    const int     offset       = packing == 2 ? sizeof(T) : 0;
    const size_t  bytes        = offset + batch * sampleStride;
    NVCVByte     *srcPtr       = nullptr;
    NVCVByte     *dstPtr       = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, bytes));
    auto wrap = [&](NVCVByte *ptr)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr + offset;
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = planar ? rowStride * height : rowStride;
        buffer.strides[2] = planar ? rowStride : channels * sizeof(T);
        buffer.strides[3] = sizeof(T);
        const auto shape = planar ? nvcv::TensorShape{{batch, channels, height, width}, "NCHW"}
                                  : nvcv::TensorShape{{batch, height, width, channels}, "NHWC"};
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, dtype, buffer});
    };
    auto       src       = wrap(srcPtr);
    auto       dst       = wrap(dstPtr);
    const auto format    = PosterizeFloatFormat<T>(dtype, channels, planar);
    auto       wrapImage = [&](NVCVByte *ptr, int n)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes = planar ? channels : 1;
        for (int p = 0; p < buffer.numPlanes; ++p)
        {
            buffer.planes[p].width     = width;
            buffer.planes[p].height    = height;
            buffer.planes[p].rowStride = static_cast<int32_t>(rowStride);
            buffer.planes[p].basePtr   = ptr + offset + n * sampleStride + p * rowStride * height;
        }
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, buffer});
    };
    nvcv::ImageBatchVarShape srcBatch(batch);
    nvcv::ImageBatchVarShape dstBatch(batch);
    for (int n = 0; n < batch; ++n)
    {
        srcBatch.pushBack(wrapImage(srcPtr, n));
        dstBatch.pushBack(wrapImage(dstPtr, n));
    }
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    auto runBits
        = [bytes, rowElements, offset, rowStride, srcPtr, stream, &src, &dst, dstPtr, &srcBatch, &dstBatch](int bits)
    {
        SCOPED_TRACE(bits);
        std::vector<uint8_t> input(bytes, 0xA5);
        std::vector<uint8_t> expected(bytes, 0xA5);
        const auto nanOffsets = MakePosterizeFloatGold<T>(bits, rowElements, offset, rowStride, input, expected);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), bytes, cudaMemcpyHostToDevice));
        {
            SCOPED_TRACE(false);
            ASSERT_NO_FATAL_FAILURE(CheckPosterizeFloatEdges<T>(
                stream, [bits, stream, &src, &dst] { invokeBits(bits)(stream, src, dst); }, dstPtr, nanOffsets,
                expected));
        }
        {
            SCOPED_TRACE(true);
            ASSERT_NO_FATAL_FAILURE(CheckPosterizeFloatEdges<T>(
                stream, [bits, stream, &srcBatch, &dstBatch] { invokeBits(bits)(stream, srcBatch, dstBatch); }, dstPtr,
                nanOffsets, expected));
        }
    };
    for (int bits = 0; bits <= 8; ++bits)
    {
        ASSERT_NO_FATAL_FAILURE(runBits(bits));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

} // namespace

static void RunPosterizeFloatEdgeTypes(int channels, bool planar, int packing)
{
    ASSERT_NO_FATAL_FAILURE(RunPosterizeFloatEdges<__half>(nvcv::TYPE_F16, channels, planar, packing));
    ASSERT_NO_FATAL_FAILURE(RunPosterizeFloatEdges<float>(nvcv::TYPE_F32, channels, planar, packing));
    ASSERT_NO_FATAL_FAILURE(RunPosterizeFloatEdges<double>(nvcv::TYPE_F64, channels, planar, packing));
}

TEST(OpPosterize, floating_boundaries_and_strides)
{
    const auto cases = test::Combine(test::ValueList<int>{1, 3, 4}, test::ValueList<bool>{false, true},
                                     test::ValueList<int>{0, 1, 2});
    for (const auto &[channels, planar, packing] : cases)
    {
        SCOPED_TRACE(::testing::Message() << "channels=" << channels << " planar=" << planar << " packing=" << packing);
        ASSERT_NO_FATAL_FAILURE(RunPosterizeFloatEdgeTypes(channels, planar, packing));
    }
}

static void RunPosterizeHalfQuantization(cudaStream_t stream, const nvcv::Tensor &src, nvcv::Tensor &dst,
                                         const nvcv::TensorDataStridedCuda &dstData, const std::vector<__half> &input,
                                         int bits)
{
    constexpr int count = 65536;
    SCOPED_TRACE(bits);
    ASSERT_NO_THROW(invokeBits(bits)(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    std::vector<__half> got;
    nvcv::util::GetImageVectorFromTensor<__half>(dstData, 0, got);
    ASSERT_EQ(input.size(), got.size());
    const auto levels = static_cast<float>(1 << bits);
    for (int i = 0; i < count; ++i)
    {
        const float value = __half2float(input[i]);
        if (std::isnan(value))
        {
            ASSERT_TRUE(std::isnan(__half2float(got[i]))) << i;
            continue;
        }
        const float  quantized = std::clamp(std::floor(value * levels), 0.f, levels - 1.f);
        const __half gold      = __float2half(quantized / levels);
        ASSERT_EQ(static_cast<__half_raw>(gold).x, static_cast<__half_raw>(got[i]).x) << i;
    }
}

TEST(OpPosterize, exhaustive_half_quantization)
{
    constexpr int       count = 65536;
    std::vector<__half> input(count);
    for (int i = 0; i < count; ++i)
    {
        input[i] = __half{__half_raw{static_cast<uint16_t>(i)}};
    }
    auto src     = nvcv::util::CreateTensor(1, 256, 256, nvcv::FMT_F16);
    auto dst     = nvcv::util::CreateTensor(1, 256, 256, nvcv::FMT_F16);
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    nvcv::util::SetImageTensorFromVector<__half>(*srcData, input, 0);
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    for (int bits = 0; bits <= 8; ++bits)
    {
        ASSERT_NO_FATAL_FAILURE(RunPosterizeHalfQuantization(stream, src, dst, *dstData, input, bits));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPosterize_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaPosterizeCreate(nullptr));
}
