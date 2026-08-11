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

#include "../../../src/cvcuda/priv/OpAutoContrast.hpp"
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpAutoContrast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace nvcvcuda = nvcv::cuda;
namespace test     = nvcv::test;
namespace ttype    = nvcv::test::type;

using uchar  = unsigned char;
using ushort = unsigned short;

static uint32_t FourCC(const char (&code)[5])
{
    uint32_t value{};
    std::memcpy(&value, code, sizeof(value));
    return value;
}

template<typename To, typename From>
static To BitwiseCopy(From value)
{
    static_assert(sizeof(To) == sizeof(From));
    To result{};
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// 16-bit unsigned multi-channel interleaved formats are not predefined; build them like
// TestOpBrightnessContrast.cpp does for its 16-bit cases.
#define NVCV_IMAGE_FORMAT_RGB16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGB16Up \
    NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, UNSIGNED, XYZ0, ASSOCIATED, X16, X16, X16)
#define NVCV_IMAGE_FORMAT_RGBA16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)

template<typename T>
using uniform_distribution
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

// ---------------------------------------------------------------------------
// Independent CPU reference. Pillow and torchvision truncate non-negative
// integer results after scaling, whereas floating-point outputs retain the
// scaled value. Do not call production conversion helpers from this oracle.
// Only the interleaved layout is referenced directly; planar correctness is
// established by the planar-parity suite (planar output == interleaved output
// bit-for-bit).
// ---------------------------------------------------------------------------
template<typename BT>
inline BT RemapGold(BT in, float lo, float hi, float bound)
{
    if constexpr (std::is_floating_point_v<BT>)
    {
        if (!std::isfinite(in))
        {
            return in;
        }
    }
    if (hi == lo)
    {
        return in; // flat channel -> unchanged (torchvision semantics)
    }
    float range = hi - lo;
    float val;
    if (std::isfinite(range))
    {
        val = (static_cast<float>(in) - lo) * bound / range;
    }
    else
    {
        const double wideRange = static_cast<double>(hi) - static_cast<double>(lo);
        val = static_cast<float>((static_cast<double>(in) - static_cast<double>(lo)) * bound / wideRange);
    }
    val = std::clamp(val, 0.f, bound);
    if constexpr (std::is_integral_v<BT>)
    {
        return static_cast<BT>(std::floor(val));
    }
    else
    {
        return static_cast<BT>(val);
    }
}

inline void UpdateFiniteExtrema(float value, float &lo, float &hi)
{
    if (std::isfinite(value))
    {
        lo = std::min(lo, value);
        hi = std::max(hi, value);
    }
}

template<typename T>
void AutoContrastGoldSample(const uint8_t *src, uint8_t *ref, int width, int height, long rowStride)
{
    using BT                  = nvcvcuda::BaseType<T>;
    constexpr int numChannels = nvcvcuda::NumElements<T>;
    const float   bound       = std::is_floating_point_v<BT> ? 1.0f : static_cast<float>(nvcvcuda::TypeTraits<BT>::max);

    std::array<float, numChannels> lo;
    std::array<float, numChannels> hi;
    for (int c = 0; c < numChannels; ++c)
    {
        lo[c] = std::numeric_limits<float>::infinity();
        hi[c] = -std::numeric_limits<float>::infinity();
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const T px = *reinterpret_cast<const T *>(src + y * rowStride + x * static_cast<long>(sizeof(T)));
            for (int c = 0; c < numChannels; ++c)
            {
                const auto value = static_cast<float>(nvcvcuda::GetElement(px, c));
                UpdateFiniteExtrema(value, lo[c], hi[c]);
            }
        }
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const long off = y * rowStride + x * static_cast<long>(sizeof(T));
            T          px  = *reinterpret_cast<const T *>(src + off);
            T          out{};
            for (int c = 0; c < numChannels; ++c)
            {
                nvcvcuda::GetElement(out, c) = RemapGold<BT>(nvcvcuda::GetElement(px, c), lo[c], hi[c], bound);
            }
            *reinterpret_cast<T *>(ref + off) = out;
        }
    }
}

template<typename T>
auto BitExactValue(T value)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return BitwiseCopy<uint32_t>(value);
    }
    else
    {
        return value;
    }
}

// Per-sample bit-exact comparison, factored out so the call sites stay within the 3-level
// control-nesting budget (one outer per-sample loop + this helper's row/col/channel loops).
template<typename T>
void ExpectSampleEqual(const uint8_t *got, const uint8_t *ref, int width, int height, long rowStride)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const long off      = y * rowStride + x * static_cast<long>(sizeof(T));
            const T    actual   = *reinterpret_cast<const T *>(got + off);
            const T    expected = *reinterpret_cast<const T *>(ref + off);
            for (int c = 0; c < nvcvcuda::NumElements<T>; ++c)
            {
                const auto gotValue = nvcvcuda::GetElement(actual, c);
                const auto refValue = nvcvcuda::GetElement(expected, c);
                ASSERT_EQ(BitExactValue(gotValue), BitExactValue(refValue))
                    << "pixel (" << x << "," << y << ") channel " << c;
            }
        }
    }
}

template<typename T>
void SetAllElements(T &pixel, nvcvcuda::BaseType<T> value)
{
    for (int c = 0; c < nvcvcuda::NumElements<T>; ++c)
    {
        nvcvcuda::GetElement(pixel, c) = value;
    }
}

template<typename T, typename Rng>
void FillRandomSample(uint8_t *buf, int width, int height, long rowStride, Rng &rng)
{
    using BT = nvcvcuda::BaseType<T>;
    uniform_distribution<BT> dist(BT{0}, std::is_integral_v<BT> ? nvcvcuda::TypeTraits<BT>::max : BT{1});
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            T px{};
            for (int c = 0; c < nvcvcuda::NumElements<T>; ++c)
            {
                nvcvcuda::GetElement(px, c) = static_cast<BT>(dist(rng));
            }
            *reinterpret_cast<T *>(buf + y * rowStride + x * static_cast<long>(sizeof(T))) = px;
        }
    }
}

template<typename T, class PixelAt>
void ExpectTensorMatchesGold(int width, int height, nvcv::ImageFormat fmt, PixelAt pixelAt)
{
    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(1, width, height, fmt);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(1, width, height, fmt);

    auto srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    const long           rowStride = srcAccess->rowStride();
    const size_t         bufSize   = static_cast<size_t>(rowStride) * height;
    std::vector<uint8_t> srcVec(bufSize, uint8_t{0});
    std::vector<uint8_t> refVec(bufSize, uint8_t{0});
    std::vector<uint8_t> dstVec(bufSize, uint8_t{0});

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            *reinterpret_cast<T *>(srcVec.data() + y * rowStride + x * static_cast<long>(sizeof(T))) = pixelAt(x, y);
        }
    }

    AutoContrastGoldSample<T>(srcVec.data(), refVec.data(), width, height, rowStride);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), bufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AutoContrast op;
    ASSERT_NO_THROW(op(stream, srcTensor, dstTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstVec.data(), dstData->basePtr(), bufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    ExpectSampleEqual<T>(dstVec.data(), refVec.data(), width, height, rowStride);
}

template<typename T, class PixelAt>
void ExpectVarShapeMatchesGold(int width, int height, nvcv::ImageFormat fmt, PixelAt pixelAt)
{
    nvcv::Image srcImage({width, height}, fmt);
    nvcv::Image dstImage({width, height}, fmt);

    auto srcData = srcImage.exportData<nvcv::ImageDataStridedCuda>();
    auto dstData = dstImage.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    ASSERT_EQ(1, srcData->numPlanes());
    ASSERT_EQ(sizeof(T), fmt.planePixelStrideBytes(0));

    const long           rowStride = srcData->plane(0).rowStride;
    const size_t         bufSize   = static_cast<size_t>(rowStride) * height;
    std::vector<uint8_t> srcVec(bufSize, uint8_t{0});
    std::vector<uint8_t> refVec(bufSize, uint8_t{0});
    std::vector<uint8_t> dstVec(bufSize, uint8_t{0});

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            *reinterpret_cast<T *>(srcVec.data() + y * rowStride + x * static_cast<long>(sizeof(T))) = pixelAt(x, y);
        }
    }

    AutoContrastGoldSample<T>(srcVec.data(), refVec.data(), width, height, rowStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2DAsync(srcData->plane(0).basePtr, rowStride, srcVec.data(), rowStride,
                                static_cast<size_t>(width) * sizeof(T), height, cudaMemcpyHostToDevice, stream));

    nvcv::ImageBatchVarShape srcBatch(1);
    nvcv::ImageBatchVarShape dstBatch(1);
    srcBatch.pushBack(srcImage);
    dstBatch.pushBack(dstImage);

    cvcuda::AutoContrast op;
    ASSERT_NO_THROW(op(stream, srcBatch, dstBatch));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2D(dstVec.data(), rowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                           static_cast<size_t>(width) * sizeof(T), height, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    ExpectSampleEqual<T>(dstVec.data(), refVec.data(), width, height, rowStride);
}

static void ExpectZeroExtentTensorStatus(const nvcv::TensorShape &shape, nvcv::DataType dtype, NVCVStatus status)
{
    std::optional<nvcv::Tensor> src;
    std::optional<nvcv::Tensor> dst;
    try
    {
        src.emplace(shape, dtype);
        dst.emplace(shape, dtype);
    }
    catch (const nvcv::Exception &e)
    {
        GTEST_SKIP() << "zero-extent tensors are not constructible: " << e.what();
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;
    EXPECT_EQ(status, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, *src, *dst); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
#define NVCV_TEST_CASE(W, H, N, T, FMT) ttype::Types<ttype::Value<W>, ttype::Value<H>, ttype::Value<N>, T, ttype::Value<FMT>>

NVCV_TYPED_TEST_SUITE(
    OpAutoContrast,
    ttype::Types<
        NVCV_TEST_CASE(  2,   2, 1,  uchar,  NVCV_IMAGE_FORMAT_U8),
        NVCV_TEST_CASE( 41,  39, 3,  uchar,  NVCV_IMAGE_FORMAT_U8),
        NVCV_TEST_CASE( 42,  17, 2, uchar3,  NVCV_IMAGE_FORMAT_RGB8),
        NVCV_TEST_CASE( 64,  64, 4, uchar4,  NVCV_IMAGE_FORMAT_RGBA8),
        NVCV_TEST_CASE(101, 107, 2, ushort,  NVCV_IMAGE_FORMAT_U16),
        NVCV_TEST_CASE( 33,  21, 3, ushort3, NVCV_IMAGE_FORMAT_RGB16U),
        NVCV_TEST_CASE( 17,  19, 2, ushort4, NVCV_IMAGE_FORMAT_RGBA16U),
        NVCV_TEST_CASE(128,   9, 2,  float,  NVCV_IMAGE_FORMAT_F32),
        NVCV_TEST_CASE( 59,  77, 3, float3,  NVCV_IMAGE_FORMAT_RGBf32),
        NVCV_TEST_CASE( 32,  48, 4, float4,  NVCV_IMAGE_FORMAT_RGBAf32),
        // 64 samples * 4 channels makes the 6 MiB workspace limit, rather than
        // MAX_PARTIALS_PER_SAMPLE, bound the reduction grid at this height.
        NVCV_TEST_CASE(  1, 16385, 64, uchar4, NVCV_IMAGE_FORMAT_RGBA8),
        // Both logical grid dimensions exceed the bounded physical grid: the
        // 8193x2 logical grid is executed by a 4096x1 physical grid.
        NVCV_TEST_CASE(2097153, 5, 1, uchar4, NVCV_IMAGE_FORMAT_RGBA8)>);

// clang-format on

TYPED_TEST(OpAutoContrast, tensor_correct_output)
{
    const int width      = ttype::GetValue<TypeParam, 0>;
    const int height     = ttype::GetValue<TypeParam, 1>;
    const int numSamples = ttype::GetValue<TypeParam, 2>;
    using T              = ttype::GetType<TypeParam, 3>;
    const nvcv::ImageFormat fmt{ttype::GetValue<TypeParam, 4>};

    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(numSamples, width, height, fmt);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(numSamples, width, height, fmt);

    auto srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    const long rowStride    = srcAccess->rowStride();
    long       sampleStride = srcAccess->sampleStride();
    if (sampleStride == 0) // single-sample tensors report a zero sample stride
    {
        sampleStride = rowStride * height;
    }
    const size_t bufSize = static_cast<size_t>(sampleStride) * numSamples;

    std::vector<uint8_t> srcVec(bufSize, uint8_t{0});
    std::vector<uint8_t> refVec(bufSize, uint8_t{0});
    std::vector<uint8_t> dstVec(bufSize, uint8_t{0});

    std::mt19937_64 rng(12345);
    for (int s = 0; s < numSamples; ++s)
    {
        FillRandomSample<T>(srcVec.data() + s * sampleStride, width, height, rowStride, rng);
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), bufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;
    ASSERT_NO_THROW(op(stream, srcTensor, dstTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstVec.data(), dstData->basePtr(), bufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int s = 0; s < numSamples; ++s)
    {
        AutoContrastGoldSample<T>(srcVec.data() + s * sampleStride, refVec.data() + s * sampleStride, width, height,
                                  rowStride);
    }

    for (int s = 0; s < numSamples; ++s)
    {
        SCOPED_TRACE("sample " + std::to_string(s));
        ExpectSampleEqual<T>(dstVec.data() + s * sampleStride, refVec.data() + s * sampleStride, width, height,
                             rowStride);
    }
}

TYPED_TEST(OpAutoContrast, varshape_correct_output)
{
    const int baseW      = ttype::GetValue<TypeParam, 0>;
    const int baseH      = ttype::GetValue<TypeParam, 1>;
    const int numSamples = ttype::GetValue<TypeParam, 2>;
    using T              = ttype::GetType<TypeParam, 3>;
    const nvcv::ImageFormat fmt{ttype::GetValue<TypeParam, 4>};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image>          imgSrc;
    std::vector<nvcv::Image>          imgDst;
    std::vector<std::vector<uint8_t>> srcVec(numSamples);
    std::vector<long>                 rowStrides(numSamples);

    std::uniform_int_distribution randW(baseW / 2 + 1, baseW * 3 / 2 + 1);
    std::uniform_int_distribution randH(baseH / 2 + 1, baseH * 3 / 2 + 1);
    std::mt19937_64               rng(12345);

    ASSERT_EQ(sizeof(T), fmt.planePixelStrideBytes(0));

    for (int s = 0; s < numSamples; ++s)
    {
        const bool   fixedShape = numSamples * nvcvcuda::NumElements<T> >= 256 || baseW > 1'000'000;
        nvcv::Size2D imgShape   = fixedShape ? nvcv::Size2D{baseW, baseH} : nvcv::Size2D{randW(rng), randH(rng)};
        imgSrc.emplace_back(imgShape, fmt);
        imgDst.emplace_back(imgShape, fmt);

        auto srcImgData = imgSrc[s].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(srcImgData, nvcv::NullOpt);
        ASSERT_EQ(srcImgData->numPlanes(), 1);

        const long rowStride = srcImgData->plane(0).rowStride;
        rowStrides[s]        = rowStride;
        srcVec[s].resize(static_cast<size_t>(rowStride) * imgShape.h, uint8_t{0});

        FillRandomSample<T>(srcVec[s].data(), imgShape.w, imgShape.h, rowStride, rng);
        if constexpr (std::is_floating_point_v<nvcvcuda::BaseType<T>>)
        {
            constexpr std::array nonFiniteValues{-std::numeric_limits<float>::infinity(),
                                                 std::numeric_limits<float>::infinity(),
                                                 std::numeric_limits<float>::quiet_NaN()};
            for (int x = 0; x < static_cast<int>(nonFiniteValues.size()); ++x)
            {
                T pixel{};
                SetAllElements(pixel, nonFiniteValues[x]);
                *reinterpret_cast<T *>(srcVec[s].data() + x * static_cast<long>(sizeof(T))) = pixel;
            }
        }

        ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(srcImgData->plane(0).basePtr, rowStride, srcVec[s].data(), rowStride,
                                                 static_cast<size_t>(imgShape.w) * sizeof(T), imgShape.h,
                                                 cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(numSamples);
    nvcv::ImageBatchVarShape batchDst(numSamples);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::AutoContrast op;
    ASSERT_NO_THROW(op(stream, batchSrc, batchDst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int s = 0; s < numSamples; ++s)
    {
        SCOPED_TRACE(s);
        const int  width     = imgSrc[s].size().w;
        const int  height    = imgSrc[s].size().h;
        const long rowStride = rowStrides[s];

        std::vector<uint8_t> refVec(static_cast<size_t>(rowStride) * height, uint8_t{0});
        std::vector<uint8_t> dstVec(static_cast<size_t>(rowStride) * height, uint8_t{0});

        const auto dstImgData = imgDst[s].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dstVec.data(), rowStride, dstImgData->plane(0).basePtr, rowStride,
                                            static_cast<size_t>(width) * sizeof(T), height, cudaMemcpyDeviceToHost));

        AutoContrastGoldSample<T>(srcVec[s].data(), refVec.data(), width, height, rowStride);
        ExpectSampleEqual<T>(dstVec.data(), refVec.data(), width, height, rowStride);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// A channel that is flat (all pixels equal) must be left unchanged.
TEST(OpAutoContrast, flat_channel_passthrough)
{
    const int    width     = 16;
    const int    height    = 12;
    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGB8);

    auto srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    const long           rowStride = srcAccess->rowStride();
    const size_t         bufSize   = static_cast<size_t>(rowStride) * height;
    std::vector<uint8_t> srcVec(bufSize, uint8_t{0});
    std::vector<uint8_t> dstVec(bufSize, uint8_t{0});

    // Constant pixel -> every channel flat -> output must equal input.
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            uchar3 px{37, 200, 5};
            *reinterpret_cast<uchar3 *>(srcVec.data() + y * rowStride + x * 3) = px;
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), bufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AutoContrast op;
    ASSERT_NO_THROW(op(stream, srcTensor, dstTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstVec.data(), dstData->basePtr(), bufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    ExpectSampleEqual<uchar3>(dstVec.data(), srcVec.data(), width, height, rowStride);
}

// Single-channel (grayscale) tensor: the lone channel must stretch to the full range.
TEST(OpAutoContrast, single_channel_stretch)
{
    const int    width     = 64;
    const int    height    = 48;
    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(2, width, height, nvcv::FMT_U8);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(2, width, height, nvcv::FMT_U8);

    auto srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    const long rowStride    = srcAccess->rowStride();
    long       sampleStride = srcAccess->sampleStride();
    if (sampleStride == 0)
    {
        sampleStride = rowStride * height;
    }
    const size_t         bufSize = static_cast<size_t>(sampleStride) * 2;
    std::vector<uint8_t> srcVec(bufSize, uint8_t{0});
    std::vector<uint8_t> refVec(bufSize, uint8_t{0});
    std::vector<uint8_t> dstVec(bufSize, uint8_t{0});

    std::mt19937_64 rng(7);
    for (int s = 0; s < 2; ++s)
    {
        // Narrow input range so auto-contrast meaningfully stretches it.
        std::uniform_int_distribution dist(40, 90);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                srcVec[s * sampleStride + y * rowStride + x] = static_cast<uint8_t>(dist(rng));
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), bufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AutoContrast op;
    ASSERT_NO_THROW(op(stream, srcTensor, dstTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstVec.data(), dstData->basePtr(), bufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int s = 0; s < 2; ++s)
    {
        AutoContrastGoldSample<uchar>(srcVec.data() + s * sampleStride, refVec.data() + s * sampleStride, width, height,
                                      rowStride);
    }
    for (int s = 0; s < 2; ++s)
    {
        SCOPED_TRACE("sample " + std::to_string(s));
        ExpectSampleEqual<uchar>(dstVec.data() + s * sampleStride, refVec.data() + s * sampleStride, width, height,
                                 rowStride);
    }
}

TEST(OpAutoContrast, integer_midpoints_truncate_like_pillow_and_torchvision)
{
    ExpectTensorMatchesGold<uchar>(3, 1, nvcv::FMT_U8, [](int x, int) { return static_cast<uchar>(x); });
    ExpectTensorMatchesGold<ushort>(3, 1, nvcv::FMT_U16, [](int x, int) { return static_cast<ushort>(x); });
}

TEST(OpAutoContrast, negative_float_reduction_matches_gold)
{
    ExpectTensorMatchesGold<float3>(37, 19, nvcv::FMT_RGBf32,
                                    [](int x, int y)
                                    {
                                        return float3{
                                            static_cast<float>((x % 11) - 5) * 0.25f,
                                            static_cast<float>((y % 7) - 3) * 0.5f,
                                            static_cast<float>(((x * 3 + y * 5) % 17) - 8) * 0.125f,
                                        };
                                    });
}

TEST(OpAutoContrast, extreme_finite_float_range_matches_gold)
{
    ExpectTensorMatchesGold<float>(3, 1, nvcv::FMT_F32,
                                   [](int x, int)
                                   {
                                       constexpr float      max = std::numeric_limits<float>::max();
                                       constexpr std::array values{-max, 0.f, max};
                                       return values[x];
                                   });
}

TEST(OpAutoContrast, nonsymmetric_extreme_finite_float_range_matches_gold)
{
    // The subtraction overflows even though every input is finite. At the interior value, a float-halving fallback
    // is one ULP above the double-precision reference.
    static constexpr std::array values{
        -0x1.66c172p+127f,
        0x1.565eb4p+126f,
        0x1.323798p+127f,
    };
    ASSERT_TRUE(std::isinf(values[2] - values[0]));

    ExpectTensorMatchesGold<float>(3, 1, nvcv::FMT_F32, [](int x, int) { return values[x]; });
}

TEST(OpAutoContrast, flat_positive_infinite_float_channel_passes_through)
{
    ExpectTensorMatchesGold<float>(7, 3, nvcv::FMT_F32,
                                   [](int, int) { return std::numeric_limits<float>::infinity(); });
}

TEST(OpAutoContrast, flat_negative_infinite_float_channel_passes_through)
{
    ExpectTensorMatchesGold<float>(7, 3, nvcv::FMT_F32,
                                   [](int, int) { return -std::numeric_limits<float>::infinity(); });
}

TEST(OpAutoContrast, mixed_finite_and_nonfinite_float_channel_matches_gold)
{
    constexpr uint32_t nanBits  = 0x7fc12345;
    const float        nanValue = BitwiseCopy<float>(nanBits);

    const std::array values{-std::numeric_limits<float>::infinity(), -2.f,    0.f, 2.f,
                            std::numeric_limits<float>::infinity(),  nanValue};
    ExpectTensorMatchesGold<float>(values.size(), 1, nvcv::FMT_F32, [&values](int x, int) { return values[x]; });
}

constexpr int BOUNDED_GRID_TEST_HEIGHT = 16385;

static float3 BoundedGridFloat3Pixel(int, int y)
{
    constexpr uint32_t nanBits  = 0x7fc12345;
    const float        nanValue = BitwiseCopy<float>(nanBits);

    float3 pixel{
        -8.f + static_cast<float>(y % 257) * 0.0625f,
        20.f + static_cast<float>((y * 17) % 251) * 0.125f,
        -3.5f,
    };
    if (y == 0)
    {
        pixel.x = -std::numeric_limits<float>::infinity();
    }
    else if (y == 4096)
    {
        pixel.x = std::numeric_limits<float>::infinity();
    }
    else if (y == 8192)
    {
        pixel.y = nanValue;
    }
    else if (y == 12288)
    {
        pixel.z = -std::numeric_limits<float>::infinity();
    }
    return pixel;
}

TEST(OpAutoContrast, bounded_grid_tensor_interleaved_multichannel_nonfinite_matches_gold)
{
    static_assert((BOUNDED_GRID_TEST_HEIGHT + 3) / 4 > 4096);
    ExpectTensorMatchesGold<float3>(1, BOUNDED_GRID_TEST_HEIGHT, nvcv::FMT_RGBf32, BoundedGridFloat3Pixel);
}

TEST(OpAutoContrast, bounded_grid_varshape_interleaved_multichannel_nonfinite_matches_gold)
{
    static_assert((BOUNDED_GRID_TEST_HEIGHT + 3) / 4 > 4096);
    ExpectVarShapeMatchesGold<float3>(1, BOUNDED_GRID_TEST_HEIGHT, nvcv::FMT_RGBf32, BoundedGridFloat3Pixel);
}

TEST(OpAutoContrast, zero_batch_tensor_is_noop)
{
    ExpectZeroExtentTensorStatus(
        nvcv::TensorShape{
            {0, 3, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, NVCV_SUCCESS);
}

TEST(OpAutoContrast, zero_height_tensor_is_noop)
{
    ExpectZeroExtentTensorStatus(
        nvcv::TensorShape{
            {1, 0, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, NVCV_SUCCESS);
}

TEST(OpAutoContrast, zero_width_tensor_is_noop)
{
    ExpectZeroExtentTensorStatus(
        nvcv::TensorShape{
            {1, 3, 0, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, NVCV_SUCCESS);
}

TEST(OpAutoContrast, empty_matching_varshape_is_noop)
{
    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;
    EXPECT_EQ(NVCV_SUCCESS, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAutoContrast, maximum_supported_height_matches_gold)
{
    constexpr int height = 4 * 65535;
    ExpectTensorMatchesGold<uchar>(1, height, nvcv::FMT_U8, [](int, int y) { return static_cast<uchar>(y % 251); });
}

TEST(OpAutoContrast, mixed_flat_and_nonflat_channels_match_gold)
{
    ExpectTensorMatchesGold<uchar3>(41, 23, nvcv::FMT_RGB8,
                                    [](int x, int y)
                                    {
                                        return uchar3{
                                            37,
                                            static_cast<uchar>(20 + (x * 7 + y * 11) % 151),
                                            static_cast<uchar>(((x + y) & 1) != 0 ? 220 : 5),
                                        };
                                    });
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// AutoContrast is independent per channel, so a planar input must produce the
// same pixels as the equivalent interleaved input. These cases run identical
// data through both layouts and require the re-interleaved planar output to
// match the interleaved output bit-for-bit.
// =============================================================================

namespace {

void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width, int height,
                               int numImages)
{
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, width, height, width, height, numImages,
        [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::AutoContrast op;
            EXPECT_NO_THROW(op(stream, src, dst));
        });
}

void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width, int height,
                                 int numImages)
{
    test::planar::RunVarShapeParity(planarFmt, interleavedFmt, width, height, width, height, numImages,
                                    [](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                       const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::AutoContrast op;
                                        EXPECT_NO_THROW(op(stream, src, dst));
                                    });
}

} // namespace

TEST(OpAutoContrast, bounded_grid_tensor_planar_matches_interleaved)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 1, BOUNDED_GRID_TEST_HEIGHT, 1);
}

TEST(OpAutoContrast, bounded_grid_varshape_planar_matches_interleaved)
{
    RunPlanarParityVarShapeCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 1, BOUNDED_GRID_TEST_HEIGHT, 1);
}

// Parameters: width, height, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpAutoContrastPlanar,
    test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {64, 48, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {37, 29, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {41, 33, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {35, 31, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {43, 27, 2, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16Up}, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16U}},
    {127, 9, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    {128, 7, 2, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16Up}, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16U}},
    {129, 5, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {255, 9, 2, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    {256, 7, 2, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16Up}, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16U}},
    {257, 5, 2, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
});

// clang-format on

TEST_P(OpAutoContrastPlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>());
}

TEST_P(OpAutoContrastPlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>());
}

// =============================================================================
// Negative / complement coverage: unsupported inputs must be rejected.
// =============================================================================

TEST(OpAutoContrast_Negative, cuda_graph_capture_is_rejected_before_enqueue)
{
    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cvcuda::AutoContrast op;
    nvcv::Tensor         srcTensor = nvcv::util::CreateTensor(1, 3, 2, nvcv::FMT_U8);
    nvcv::Tensor         dstTensor = nvcv::util::CreateTensor(1, 3, 2, nvcv::FMT_U8);

    nvcv::Image              srcImage({3, 2}, nvcv::FMT_U8);
    nvcv::Image              dstImage({3, 2}, nvcv::FMT_U8);
    nvcv::ImageBatchVarShape srcBatch(1);
    nvcv::ImageBatchVarShape dstBatch(1);
    srcBatch.pushBack(srcImage);
    dstBatch.pushBack(dstImage);

    auto expectRejectedWithoutNodes = [stream](const auto &submit)
    {
        ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        EXPECT_EQ(NVCV_ERROR_INVALID_OPERATION, nvcv::ProtectCall(submit));

        cudaGraph_t graph{};
        ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));
        size_t numNodes = 1;
        EXPECT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nullptr, &numNodes));
        EXPECT_EQ(0, numNodes);
        EXPECT_EQ(cudaSuccess, cudaGraphDestroy(graph));
    };

    expectRejectedWithoutNodes([&op, stream, &srcTensor, &dstTensor] { op(stream, srcTensor, dstTensor); });
    expectRejectedWithoutNodes([&op, stream, &srcBatch, &dstBatch] { op(stream, srcBatch, dstBatch); });

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAutoContrast_Negative, createWithNullHandle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaAutoContrastCreate(nullptr));
}

TEST(OpAutoContrast_Negative, zero_extent_varshape_image)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;
    for (const nvcv::Size2D size : {
             nvcv::Size2D{0, 1},
             nvcv::Size2D{1, 0}
    })
    {
        SCOPED_TRACE(testing::Message() << "size=" << size.w << 'x' << size.h);

        nvcv::Image              srcImage(size, nvcv::FMT_U8);
        nvcv::Image              dstImage(size, nvcv::FMT_U8);
        nvcv::ImageBatchVarShape src(1);
        nvcv::ImageBatchVarShape dst(1);
        src.pushBack(srcImage);
        dst.pushBack(dstImage);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAutoContrast_Negative, zero_extent_unsupported_dtype)
{
    ExpectZeroExtentTensorStatus(
        nvcv::TensorShape{
            {1, 0, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_S16, NVCV_ERROR_INVALID_ARGUMENT);
}

// clang-format off
#define NVCV_NEG_CASE(W, H, N, FMT) ttype::Types<ttype::Value<W>, ttype::Value<H>, ttype::Value<N>, ttype::Value<FMT>>

NVCV_TYPED_TEST_SUITE(
    OpAutoContrast_Negative,
    ttype::Types<
        NVCV_NEG_CASE(32, 32, 1, NVCV_IMAGE_FORMAT_S8),    // unsupported signed 8-bit
        NVCV_NEG_CASE(32, 32, 1, NVCV_IMAGE_FORMAT_S16),   // unsupported signed 16-bit
        NVCV_NEG_CASE(32, 32, 1, NVCV_IMAGE_FORMAT_S32),   // unsupported signed 32-bit
        NVCV_NEG_CASE(32, 32, 1, NVCV_IMAGE_FORMAT_F64),   // unsupported double
        NVCV_NEG_CASE(32, 32, 1, NVCV_IMAGE_FORMAT_2F32)>); // unsupported 2-channel

// clang-format on

TYPED_TEST(OpAutoContrast_Negative, unsupported_dtype_or_channels)
{
    const int               width      = ttype::GetValue<TypeParam, 0>;
    const int               height     = ttype::GetValue<TypeParam, 1>;
    const int               numSamples = ttype::GetValue<TypeParam, 2>;
    const nvcv::ImageFormat fmt{ttype::GetValue<TypeParam, 3>};

    nvcv::Tensor src = nvcv::util::CreateTensor(numSamples, width, height, fmt);
    nvcv::Tensor dst = nvcv::util::CreateTensor(numSamples, width, height, fmt);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAutoContrast_Negative, mismatched_tensors)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;

    // Different number of samples.
    {
        nvcv::Tensor src = nvcv::util::CreateTensor(2, 32, 32, nvcv::FMT_RGB8);
        nvcv::Tensor dst = nvcv::util::CreateTensor(1, 32, 32, nvcv::FMT_RGB8);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    }
    // Different width/height.
    {
        nvcv::Tensor src = nvcv::util::CreateTensor(1, 32, 32, nvcv::FMT_RGB8);
        nvcv::Tensor dst = nvcv::util::CreateTensor(1, 30, 32, nvcv::FMT_RGB8);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    }
    // Different layout (interleaved vs planar).
    {
        nvcv::Tensor src = nvcv::util::CreateTensor(1, 32, 32, nvcv::FMT_RGB8);
        nvcv::Tensor dst = nvcv::util::CreateTensor(1, 32, 32, nvcv::FMT_RGB8p);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    }
    // Different data type.
    {
        nvcv::Tensor src = nvcv::util::CreateTensor(1, 32, 32, nvcv::FMT_RGB8);
        nvcv::Tensor dst = nvcv::util::CreateTensor(1, 32, 32, nvcv::FMT_RGBf32);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAutoContrast_Negative, compound_tensor_dtype)
{
    nvcv::Tensor src(
        {
            {1, 4, 5, 3},
            "NHWC"
    },
        nvcv::TYPE_3U8);
    nvcv::Tensor dst(
        {
            {1, 4, 5, 3},
            "NHWC"
    },
        nvcv::TYPE_3U8);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAutoContrast_Negative, nonpacked_singleton_channel_tensor)
{
    constexpr int64_t bufferSize = 64;
    NVCVByte         *srcAllocation{};
    NVCVByte         *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bufferSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), bufferSize));

    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    srcBuffer.basePtr    = srcAllocation;
    srcBuffer.strides[0] = 32;
    srcBuffer.strides[1] = 16;
    srcBuffer.strides[2] = 2;
    srcBuffer.strides[3] = 2;
    auto dstBuffer       = srcBuffer;
    dstBuffer.basePtr    = dstAllocation;

    {
        nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, 2, 4, 1}, "NHWC"},
            nvcv::TYPE_U8, srcBuffer
        });
        nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, 2, 4, 1}, "NHWC"},
            nvcv::TYPE_U8, dstBuffer
        });

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        cvcuda::AutoContrast op;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpAutoContrast_Negative, dynamic_tensor_strides_must_fit_int32)
{
    constexpr int64_t oversizedStride = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    NVCVByte         *srcAllocation{};
    NVCVByte         *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), 1));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), 1));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AutoContrast op;

    for (const nvcv::TensorLayout layout : {nvcv::TENSOR_HWC, nvcv::TENSOR_CHW})
    {
        SCOPED_TRACE(layout);
        nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.basePtr    = srcAllocation;
        srcBuffer.strides[0] = oversizedStride;
        srcBuffer.strides[1] = 1;
        srcBuffer.strides[2] = 1;
        if (layout == nvcv::TENSOR_CHW)
        {
            srcBuffer.strides[1] = oversizedStride;
        }
        auto dstBuffer    = srcBuffer;
        dstBuffer.basePtr = dstAllocation;

        nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, 1, 1}, layout},
            nvcv::TYPE_U8, srcBuffer
        });
        nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, 1, 1}, layout},
            nvcv::TYPE_U8, dstBuffer
        });

        EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    }

    {
        SCOPED_TRACE("CHW plane stride");
        nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.basePtr    = srcAllocation;
        srcBuffer.strides[0] = oversizedStride;
        srcBuffer.strides[1] = 1;
        srcBuffer.strides[2] = 1;
        auto dstBuffer       = srcBuffer;
        dstBuffer.basePtr    = dstAllocation;

        nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{3, 1, 1}, "CHW"},
            nvcv::TYPE_U8, srcBuffer
        });
        nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{3, 1, 1}, "CHW"},
            nvcv::TYPE_U8, dstBuffer
        });

        EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    }

    {
        SCOPED_TRACE("maximum byte offset");
        nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.basePtr    = srcAllocation;
        srcBuffer.strides[0] = std::numeric_limits<int32_t>::max();
        srcBuffer.strides[1] = 1;
        srcBuffer.strides[2] = 1;
        auto dstBuffer       = srcBuffer;
        dstBuffer.basePtr    = dstAllocation;

        nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{2, 2, 1}, "HWC"},
            nvcv::TYPE_U8, srcBuffer
        });
        nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{2, 2, 1}, "HWC"},
            nvcv::TYPE_U8, dstBuffer
        });

        EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpAutoContrast_Negative, raw_tensor_dimensions_are_validated_before_narrowing)
{
    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), 1));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), 1));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AutoContrast op;
    auto                 expectStatus = [&](const nvcv::TensorShape &shape, NVCVStatus expected)
    {
        nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.basePtr = srcAllocation;
        for (int d = 0; d < shape.rank(); ++d)
        {
            srcBuffer.strides[d] = 1;
        }
        auto dstBuffer    = srcBuffer;
        dstBuffer.basePtr = dstAllocation;

        nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, srcBuffer});
        nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, dstBuffer});
        EXPECT_EQ(expected, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));
    };

    constexpr int64_t wrapsToOne = (int64_t{1} << 32) + 1;
    expectStatus(
        nvcv::TensorShape{
            {wrapsToOne, 1, 1, 1},
            "NHWC"
    },
        NVCV_ERROR_INVALID_ARGUMENT);
    expectStatus(
        nvcv::TensorShape{
            {wrapsToOne, 1, 1},
            "HWC"
    },
        NVCV_ERROR_INVALID_ARGUMENT);
    expectStatus(
        nvcv::TensorShape{
            {1, wrapsToOne, 1},
            "HWC"
    },
        NVCV_ERROR_OVERFLOW);
    expectStatus(
        nvcv::TensorShape{
            {1, 1, wrapsToOne},
            "HWC"
    },
        NVCV_ERROR_INVALID_ARGUMENT);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpAutoContrast_Negative, height_exceeding_cuda_grid_limit)
{
    constexpr int64_t height     = 4 * 65535 + 1;
    constexpr int64_t bufferSize = height;
    NVCVByte         *srcAllocation{};
    NVCVByte         *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bufferSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), bufferSize));

    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    srcBuffer.basePtr    = srcAllocation;
    srcBuffer.strides[0] = height;
    srcBuffer.strides[1] = 1;
    srcBuffer.strides[2] = 1;
    srcBuffer.strides[3] = 1;
    auto dstBuffer       = srcBuffer;
    dstBuffer.basePtr    = dstAllocation;

    {
        nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, height, 1, 1}, "NHWC"},
            nvcv::TYPE_U8, srcBuffer
        });
        nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, height, 1, 1}, "NHWC"},
            nvcv::TYPE_U8, dstBuffer
        });

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        cvcuda::AutoContrast op;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpAutoContrast_Negative, varshape_height_exceeding_cuda_grid_limit)
{
    constexpr int            height = 4 * 65535 + 1;
    nvcv::Image              srcImage({1, height}, nvcv::FMT_U8);
    nvcv::Image              dstImage({1, height}, nvcv::FMT_U8);
    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);
    src.pushBack(srcImage);
    dst.pushBack(dstImage);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAutoContrast_Negative, varshape_plane_offset_must_fit_int32)
{
    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), 1));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), 1));

    nvcv::ImageDataStridedCuda::Buffer srcBuffer{};
    srcBuffer.numPlanes           = 1;
    srcBuffer.planes[0].width     = 2;
    srcBuffer.planes[0].height    = 2;
    srcBuffer.planes[0].rowStride = std::numeric_limits<int32_t>::max();
    srcBuffer.planes[0].basePtr   = srcAllocation;
    auto dstBuffer                = srcBuffer;
    dstBuffer.planes[0].basePtr   = dstAllocation;

    {
        nvcv::Image              srcImage = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_U8, srcBuffer});
        nvcv::Image              dstImage = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_U8, dstBuffer});
        nvcv::ImageBatchVarShape srcBatch(1);
        nvcv::ImageBatchVarShape dstBatch(1);
        srcBatch.pushBack(srcImage);
        dstBatch.pushBack(dstImage);

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
        cvcuda::AutoContrast op;
        EXPECT_EQ(NVCV_ERROR_OVERFLOW,
                  nvcv::ProtectCall([&op, stream, &srcBatch, &dstBatch] { op(stream, srcBatch, dstBatch); }));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpAutoContrast, launch_grid_arithmetic_does_not_overflow)
{
    constexpr int64_t maxWidth = std::numeric_limits<int32_t>::max();
    EXPECT_EQ(8388608, cvcuda::priv::detail::AutoContrastGridX(maxWidth, 32, 8));
    EXPECT_EQ(16777216, cvcuda::priv::detail::AutoContrastGridX(maxWidth, 32, 4));
    EXPECT_EQ(67108864, cvcuda::priv::detail::AutoContrastGridX(maxWidth, 32, 4, 4));
}

TEST(OpAutoContrast_Negative, varshape_plane_descriptors_must_match_format)
{
    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), 1));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), 1));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AutoContrast op;
    auto                 expectInvalid = [&](nvcv::ImageDataStridedCuda::Buffer srcBuffer)
    {
        auto dstBuffer = srcBuffer;
        for (int p = 0; p < NVCV_MAX_PLANE_COUNT; ++p)
        {
            if (srcBuffer.planes[p].width > 0)
            {
                srcBuffer.planes[p].basePtr = srcAllocation;
                dstBuffer.planes[p].basePtr = dstAllocation;
            }
        }

        nvcv::Image              srcImage = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8p, srcBuffer});
        nvcv::Image              dstImage = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8p, dstBuffer});
        nvcv::ImageBatchVarShape srcBatch(1);
        nvcv::ImageBatchVarShape dstBatch(1);
        srcBatch.pushBack(srcImage);
        dstBatch.pushBack(dstImage);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&op, stream, &srcBatch, &dstBatch] { op(stream, srcBatch, dstBatch); }));
    };

    nvcv::ImageDataStridedCuda::Buffer missingPlanes{};
    missingPlanes.numPlanes = 1;
    for (int p = 0; p < 3; ++p)
    {
        missingPlanes.planes[p].width     = 1;
        missingPlanes.planes[p].height    = 1;
        missingPlanes.planes[p].rowStride = 1;
    }
    expectInvalid(missingPlanes);

    nvcv::ImageDataStridedCuda::Buffer wrongPlaneSize{};
    wrongPlaneSize.numPlanes           = 3;
    wrongPlaneSize.planes[0].width     = 1;
    wrongPlaneSize.planes[0].height    = 1;
    wrongPlaneSize.planes[0].rowStride = 1;
    for (int p = 1; p < wrongPlaneSize.numPlanes; ++p)
    {
        wrongPlaneSize.planes[p].width     = 2;
        wrongPlaneSize.planes[p].height    = 2;
        wrongPlaneSize.planes[p].rowStride = 2;
    }
    expectInvalid(wrongPlaneSize);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpAutoContrast_Negative, different_format_varshape)
{
    const int    numSamples = 4;
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int s = 0; s < numSamples - 1; ++s)
    {
        imgSrc.emplace_back(nvcv::Size2D{32, 32}, nvcv::FMT_RGB8);
        imgDst.emplace_back(nvcv::Size2D{32, 32}, nvcv::FMT_RGB8);
    }
    // Last image has a different format -> non-unique batch format must be rejected.
    imgSrc.emplace_back(nvcv::Size2D{32, 32}, nvcv::FMT_RGBA8);
    imgDst.emplace_back(nvcv::Size2D{32, 32}, nvcv::FMT_RGBA8);

    nvcv::ImageBatchVarShape batchSrc(numSamples);
    nvcv::ImageBatchVarShape batchDst(numSamples);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::AutoContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, stream, &batchSrc, &batchDst] { op(stream, batchSrc, batchDst); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void ExpectRejectsSubsampledVarShape(const char (&fourcc)[5], nvcv::ChromaSubsampling expectedCss,
                                            nvcv::Size2D expectedChromaSize)
{
    constexpr nvcv::Size2D  size{64, 48};
    const nvcv::ImageFormat fmt = nvcv::ImageFormat::FromFourCC(FourCC(fourcc), nvcv::CSPEC_BT601, nvcv::MemLayout::PL);

    SCOPED_TRACE(fourcc);
    ASSERT_EQ(3, fmt.numChannels());
    ASSERT_EQ(3, fmt.numPlanes());
    ASSERT_EQ(expectedCss, fmt.chromaSubsampling());
    for (int p = 0; p < 3; ++p)
    {
        ASSERT_EQ(nvcv::TYPE_U8, fmt.planeDataType(p));
    }
    ASSERT_EQ(expectedChromaSize, fmt.planeSize(size, 1));

    nvcv::Image              srcImage(size, fmt);
    nvcv::Image              dstImage(size, fmt);
    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);
    src.pushBack(srcImage);
    dst.pushBack(dstImage);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::AutoContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAutoContrast_Negative, i420_varshape)
{
    ExpectRejectsSubsampledVarShape("I420", nvcv::ChromaSubsampling::CSS_420, {32, 24});
}

TEST(OpAutoContrast_Negative, yv16_varshape)
{
    ExpectRejectsSubsampledVarShape("YV16", nvcv::ChromaSubsampling::CSS_422, {32, 48});
}
