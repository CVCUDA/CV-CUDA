/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "HalfTestUtils.hpp"
#include "PlanarParityUtils.hpp"
#include "TestCAPIEnum.h"

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cuda_fp16.h>
#include <cvcuda/OpBrightnessContrast.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

using uchar = unsigned char;

using BrightnessContrastTensorScalarCall
    = void (cvcuda::BrightnessContrast::*)(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &, double, double,
                                           double, double, bool) const;
using BrightnessContrastVarShapeScalarCall
    = void (cvcuda::BrightnessContrast::*)(cudaStream_t, const nvcv::ImageBatch &, const nvcv::ImageBatch &, double,
                                           double, double, double, bool) const;
static_assert(
    std::is_same_v<decltype(static_cast<BrightnessContrastTensorScalarCall>(&cvcuda::BrightnessContrast::operator())),
                   BrightnessContrastTensorScalarCall>);
static_assert(
    std::is_same_v<decltype(static_cast<BrightnessContrastVarShapeScalarCall>(&cvcuda::BrightnessContrast::operator())),
                   BrightnessContrastVarShapeScalarCall>);

// __half cannot parameterize std::uniform_real_distribution, so F16 rows sample floats in the
// same [0, 1] range the float rows use and quantize to half on assignment.
template<typename T>
using DistValueType = std::conditional_t<std::is_same_v<T, __half>, float, T>;

template<typename T>
using uniform_distribution = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>,
                                                std::uniform_real_distribution<DistValueType<T>>>;

// Builds the value distribution without naming TypeTraits<T>::max for __half, which has no
// constexpr max member (non-literal type).
template<typename T>
uniform_distribution<T> MakeValueDistribution()
{
    using DT = DistValueType<T>;
    if constexpr (std::is_integral_v<T>)
    {
        return uniform_distribution<T>(T{0}, cuda::TypeTraits<T>::max);
    }
    else
    {
        return uniform_distribution<T>(DT{0}, DT{1});
    }
}

template<typename ValueType>
void ComparePixel(std::vector<uint8_t> &dst, std::vector<uint8_t> &ref, const long4_16a &strides, const int4 &coord,
                  float tolerance, bool requireExact)
{
    auto dstPixel = test::ValueAt<ValueType>(dst, strides, coord);
    auto refPixel = test::ValueAt<ValueType>(ref, strides, coord);
    for (int k = 0; k < cuda::NumElements<ValueType>; ++k)
    {
        if constexpr (std::is_same_v<cuda::BaseType<ValueType>, __half>)
        {
            // F16 outputs: the reference evaluates the same single mul-add chain in the float
            // ArgT and rounds once to half, mirroring the kernel; per the HalfTestUtils.hpp
            // policy the tolerance is kUlps = 1 half-ULP at the reference magnitude (CPU and
            // GPU float chains may round differently, e.g. through FMA contraction).
            const auto refValue = static_cast<float>(cuda::GetElement(refPixel, k));
            EXPECT_NEAR(static_cast<float>(cuda::GetElement(dstPixel, k)), refValue,
                        tolerance * test::HalfUlp(refValue));
        }
        else if (requireExact || std::is_same_v<cuda::BaseType<ValueType>, long long>)
        {
            EXPECT_EQ(cuda::GetElement(dstPixel, k), cuda::GetElement(refPixel, k));
        }
        else
        {
            EXPECT_NEAR(cuda::GetElement(dstPixel, k), cuda::GetElement(refPixel, k), tolerance);
        }
    }
}

template<typename ValueType>
void CompareTensors(std::vector<uint8_t> &dst, std::vector<uint8_t> &ref, const long4_16a &strides, const int3 &shape,
                    int numPlanes, float tolerance, bool requireExact = false)
{
    const int numPixels = shape.x * shape.y;
    for (int z = 0; z < shape.z; ++z)
    {
        for (int p = 0; p < numPlanes; p++)
        {
            for (int idx = 0; idx < numPixels; ++idx)
            {
                ComparePixel<ValueType>(dst, ref, strides, int4{idx % shape.x, idx / shape.x, p, z}, tolerance,
                                        requireExact);
            }
        }
    }
}

template<typename ValueType>
inline nvcv::Tensor GetArgTensor(int numSamples)
{
    static_assert(std::is_same_v<float, ValueType> || std::is_same_v<double, ValueType>);
    auto dType = std::is_same_v<float, ValueType> ? nvcv::TYPE_F32 : nvcv::TYPE_F64;
    if (numSamples == 0)
    {
        return nvcv::Tensor{nullptr};
    }
    else
    {
        nvcv::TensorShape shape{{numSamples}, "N"};
        return nvcv::Tensor{shape, dType};
    }
}

template<typename ValueType, typename Ret>
Ret GetHalfRange()
{
    if constexpr (std::is_same_v<ValueType, uchar> || std::is_same_v<ValueType, signed char>)
    {
        return static_cast<Ret>(128.);
    }
    else if constexpr (std::is_same_v<ValueType, unsigned short>)
    {
        return static_cast<Ret>(32768.);
    }
    else if constexpr (std::is_same_v<ValueType, short>)
    {
        return static_cast<Ret>(16384.);
    }
    else if constexpr (std::is_same_v<ValueType, unsigned int>)
    {
        return static_cast<Ret>(2147483648.);
    }
    else if constexpr (std::is_same_v<ValueType, int>)
    {
        return static_cast<Ret>(1073741824);
    }
    else if constexpr (std::is_same_v<ValueType, long long>)
    {
        // 2^62 = T{1} << (64 - 1 - 1) — exactly representable as double (power of two).
        return static_cast<Ret>(4611686018427387904LL);
    }
    else
    {
        static_assert(!std::is_integral_v<ValueType>);
        {
            return static_cast<Ret>(0.5);
        }
    }
}

template<typename BT_>
struct Argument
{
    using BT = BT_;

    template<typename Rng>
    inline void populate(Rng &rng, int numSamples, BT lo, BT hi, BT defaultVal)
    {
        m_numSamples = numSamples;
        m_default    = defaultVal;
        m_argTensor  = GetArgTensor<BT>(numSamples);
        if (numSamples > 0)
        {
            auto argData = m_argTensor.exportData<nvcv::TensorDataStridedCuda>();
            ASSERT_TRUE(argData);

            m_stride          = argData->stride(0);
            size_t argBufSize = m_stride * numSamples;
            m_argVec          = std::vector<uint8_t>(argBufSize, uint8_t{0});

            std::uniform_real_distribution<BT> coeffDist(lo, hi);
            for (int z = 0; z < numSamples; ++z)
            {
                const BT   value = coeffDist(rng);
                const auto offs  = static_cast<size_t>(m_stride) * static_cast<size_t>(z);
                std::memcpy(m_argVec.data() + offs, &value, sizeof(value));
            }

            if (numSamples)
            {
                ASSERT_EQ(cudaSuccess,
                          cudaMemcpy(argData->basePtr(), m_argVec.data(), argBufSize, cudaMemcpyHostToDevice));
            }
        }
    }

    BT GetHostElement(int idx) const
    {
        if (m_numSamples == 0)
        {
            return m_default;
        }
        else if (m_numSamples == 1)
        {
            BT value{};
            std::memcpy(&value, m_argVec.data(), sizeof(value));
            return value;
        }
        else
        {
            BT         value{};
            const auto offs = static_cast<size_t>(m_stride) * static_cast<size_t>(idx);
            std::memcpy(&value, m_argVec.data() + offs, sizeof(value));
            return value;
        }
    }

    BT                   m_default;
    int                  m_numSamples;
    nvcv::Tensor         m_argTensor;
    long                 m_stride = 0;
    std::vector<uint8_t> m_argVec = {};
};

template<typename ArgType>
struct BrightnessContrastArguments
{
    template<typename SrcBT, typename DstBT, typename Rng, typename SrcDist, typename DstDist>
    void populate(Rng &rng, const int4 &argCounts, SrcDist &srcRand, DstDist &dstRand)
    {
        ArgType normalizationFactor = GetHalfRange<DstBT, ArgType>() / GetHalfRange<SrcBT, ArgType>();
        brightness.populate(rng, argCounts.x, static_cast<ArgType>(0), static_cast<ArgType>(2) * normalizationFactor,
                            static_cast<ArgType>(1));
        contrast.populate(rng, argCounts.y, static_cast<ArgType>(0), static_cast<ArgType>(2), static_cast<ArgType>(1));
        brightnessShift.populate(rng, argCounts.z, -static_cast<ArgType>(dstRand(rng)) / static_cast<ArgType>(2),
                                 static_cast<ArgType>(dstRand(rng)) / static_cast<ArgType>(2), static_cast<ArgType>(0));
        contrastCenter.populate(rng, argCounts.w, static_cast<ArgType>(0), static_cast<ArgType>(srcRand(rng)),
                                GetHalfRange<SrcBT, ArgType>());
    }

    Argument<ArgType> brightness;
    Argument<ArgType> contrast;
    Argument<ArgType> brightnessShift;
    Argument<ArgType> contrastCenter;
};

template<typename DstT, typename ArgT>
DstT RoundSaturatedS64(ArgT value)
{
    if (constexpr auto hi = std::numeric_limits<DstT>::max(); value >= static_cast<ArgT>(hi))
        return hi;
    if (constexpr auto lo = std::numeric_limits<DstT>::lowest(); value <= static_cast<ArgT>(lo))
        return lo;
    return static_cast<DstT>(std::nearbyint(value));
}

template<typename SrcType, typename DstType, typename ArgT>
void ApplyBrightnessContrastPixel(std::vector<uint8_t> &src, std::vector<uint8_t> &dst, const long4_16a &srcStrides,
                                  const long4_16a &dstStrides, const int4 &coord, ArgT brightness, ArgT contrast,
                                  ArgT brightnessShift, ArgT contrastCenter)
{
    using DstBT               = cuda::BaseType<DstType>;
    constexpr int numChannels = cuda::NumElements<DstType>;

    auto  srcPixel = test::ValueAt<SrcType>(src, srcStrides, coord);
    auto &outPixel = test::ValueAt<DstType>(dst, dstStrides, coord);
    for (int k = 0; k < numChannels; k++)
    {
        auto v = static_cast<ArgT>(cuda::GetElement(srcPixel, k));
        if constexpr (std::is_same_v<DstBT, long long>)
        {
            // Explicit fusion keeps the integer oracle independent of host FMA availability.
            const ArgT difference = v - contrastCenter;
            v = std::fma(brightness, std::fma(contrast, difference, contrastCenter), brightnessShift);
            cuda::GetElement(outPixel, k) = RoundSaturatedS64<DstBT>(v);
        }
        else
        {
            v = brightness * (contrast * (v - contrastCenter) + contrastCenter) + brightnessShift;
            cuda::GetElement(outPixel, k) = cuda::SaturateCast<DstBT>(v);
        }
    }
}

template<typename SrcType, typename DstType, typename ArgT>
void BrightnessContrast(std::vector<uint8_t> &src, std::vector<uint8_t> &dst, const long4_16a &srcStrides,
                        const long4_16a &dstStrides, const int3 &shape, int numPlanes, int sampleIdx, ArgT brightness,
                        ArgT contrast, ArgT brightnessShift, ArgT contrastCenter)
{
    constexpr int numChannels = cuda::NumElements<DstType>;
    static_assert(cuda::NumElements<SrcType> == numChannels);

    const int numPixels = shape.x * shape.y;
    for (int p = 0; p < numPlanes; p++)
    {
        for (int idx = 0; idx < numPixels; ++idx)
        {
            ApplyBrightnessContrastPixel<SrcType, DstType>(src, dst, srcStrides, dstStrides,
                                                           int4{idx % shape.x, idx / shape.x, p, sampleIdx}, brightness,
                                                           contrast, brightnessShift, contrastCenter);
        }
    }
}

template<typename SrcType, typename Strides, typename Coord, typename Distribution, typename Rng>
void FillRandomPixel(std::vector<uint8_t> &src, const Strides &strides, const Coord &coord, Distribution &srcRand,
                     Rng &rng)
{
    auto &pixel = test::ValueAt<SrcType>(src, strides, coord);
    for (int k = 0; k < cuda::NumElements<SrcType>; ++k)
    {
        using ElementType          = std::remove_reference_t<decltype(cuda::GetElement(pixel, k))>;
        cuda::GetElement(pixel, k) = static_cast<ElementType>(srcRand(rng));
    }
}

#define NVCV_SHAPE(w, h, n) (int3{w, h, n})
#define NVCV_ARGS_COUNT(brightness, contrast, brightnessShift, contrastCenter) \
    (int4{brightness, contrast, brightnessShift, contrastCenter})

#define NVCV_CASE(SrcDstShape, SrcType, DstType, SrcImgFormat, DstImgFormat, ArgType, ArgCounts)                      \
    ttype::Types<ttype::Value<SrcDstShape>, SrcType, DstType, ttype::Value<SrcImgFormat>, ttype::Value<DstImgFormat>, \
                 ArgType, ttype::Value<ArgCounts>>

#define NVCV_IMAGE_FORMAT_RGB16S NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGB16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGBA16S \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)
#define NVCV_IMAGE_FORMAT_RGB32S NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X32_Y32_Z32)
#define NVCV_IMAGE_FORMAT_RGBA32S \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZW, ASSOCIATED, X32_Y32_Z32_W32)
#define NVCV_IMAGE_FORMAT_S16    NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X16)
#define NVCV_IMAGE_FORMAT_S64    NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X64)
#define NVCV_IMAGE_FORMAT_RGB64S NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X64_Y64_Z64)
#define NVCV_IMAGE_FORMAT_F64    NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, X000, ASSOCIATED, X64)
#define NVCV_IMAGE_FORMAT_RGBf64 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZ1, ASSOCIATED, X64_Y64_Z64)

// clang-format off
NVCV_TYPED_TEST_SUITE(OpBrightnessContrast,
                      ttype::Types<NVCV_CASE(NVCV_SHAPE(41, 39, 1), uchar, uchar, NVCV_IMAGE_FORMAT_U8,
                                             NVCV_IMAGE_FORMAT_U8, float, NVCV_ARGS_COUNT(0, 1, 1, 1)),
                                   // Identity arguments give an independent bit-exact oracle on odd widths. Together,
                                   // these cover scalar/interleaved/planar addressing in both Tensor and VarShape tests.
                                   NVCV_CASE(NVCV_SHAPE(37, 29, 3), uchar, uchar, NVCV_IMAGE_FORMAT_U8,
                                             NVCV_IMAGE_FORMAT_U8, float, NVCV_ARGS_COUNT(0, 0, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(37, 29, 3), uchar3, uchar3, NVCV_IMAGE_FORMAT_RGB8,
                                             NVCV_IMAGE_FORMAT_RGB8, float, NVCV_ARGS_COUNT(0, 0, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(37, 29, 3), uchar, uchar, NVCV_IMAGE_FORMAT_RGB8p,
                                             NVCV_IMAGE_FORMAT_RGB8p, float, NVCV_ARGS_COUNT(0, 0, 0, 0)),
                                   // Packed-row widths (the cases above are all row-padded).
                                   NVCV_CASE(NVCV_SHAPE(64, 48, 4), uchar3, uchar3, NVCV_IMAGE_FORMAT_RGB8,
                                             NVCV_IMAGE_FORMAT_RGB8, float, NVCV_ARGS_COUNT(0, 0, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(64, 16, 2), uchar4, uchar4, NVCV_IMAGE_FORMAT_RGBA8,
                                             NVCV_IMAGE_FORMAT_RGBA8, float, NVCV_ARGS_COUNT(0, 0, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(32, 32, 3), uchar, uchar, NVCV_IMAGE_FORMAT_RGB8p,
                                             NVCV_IMAGE_FORMAT_RGB8p, float, NVCV_ARGS_COUNT(0, 0, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(32, 32, 2), float3, float3, NVCV_IMAGE_FORMAT_RGBf32,
                                             NVCV_IMAGE_FORMAT_RGBf32, float, NVCV_ARGS_COUNT(0, 0, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(42, 17, 1), uchar, short, NVCV_IMAGE_FORMAT_U8,
                                             NVCV_IMAGE_FORMAT_S16, float, NVCV_ARGS_COUNT(1, 0, 1, 1)),
                                   NVCV_CASE(NVCV_SHAPE(42, 17, 1), uchar, ushort, NVCV_IMAGE_FORMAT_U8,
                                             NVCV_IMAGE_FORMAT_U16, float, NVCV_ARGS_COUNT(1, 1, 0, 1)),
                                   NVCV_CASE(NVCV_SHAPE(256, 256, 1), uchar, int, NVCV_IMAGE_FORMAT_U8,
                                             NVCV_IMAGE_FORMAT_S32, double, NVCV_ARGS_COUNT(1, 1, 1, 0)),
                                   NVCV_CASE(NVCV_SHAPE(101, 107, 1), uchar, float, NVCV_IMAGE_FORMAT_U8,
                                             NVCV_IMAGE_FORMAT_F32, float, NVCV_ARGS_COUNT(1, 1, 1, 1)),
                                   NVCV_CASE(NVCV_SHAPE(17, 256, 2), short, short, NVCV_IMAGE_FORMAT_S16,
                                             NVCV_IMAGE_FORMAT_S16, float, NVCV_ARGS_COUNT(2, 2, 2, 2)),
                                   NVCV_CASE(NVCV_SHAPE(256, 17, 3), int, int, NVCV_IMAGE_FORMAT_S32,
                                             NVCV_IMAGE_FORMAT_S32, double, NVCV_ARGS_COUNT(3, 3, 3, 1)),
                                   NVCV_CASE(NVCV_SHAPE(101, 47, 4), float, float, NVCV_IMAGE_FORMAT_F32,
                                             NVCV_IMAGE_FORMAT_F32, float, NVCV_ARGS_COUNT(4, 4, 1, 4)),
                                   NVCV_CASE(NVCV_SHAPE(51, 103, 5), short2, short2, NVCV_IMAGE_FORMAT_2S16,
                                             NVCV_IMAGE_FORMAT_2S16, float, NVCV_ARGS_COUNT(5, 1, 5, 5)),
                                   NVCV_CASE(NVCV_SHAPE(41, 42, 6), float2, float2, NVCV_IMAGE_FORMAT_2F32,
                                             NVCV_IMAGE_FORMAT_2F32, float, NVCV_ARGS_COUNT(1, 6, 6, 6)),
                                   NVCV_CASE(NVCV_SHAPE(42, 41, 7), uchar3, uchar3, NVCV_IMAGE_FORMAT_RGB8,
                                             NVCV_IMAGE_FORMAT_RGB8, float, NVCV_ARGS_COUNT(7, 7, 7, 7)),
                                   NVCV_CASE(NVCV_SHAPE(128, 128, 7), uchar3, float3, NVCV_IMAGE_FORMAT_RGB8,
                                             NVCV_IMAGE_FORMAT_RGBf32, float, NVCV_ARGS_COUNT(7, 7, 7, 7)),
                                   NVCV_CASE(NVCV_SHAPE(64, 64, 1), uchar3, uchar3, NVCV_IMAGE_FORMAT_RGB8,
                                             NVCV_IMAGE_FORMAT_RGB8, float, NVCV_ARGS_COUNT(1, 1, 1, 1)),
                                   NVCV_CASE(NVCV_SHAPE(10, 10, 8), uchar, uchar, NVCV_IMAGE_FORMAT_RGB8p,
                                             NVCV_IMAGE_FORMAT_RGB8p, float, NVCV_ARGS_COUNT(8, 8, 8, 8)),
                                   NVCV_CASE(NVCV_SHAPE(201, 101, 1), uchar, uchar, NVCV_IMAGE_FORMAT_RGB8p,
                                             NVCV_IMAGE_FORMAT_RGB8p, float, NVCV_ARGS_COUNT(1, 1, 1, 0)),
                                   NVCV_CASE(NVCV_SHAPE(101, 10, 9), short3, short3, NVCV_IMAGE_FORMAT_RGB16S,
                                             NVCV_IMAGE_FORMAT_RGB16S, float, NVCV_ARGS_COUNT(1, 1, 1, 1)),
                                   NVCV_CASE(NVCV_SHAPE(101, 10, 9), ushort3, ushort3, NVCV_IMAGE_FORMAT_RGB16U,
                                             NVCV_IMAGE_FORMAT_RGB16U, float, NVCV_ARGS_COUNT(9, 9, 9, 9)),
                                   NVCV_CASE(NVCV_SHAPE(79, 10, 10), int3, int3, NVCV_IMAGE_FORMAT_RGB32S,
                                             NVCV_IMAGE_FORMAT_RGB32S, double, NVCV_ARGS_COUNT(10, 10, 10, 10)),
                                   NVCV_CASE(NVCV_SHAPE(10, 10, 11), float3, float3, NVCV_IMAGE_FORMAT_RGBf32,
                                             NVCV_IMAGE_FORMAT_RGBf32, float, NVCV_ARGS_COUNT(11, 11, 11, 11)),
                                   NVCV_CASE(NVCV_SHAPE(10, 10, 11), float3, float3, NVCV_IMAGE_FORMAT_RGBf32,
                                             NVCV_IMAGE_FORMAT_RGBf32, float, NVCV_ARGS_COUNT(0, 0, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(59, 77, 11), float3, uchar3, NVCV_IMAGE_FORMAT_RGBf32,
                                             NVCV_IMAGE_FORMAT_RGB8, float, NVCV_ARGS_COUNT(11, 11, 0, 1)),
                                   NVCV_CASE(NVCV_SHAPE(101, 10, 12), float, float, NVCV_IMAGE_FORMAT_RGBf32p,
                                             NVCV_IMAGE_FORMAT_RGBf32p, float, NVCV_ARGS_COUNT(12, 1, 0, 12)),
                                   NVCV_CASE(NVCV_SHAPE(128, 127, 12), float, uchar, NVCV_IMAGE_FORMAT_RGBf32p,
                                             NVCV_IMAGE_FORMAT_RGB8p, float, NVCV_ARGS_COUNT(1, 1, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(17, 17, 13), uchar4, uchar4, NVCV_IMAGE_FORMAT_RGBA8,
                                             NVCV_IMAGE_FORMAT_RGBA8, float, NVCV_ARGS_COUNT(13, 13, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(127, 128, 14), uchar, uchar, NVCV_IMAGE_FORMAT_BGRA8p,
                                             NVCV_IMAGE_FORMAT_BGRA8p, float, NVCV_ARGS_COUNT(14, 1, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(9, 9, 15), short4, short4, NVCV_IMAGE_FORMAT_RGBA16S,
                                             NVCV_IMAGE_FORMAT_RGBA16S, float, NVCV_ARGS_COUNT(15, 0, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(31, 127, 16), int4, int4, NVCV_IMAGE_FORMAT_RGBA32S,
                                             NVCV_IMAGE_FORMAT_RGBA32S, double, NVCV_ARGS_COUNT(0, 16, 0, 0)),
                                   NVCV_CASE(NVCV_SHAPE(128, 32, 17), float, float, NVCV_IMAGE_FORMAT_RGBAf32p,
                                             NVCV_IMAGE_FORMAT_RGBAf32p, float, NVCV_ARGS_COUNT(0, 0, 17, 0)),
                                   NVCV_CASE(NVCV_SHAPE(32, 128, 18), float4, float4, NVCV_IMAGE_FORMAT_RGBAf32,
                                             NVCV_IMAGE_FORMAT_RGBAf32, float, NVCV_ARGS_COUNT(0, 0, 0, 18)),
                                   // F16 rows use F32 args and are validated within 1 half-ULP (see ComparePixel):
                                   // interleaved RGB (vectorized uint3-pack path), single channel, planar, RGBA
                                   // (half4 pack), and both mixed-type directions.
                                   NVCV_CASE(NVCV_SHAPE(41, 27, 3), half3, half3, NVCV_IMAGE_FORMAT_RGBf16,
                                             NVCV_IMAGE_FORMAT_RGBf16, float, NVCV_ARGS_COUNT(3, 3, 3, 3)),
                                   NVCV_CASE(NVCV_SHAPE(37, 19, 2), __half, __half, NVCV_IMAGE_FORMAT_F16,
                                             NVCV_IMAGE_FORMAT_F16, float, NVCV_ARGS_COUNT(2, 1, 2, 0)),
                                   NVCV_CASE(NVCV_SHAPE(23, 18, 3), __half, __half, NVCV_IMAGE_FORMAT_RGBf16p,
                                             NVCV_IMAGE_FORMAT_RGBf16p, float, NVCV_ARGS_COUNT(3, 3, 3, 3)),
                                   NVCV_CASE(NVCV_SHAPE(17, 17, 2), half4, half4, NVCV_IMAGE_FORMAT_RGBAf16,
                                             NVCV_IMAGE_FORMAT_RGBAf16, float, NVCV_ARGS_COUNT(2, 2, 0, 2)),
                                   NVCV_CASE(NVCV_SHAPE(31, 13, 2), uchar3, half3, NVCV_IMAGE_FORMAT_RGB8,
                                             NVCV_IMAGE_FORMAT_RGBf16, float, NVCV_ARGS_COUNT(2, 2, 2, 2)),
                                   NVCV_CASE(NVCV_SHAPE(29, 14, 2), half3, float3, NVCV_IMAGE_FORMAT_RGBf16,
                                             NVCV_IMAGE_FORMAT_RGBf32, float, NVCV_ARGS_COUNT(2, 2, 2, 2)),
                                   // S64 1ch: long long src/dst; double args (RequiresDouble<long long>=true);
                                   // exercises scalar kernel path.
                                   NVCV_CASE(NVCV_SHAPE(17, 13, 1), long long, long long, NVCV_IMAGE_FORMAT_S64,
                                             NVCV_IMAGE_FORMAT_S64, double, NVCV_ARGS_COUNT(1, 1, 1, 1)),
                                   // S64 3ch: longlong3 — BC_NIX<longlong3>=0, exercises the kNix==0 scalar fallback.
                                   NVCV_CASE(NVCV_SHAPE(17, 13, 2), longlong3, longlong3, NVCV_IMAGE_FORMAT_RGB64S,
                                             NVCV_IMAGE_FORMAT_RGB64S, double, NVCV_ARGS_COUNT(2, 2, 2, 2)),
                                   // F64 1ch: double src/dst; double args (RequiresDouble<double>=true).
                                   NVCV_CASE(NVCV_SHAPE(17, 13, 1), double, double, NVCV_IMAGE_FORMAT_F64,
                                             NVCV_IMAGE_FORMAT_F64, double, NVCV_ARGS_COUNT(1, 1, 1, 1)),
                                   // F64 3ch: double3 — BC_NIX<double3>=0, exercises the kNix==0 scalar fallback.
                                   NVCV_CASE(NVCV_SHAPE(17, 13, 2), double3, double3, NVCV_IMAGE_FORMAT_RGBf64,
                                             NVCV_IMAGE_FORMAT_RGBf64, double, NVCV_ARGS_COUNT(2, 2, 2, 2))>);

// clang-format on

TYPED_TEST(OpBrightnessContrast, correct_output)
{
    const int3 shape = ttype::GetValue<TypeParam, 0>;
    using SrcType    = ttype::GetType<TypeParam, 1>;
    using DstType    = ttype::GetType<TypeParam, 2>;
    using SrcBT      = cuda::BaseType<SrcType>;
    using DstBT      = cuda::BaseType<DstType>;

    const nvcv::ImageFormat srcImgFormat{ttype::GetValue<TypeParam, 3>};
    const nvcv::ImageFormat dstImgFormat{ttype::GetValue<TypeParam, 4>};
    const int               numChannels = cuda::NumElements<SrcType>;
    static_assert(1 <= numChannels && numChannels <= 4);
    static_assert(numChannels == cuda::NumElements<DstType>);

    using ArgType        = ttype::GetType<TypeParam, 5>;
    const int4 argCounts = ttype::GetValue<TypeParam, 6>;

    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, srcImgFormat);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, dstImgFormat);

    auto srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);

    ASSERT_TRUE(srcAccess && dstAccess);
    int numPlanes = srcAccess->numPlanes();
    ASSERT_EQ(numPlanes, dstAccess->numPlanes());
    ASSERT_TRUE(numChannels == 1 || numPlanes == 1);
    auto numSamples = static_cast<int>(srcAccess->numSamples());
    ASSERT_EQ(numSamples, dstAccess->numSamples());
    for (auto argCount : {argCounts.x, argCounts.y, argCounts.z, argCounts.w})
    {
        ASSERT_TRUE(argCount == 0 || argCount == 1 || argCount == numSamples);
    }
    int numRows = dstAccess->numRows();
    ASSERT_EQ(srcAccess->numRows(), numRows);
    ASSERT_TRUE(srcAccess->numCols() == dstAccess->numCols() && srcAccess->numChannels() == dstAccess->numChannels());
    long4_16a srcStrides{srcAccess->sampleStride(), srcAccess->planeStride(), srcAccess->rowStride(),
                         srcAccess->colStride()};
    long4_16a dstStrides{dstAccess->sampleStride(), dstAccess->planeStride(), dstAccess->rowStride(),
                         dstAccess->colStride()};

    // if the image is not planar, the stride is set 0 by `Access` helper,
    // compute the "proper" stride
    if (srcStrides.y == 0 || dstStrides.y == 0)
    {
        ASSERT_EQ(dstStrides.y, srcStrides.y);
        srcStrides.y = srcStrides.z * numRows;
        dstStrides.y = dstStrides.z * numRows;
    }
    // if the tensor represents only a single image (N=1), the plane stride is set to 0,
    // replace it with the "proper" one
    if (srcStrides.x == 0 || dstStrides.x == 0)
    {
        ASSERT_EQ(dstStrides.x, srcStrides.x);
        srcStrides.x = srcStrides.y * numPlanes;
        dstStrides.x = dstStrides.y * numPlanes;
    }

    size_t               srcBufSize = srcStrides.x * numSamples;
    size_t               dstBufSize = dstStrides.x * numSamples;
    std::vector<uint8_t> srcVec(srcBufSize, uint8_t{0});
    std::vector<uint8_t> dstVec(dstBufSize, uint8_t{0});
    std::vector<uint8_t> refVec(dstBufSize, uint8_t{0});

    auto            srcRand = MakeValueDistribution<SrcBT>();
    auto            dstRand = MakeValueDistribution<DstBT>();
    std::mt19937_64 rng(12345);

    const int numPixels = shape.x * shape.y;
    for (int z = 0; z < shape.z; ++z)
    {
        for (int p = 0; p < numPlanes; p++)
        {
            for (int idx = 0; idx < numPixels; ++idx)
            {
                FillRandomPixel<SrcType>(srcVec, srcStrides, int4{idx % shape.x, idx / shape.x, p, z}, srcRand, rng);
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), srcBufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    BrightnessContrastArguments<ArgType> args;
    args.template populate<SrcBT, DstBT>(rng, argCounts, srcRand, dstRand);

    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, srcTensor, dstTensor, args.brightness.m_argTensor, args.contrast.m_argTensor,
                       args.brightnessShift.m_argTensor, args.contrastCenter.m_argTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstVec.data(), dstData->basePtr(), dstBufSize, cudaMemcpyDeviceToHost));

    for (int z = 0; z < shape.z; z++)
    {
        BrightnessContrast<SrcType, DstType>(srcVec, refVec, srcStrides, dstStrides, shape, numPlanes, z,
                                             args.brightness.GetHostElement(z), args.contrast.GetHostElement(z),
                                             args.brightnessShift.GetHostElement(z),
                                             args.contrastCenter.GetHostElement(z));
    }

    // For F16 outputs the tolerance is a half-ULP count (see ComparePixel); one mul-add chain
    // with a single rounding on the half store justifies kUlps = 1 (HalfTestUtils.hpp policy).
    constexpr float absTolerance = std::is_integral_v<DstBT> || std::is_same_v<DstBT, __half> ? 1.f : 1e-5f;
    const bool      requireExact = std::is_integral_v<SrcBT> && std::is_same_v<SrcType, DstType> && argCounts.x == 0
                           && argCounts.y == 0 && argCounts.z == 0 && argCounts.w == 0;
    CompareTensors<DstType>(dstVec, refVec, dstStrides, shape, numPlanes, absTolerance, requireExact);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TYPED_TEST(OpBrightnessContrast, varshape_correct_output)
{
    const int3 shape = ttype::GetValue<TypeParam, 0>;
    using SrcType    = ttype::GetType<TypeParam, 1>;
    using DstType    = ttype::GetType<TypeParam, 2>;
    using SrcBT      = cuda::BaseType<SrcType>;
    using DstBT      = cuda::BaseType<DstType>;

    const nvcv::ImageFormat srcImgFormat{ttype::GetValue<TypeParam, 3>};
    const nvcv::ImageFormat dstImgFormat{ttype::GetValue<TypeParam, 4>};
    const int               numChannels = cuda::NumElements<SrcType>;
    static_assert(1 <= numChannels && numChannels <= 4);
    static_assert(numChannels == cuda::NumElements<DstType>);

    using ArgType        = ttype::GetType<TypeParam, 5>;
    const int4 argCounts = ttype::GetValue<TypeParam, 6>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image>          imgSrc;
    std::vector<nvcv::Image>          imgDst;
    std::vector<std::vector<uint8_t>> srcVec(shape.z);

    std::uniform_int_distribution randW(shape.x / 2, shape.x * 3 / 2);
    std::uniform_int_distribution randH(shape.y / 2, shape.y * 3 / 2);
    auto                          srcRand = MakeValueDistribution<SrcBT>();
    auto                          dstRand = MakeValueDistribution<DstBT>();
    std::mt19937_64               rng(12345);

    ASSERT_EQ(sizeof(SrcType), srcImgFormat.planePixelStrideBytes(0));
    ASSERT_EQ(sizeof(DstType), dstImgFormat.planePixelStrideBytes(0));

    for (int z = 0; z < shape.z; ++z)
    {
        nvcv::Size2D imgShape{randW(rng), randH(rng)};
        imgSrc.emplace_back(imgShape, srcImgFormat);
        imgDst.emplace_back(imgShape, dstImgFormat);

        auto srcImgData = imgSrc[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(srcImgData, nvcv::NullOpt);

        int numPlanes = srcImgFormat.numPlanes();
        if (numPlanes == 1)
        {
            ASSERT_EQ(srcImgFormat.numChannels(), numChannels);
        }
        else
        {
            ASSERT_EQ(srcImgFormat.numChannels(), numPlanes);
        }

        for (int p = 1; p < numPlanes; p++)
        {
            ASSERT_EQ(srcImgData->plane(0).rowStride, srcImgData->plane(p).rowStride);
        }

        int   srcRowStride = srcImgData->plane(0).rowStride;
        int   planeStride  = srcRowStride * imgSrc[z].size().h;
        long3 srcStrides{planeStride, srcRowStride, sizeof(SrcType)};

        srcVec[z].resize(srcStrides.x * numPlanes);

        const int numPixels = imgSrc[z].size().w * imgSrc[z].size().h;
        for (int p = 0; p < numPlanes; p++)
        {
            for (int idx = 0; idx < numPixels; ++idx)
            {
                FillRandomPixel<SrcType>(srcVec[z], srcStrides,
                                         int3{idx % imgSrc[z].size().w, idx / imgSrc[z].size().w, p}, srcRand, rng);
            }
        }

        for (int p = 0; p < numPlanes; p++)
        {
            ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(srcImgData->plane(p).basePtr, srcRowStride,
                                                     srcVec[z].data() + planeStride * p, srcRowStride, srcRowStride,
                                                     imgSrc[z].size().h, cudaMemcpyHostToDevice, stream));
        }
    }

    nvcv::ImageBatchVarShape batchSrc(shape.z);
    nvcv::ImageBatchVarShape batchDst(shape.z);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    for (auto argCount : {argCounts.x, argCounts.y, argCounts.z, argCounts.w})
    {
        ASSERT_TRUE(argCount == 0 || argCount == 1 || argCount == shape.z);
    }

    BrightnessContrastArguments<ArgType> args;
    args.template populate<SrcBT, DstBT>(rng, argCounts, srcRand, dstRand);

    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, batchSrc, batchDst, args.brightness.m_argTensor, args.contrast.m_argTensor,
                       args.brightnessShift.m_argTensor, args.contrastCenter.m_argTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int z = 0; z < shape.z; z++)
    {
        SCOPED_TRACE(z);

        const auto srcData = imgSrc[z].exportData<nvcv::ImageDataStridedCuda>();
        const auto dstData = imgDst[z].exportData<nvcv::ImageDataStridedCuda>();

        ASSERT_EQ(srcData->numPlanes(), dstData->numPlanes());

        int       srcRowStride = srcData->plane(0).rowStride;
        int       dstRowStride = dstData->plane(0).rowStride;
        long4_16a srcStrides{0, imgSrc[z].size().h * srcRowStride, srcRowStride, sizeof(SrcType)};
        long4_16a dstStrides{0, imgDst[z].size().h * dstRowStride, dstRowStride, sizeof(DstType)};

        int numPlanes = srcImgFormat.numPlanes();

        std::vector<uint8_t> dstVec(dstStrides.y * numPlanes);
        std::vector<uint8_t> refVec(dstStrides.y * numPlanes);

        int3 sampleShape{srcData->plane(0).width, srcData->plane(0).height, 1};
        for (int p = 1; p < numPlanes; p++)
        {
            ASSERT_EQ(srcData->plane(0).width, srcData->plane(p).width);
            ASSERT_EQ(srcData->plane(0).height, srcData->plane(p).height);
        }

        for (int p = 0; p < numPlanes; p++)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(dstVec.data() + dstStrides.y * p, dstStrides.z, dstData->plane(p).basePtr,
                                   dstStrides.z, dstStrides.z, imgDst[z].size().h, cudaMemcpyDeviceToHost));
        }

        BrightnessContrast<SrcType, DstType>(srcVec[z], refVec, srcStrides, dstStrides, sampleShape, numPlanes, 0,
                                             args.brightness.GetHostElement(z), args.contrast.GetHostElement(z),
                                             args.brightnessShift.GetHostElement(z),
                                             args.contrastCenter.GetHostElement(z));

        // F16 tolerance is a half-ULP count (kUlps = 1); see the tensor test above.
        constexpr float absTolerance = std::is_integral_v<DstBT> || std::is_same_v<DstBT, __half> ? 1.f : 1e-5f;
        const bool      requireExact = std::is_integral_v<SrcBT> && std::is_same_v<SrcType, DstType> && argCounts.x == 0
                               && argCounts.y == 0 && argCounts.z == 0 && argCounts.w == 0;
        CompareTensors<DstType>(dstVec, refVec, dstStrides, sampleShape, numPlanes, absTolerance, requireExact);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// BrightnessContrast is independent per channel, so a planar input must produce
// the same pixels as the equivalent interleaved input. These cases run identical
// data and per-sample arguments through both layouts and require the
// re-interleaved planar output to match the interleaved output bit-for-bit.
// =============================================================================

namespace {

nvcv::Tensor MakeF32ArgTensor(int numSamples, float first, float step)
{
    nvcv::Tensor arg({{numSamples}, "N"}, nvcv::TYPE_F32);
    auto         argData = arg.exportData<nvcv::TensorDataStridedCuda>();
    EXPECT_TRUE(argData);
    if (!argData)
    {
        return arg;
    }

    const long           stride = argData->stride(0);
    std::vector<uint8_t> host(stride * numSamples, uint8_t{0});
    for (int i = 0; i < numSamples; ++i)
    {
        const float value = first + step * static_cast<float>(i);
        std::memcpy(host.data() + static_cast<size_t>(stride) * static_cast<size_t>(i), &value, sizeof(value));
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(argData->basePtr(), host.data(), host.size(), cudaMemcpyHostToDevice));
    return arg;
}

struct PlanarBrightnessContrastArgs
{
    nvcv::Tensor brightness;
    nvcv::Tensor contrast;
    nvcv::Tensor brightnessShift;
    nvcv::Tensor contrastCenter;
};

PlanarBrightnessContrastArgs MakePlanarBrightnessContrastArgs(int numImages)
{
    return {
        MakeF32ArgTensor(numImages, 0.75f, 0.05f),
        MakeF32ArgTensor(numImages, 1.25f, -0.03f),
        MakeF32ArgTensor(numImages, 3.0f, 1.0f),
        MakeF32ArgTensor(numImages, 97.0f, 2.0f),
    };
}

void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width, int height,
                               int numImages)
{
    auto args = MakePlanarBrightnessContrastArgs(numImages);
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, width, height, width, height, numImages,
        [&args](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::BrightnessContrast op;
            EXPECT_NO_THROW(
                op(stream, src, dst, args.brightness, args.contrast, args.brightnessShift, args.contrastCenter));
        });
}

void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width, int height,
                                 int numImages)
{
    auto args = MakePlanarBrightnessContrastArgs(numImages);
    test::planar::RunVarShapeParity(planarFmt, interleavedFmt, width, height, width, height, numImages,
                                    [&args](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                            const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::BrightnessContrast op;
                                        EXPECT_NO_THROW(op(stream, src, dst, args.brightness, args.contrast,
                                                           args.brightnessShift, args.contrastCenter));
                                    });
}

} // namespace

TEST(OpBrightnessContrastScalar, tensor_planar_matches_interleaved)
{
    constexpr double brightness      = 0.75;
    constexpr double contrast        = 1.25;
    constexpr double brightnessShift = 0.125;
    constexpr double contrastCenter  = 0.5;

    test::planar::RunTensorParity(
        nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 13, 9, 13, 9, 2,
        [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::BrightnessContrast op;
            op(stream, src, dst, brightness, contrast, brightnessShift, contrastCenter);
        });
}

TEST(OpBrightnessContrastScalar, varshape_planar_matches_interleaved)
{
    constexpr double brightness      = 0.75;
    constexpr double contrast        = 1.25;
    constexpr double brightnessShift = 0.125;
    constexpr double contrastCenter  = 0.5;

    test::planar::RunVarShapeParity(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 13, 9, 13, 9, 2,
                                    [](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                       const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::BrightnessContrast op;
                                        op(stream, src, dst, brightness, contrast, brightnessShift, contrastCenter);
                                    });
}

TEST(OpBrightnessContrastCoverage, varshape_scalar_u8_batched_vector_path)
{
    // The typed VarShape cases use per-image argument tensors; this scalar U8 case instantiates
    // the batched vector kernel with the runtime clamp flag.
    test::planar::RunVarShapeParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 17, 5, 17, 5, 2,
                                    [](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                       const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::BrightnessContrast op;
                                        EXPECT_NO_THROW(op(stream, src, dst, 1.0, 2.0, 0.0, 128.0, false));
                                    });
}

TEST(OpBrightnessContrastScalar, clamp_float_output)
{
    const std::vector<float> srcVec{0.f, 0.25f, 0.75f, 1.f};
    const std::vector<float> expectedUnclamped{-0.25f, 0.25f, 1.25f, 1.75f};
    const std::vector<float> expectedClamped{0.f, 0.25f, 1.f, 1.f};

    nvcv::Tensor src           = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_F32);
    nvcv::Tensor unclampedDst  = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_F32);
    nvcv::Tensor clampedDst    = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_F32);
    auto         srcData       = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         unclampedData = unclampedDst.exportData<nvcv::TensorDataStridedCuda>();
    auto         clampedData   = clampedDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && unclampedData && clampedData);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(srcData->basePtr(), srcVec.data(), srcVec.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, src, unclampedDst, 2., 1., -0.25, 0., false));
    ASSERT_NO_THROW(op(stream, src, clampedDst, 2., 1., -0.25, 0., true));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<float> unclamped(srcVec.size());
    std::vector<float> clamped(srcVec.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(unclamped.data(), unclampedData->basePtr(), unclamped.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(clamped.data(), clampedData->basePtr(), clamped.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
    EXPECT_EQ(expectedUnclamped, unclamped);
    EXPECT_EQ(expectedClamped, clamped);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename Visit>
static void ForEachBrightnessContrastPixel(int samples, int channels, int height, int width, Visit visit)
{
    for (int n = 0; n < samples; ++n)
        for (int c = 0; c < channels; ++c)
            for (int pixel = 0; pixel < height * width; ++pixel) visit(n, c, pixel / width, pixel % width);
}

template<typename Offset>
static void CheckBrightnessContrastS64Rounding(const std::vector<uint8_t> &input, const std::vector<uint8_t> &output,
                                               int channels, int width, Offset offset, bool clamp, double brightness,
                                               double shift)
{
    ForEachBrightnessContrastPixel(2, channels, 3, width,
                                   [&input, &output, offset, clamp, brightness, shift](int n, int c, int y, int x)
                                   {
                                       const size_t off = offset(n, c, y, x);
                                       int64_t      value;
                                       int64_t      actual;
                                       std::memcpy(&value, input.data() + off, sizeof(value));
                                       std::memcpy(&actual, output.data() + off, sizeof(actual));
                                       double expected = std::fma(brightness, static_cast<double>(value), shift);
                                       if (clamp && expected < 0)
                                           expected = 0;
                                       EXPECT_EQ(RoundSaturatedS64<int64_t>(expected), actual);
                                   });
}

static void RunBrightnessContrastS64Rounding(const std::vector<int64_t> &values, int channels, bool planar)
{
    const int width = static_cast<int>(values.size()) | 1;
    const nvcv::TensorShape shape = planar ? nvcv::TensorShape{{2, channels, 3, width}, "NCHW"}
                                          : nvcv::TensorShape{{2, 3, width, channels}, "NHWC"};
    nvcv::Tensor src(shape, nvcv::TYPE_S64);
    nvcv::Tensor dst(shape, nvcv::TYPE_S64);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto         access  = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(access);
    const size_t         bytes = access->sampleStride() * 2;
    std::vector<uint8_t> input(bytes, 0);
    std::vector<uint8_t> output(bytes);
    const auto           offset = [&access, planar, channels](int n, int c, int y, int x)
    {
        return n * access->sampleStride() + y * access->rowStride()
             + (planar ? c * access->planeStride() + x * sizeof(int64_t) : (x * channels + c) * sizeof(int64_t));
    };
    ForEachBrightnessContrastPixel(2, channels, 3, width,
                                   [&values, &input, offset, width](int n, int c, int y, int x)
                                   {
                                       const auto v = values[(x + width * y + c + n) % values.size()];
                                       std::memcpy(input.data() + offset(n, c, y, x), &v, sizeof(v));
                                   });
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice));
    cvcuda::BrightnessContrast op;
    const auto cases = test::Combine(test::ValueList<bool>{false, true}, test::ValueList<double>{0.5, 1., 2.},
                                     test::ValueList<double>{-0.5, 0., 0.5});
    for (const auto &[clamp, brightness, shift] : cases)
    {
        SCOPED_TRACE(testing::Message() << channels << ":" << planar << ":" << clamp << ":" << brightness << ":"
                                        << shift);
        op(nullptr, src, dst, brightness, 1., shift, 0., clamp);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), dstData->basePtr(), bytes, cudaMemcpyDeviceToHost));
        CheckBrightnessContrastS64Rounding(input, output, channels, width, offset, clamp, brightness, shift);
    }
}

TEST(OpBrightnessContrastScalar, s64_rounding_and_saturation_exact)
{
    constexpr int64_t    lo = std::numeric_limits<int64_t>::min();
    constexpr int64_t    hi = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> values{lo, lo + 1024, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, hi - 1023, hi};
    for (int exponent = 1; exponent < 63; ++exponent)
        for (int delta : {-2, -1, 0, 1, 2})
        {
            const int64_t value = (int64_t{1} << exponent) + delta;
            values.push_back(value);
            values.push_back(-value);
        }
    for (int channels : {1, 3, 4})
        for (bool planar : {false, true})
            ASSERT_NO_FATAL_FAILURE(RunBrightnessContrastS64Rounding(values, channels, planar));
}

enum class BrightnessContrastValuePattern
{
    TwoKeys,
    Unique,
    MisleadingProbe,
    MixedWarps,
    MixedWarps128,
    FloatingEdges
};

template<typename ArgT>
static std::array<nvcv::Tensor, 4> MakeBrightnessContrastArgumentTensors(int samples)
{
    return {GetArgTensor<ArgT>(samples), GetArgTensor<ArgT>(samples), GetArgTensor<ArgT>(samples),
            GetArgTensor<ArgT>(samples)};
}

template<typename Batch>
static void SubmitBrightnessContrast(cvcuda::BrightnessContrast &op, const Batch &src, const Batch &dst,
                                     const std::array<nvcv::Tensor, 4> &args, const std::array<double, 4> &scalars,
                                     bool tensorArgs, bool clamp)
{
    if (tensorArgs)
        ASSERT_NO_THROW(op(nullptr, src, dst, args[0], args[1], args[2], args[3]));
    else
        ASSERT_NO_THROW(op(nullptr, src, dst, scalars[0], scalars[1], scalars[2], scalars[3], clamp));
}

struct BrightnessContrastObservedLayout
{
    static constexpr int samples = 3;
    int                  width;
    int                  height;
    int                  channels;
    bool                 planar;
    bool                 padded;
    int64_t              row;
    int64_t              plane;
    int64_t              sample;

    size_t offset(int n, int c, int y, int x) const
    {
        return n * sample + y * row + (planar ? c * plane + x * sizeof(int64_t) : (x * channels + c) * sizeof(int64_t));
    }

    nvcv::Tensor wrap(NVCVByte *ptr, nvcv::DataType dtype) const
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr;
        buffer.strides[0] = sample;
        buffer.strides[1] = planar ? plane : row;
        buffer.strides[2] = planar ? row : channels * sizeof(int64_t);
        buffer.strides[3] = sizeof(int64_t);
        const nvcv::TensorShape shape = planar ? nvcv::TensorShape{{samples, channels, height, width}, "NCHW"}
                                              : nvcv::TensorShape{{samples, height, width, channels}, "NHWC"};
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, dtype, buffer});
    }
};

template<typename T>
static auto BrightnessContrastObservedKeys()
{
    if constexpr (std::is_same_v<T, double>)
        return std::array<std::array<double, 2>, 3>{
            {{-0.37123456789, 0.7137142857143},
             {-13.25, 0x1.0000000000001p-53},
             {-0x1.123456789abcdp500, 0x1.7654321fedcbap500}}
        };
    else
    {
        constexpr int64_t lo = std::numeric_limits<int64_t>::min();
        constexpr int64_t hi = std::numeric_limits<int64_t>::max();
        return std::array<std::array<int64_t, 2>, 3>{
            {{-7, 13}, {-(int64_t{1} << 55) - 3, (int64_t{1} << 56) + 5}, {lo + 1024, hi - 1023}}
        };
    }
}

template<typename T>
static T BrightnessContrastFromBits(uint64_t bits)
{
    static_assert(sizeof(T) == sizeof(bits));
#if defined(__cpp_lib_bit_cast)
    return std::bit_cast<T>(bits);
#else
    // GCC 10's standard library does not provide C++20 bit_cast.
    T value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

static double BrightnessContrastFloatingEdge(int segmentIndex)
{
    const std::array<std::array<uint64_t, 2>, 8> edges{
        {{0, 0x8000000000000000ULL},
         {0x7ff0000000000000ULL, 0xfff0000000000000ULL},
         {1, 0x8000000000000001ULL},
         {0x000fffffffffffffULL, 0x0010000000000000ULL},
         {0x7ff8000000001234ULL, 0xfff8000000005678ULL},
         {0x7ff0000000000001ULL, 0xfff0000000000001ULL},
         {0x7fefffffffffffffULL, 0xffefffffffffffffULL},
         {0x3fdfffffffffffffULL, 0x3fe0000000000001ULL}}
    };
    const int blockIndex = segmentIndex % 1024;
    const int pair       = (segmentIndex / 1024 + (blockIndex / 2 % 256) / 32) % 8;
    return BrightnessContrastFromBits<double>(edges[pair][segmentIndex & 1]);
}

template<typename T>
static T BrightnessContrastObservedValue(BrightnessContrastValuePattern pattern, int segmentIndex, int n, uint64_t raw,
                                         const std::array<T, 2> &keys)
{
    const int  blockIndex = segmentIndex % 1024;
    const bool hit        = pattern == BrightnessContrastValuePattern::TwoKeys
                  || (pattern == BrightnessContrastValuePattern::MisleadingProbe && blockIndex < 32)
                  || (pattern == BrightnessContrastValuePattern::MixedWarps && (blockIndex / 2 % 256) / 32 < 3)
                  || (pattern == BrightnessContrastValuePattern::MixedWarps128 && segmentIndex % 256 < 64 * (n + 2));
    auto value = BrightnessContrastFromBits<T>(raw);
    if (hit)
        value = keys[(raw >> 32) & 1];
    if (pattern == BrightnessContrastValuePattern::MixedWarps128)
    {
        // Both groups have exactly two, three, or four matching warps in successive samples.
        value = hit ? keys[segmentIndex & 1] : static_cast<T>(101 + n);
    }
    if constexpr (std::is_same_v<T, double>)
    {
        // Distinct fixed probe lanes make repeated-key warps exercise both cached results.
        if (hit && (blockIndex % 64 == 0 || blockIndex % 64 == 63))
            value = keys[blockIndex % 64 == 63];
        if (pattern == BrightnessContrastValuePattern::FloatingEdges)
            value = BrightnessContrastFloatingEdge(segmentIndex);
    }
    return value;
}

template<typename T>
static void FillBrightnessContrastObservedValues(std::vector<uint8_t>                   &input,
                                                 const BrightnessContrastObservedLayout &layout,
                                                 BrightnessContrastValuePattern          pattern)
{
    const auto keys = BrightnessContrastObservedKeys<T>();
    ForEachBrightnessContrastPixel(
        BrightnessContrastObservedLayout::samples, layout.channels, layout.height, layout.width,
        [&input, &layout, pattern, &keys](int n, int c, int y, int x)
        {
            const int      linear       = layout.planar ? (c * layout.height + y) * layout.width + x
                                                        : (y * layout.width + x) * layout.channels + c;
            const int      rowIndex     = layout.planar ? x : x * layout.channels + c;
            const int      segmentIndex = layout.padded ? rowIndex : linear;
            const uint64_t raw
                = 0x9e3779b97f4a7c15ULL * (1ULL + linear + n * layout.width * layout.height * layout.channels);
            const T value = BrightnessContrastObservedValue(pattern, segmentIndex, n, raw, keys[n]);
            std::memcpy(input.data() + layout.offset(n, c, y, x), &value, sizeof(value));
        });
}

template<typename T>
static std::array<std::array<double, 4>, 3> BrightnessContrastObservedArguments()
{
    if constexpr (std::is_same_v<T, double>)
    {
        // Negative zero and subnormals survive these sample-specific maps.
        return {
            {{-1., 1., -0., 0.}, {1., 1., -0., -0.}, {0.75, -1.125, 0.125, 0.5}}
        };
    }
    else
        return {
            {{0.75, 1.125, -0.5, 1024.5}, {1.25, -0.625, 0.5, -0.75}, {-1.5, 0.875, 13.25, 0x1p62}}
        };
}

static void UploadBrightnessContrastObservedArguments(nvcv::Tensor &tensor, int column,
                                                      const std::array<std::array<double, 4>, 3> &args)
{
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(data);
    std::vector<uint8_t> values(data->stride(0) * args.size(), 0);
    for (size_t n = 0; n < args.size(); ++n)
        std::memcpy(values.data() + n * data->stride(0), &args[n][column], sizeof(double));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(data->basePtr(), values.data(), values.size(), cudaMemcpyHostToDevice));
}

template<typename T>
static void MakeBrightnessContrastObservedGold(const std::vector<uint8_t> &input, std::vector<uint8_t> &expected,
                                               const BrightnessContrastObservedLayout     &layout,
                                               const std::array<std::array<double, 4>, 3> &args,
                                               const std::array<double, 4> &scalarArgs, int mode)
{
    const bool clamp = mode == 2;
    ForEachBrightnessContrastPixel(
        BrightnessContrastObservedLayout::samples, layout.channels, layout.height, layout.width,
        [&input, &expected, &layout, &args, &scalarArgs, mode, clamp](int n, int c, int y, int x)
        {
            const auto off = layout.offset(n, c, y, x);
            T          value;
            std::memcpy(&value, input.data() + off, sizeof(value));
            const auto  &a          = mode == 0 ? args[n] : scalarArgs;
            const double difference = static_cast<double>(value) - a[3];
            double       gold       = std::fma(a[0], std::fma(a[1], difference, a[3]), a[2]);
            if (clamp && gold < 0.)
                gold = 0.;
            T rounded;
            if constexpr (std::is_same_v<T, double>)
                rounded = clamp && gold > 1. ? 1. : gold;
            else
                rounded = RoundSaturatedS64<int64_t>(gold);
            std::memcpy(expected.data() + off, &rounded, sizeof(rounded));
        });
}

static void NormalizeBrightnessContrastNaNPayloads(const std::vector<uint8_t> &expected, std::vector<uint8_t> &actual,
                                                   const BrightnessContrastObservedLayout &layout)
{
    ForEachBrightnessContrastPixel(BrightnessContrastObservedLayout::samples, layout.channels, layout.height,
                                   layout.width,
                                   [&expected, &actual, &layout](int n, int c, int y, int x)
                                   {
                                       const auto off = layout.offset(n, c, y, x);
                                       double     gold;
                                       double     result;
                                       std::memcpy(&gold, expected.data() + off, sizeof(gold));
                                       std::memcpy(&result, actual.data() + off, sizeof(result));
                                       // NaN payloads are unspecified; every other result and padding byte stays exact.
                                       if (std::isnan(gold) && std::isnan(result))
                                           std::memcpy(actual.data() + off, &gold, sizeof(gold));
                                   });
}

template<typename T>
static void CheckBrightnessContrastObservedResult(cvcuda::BrightnessContrast &op, const nvcv::Tensor &src,
                                                  const nvcv::Tensor &dst, const std::array<nvcv::Tensor, 4> &args,
                                                  const std::array<double, 4>            &scalarArgs,
                                                  const BrightnessContrastObservedLayout &layout,
                                                  const std::vector<uint8_t>             &input,
                                                  const std::vector<uint8_t> &expected, int mode, bool inPlace)
{
    SCOPED_TRACE(testing::Message() << mode << ":" << inPlace);
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    const auto bytes = input.size();
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstData->basePtr(), 0xA5, bytes));
    const auto &target = inPlace ? src : dst;
    ASSERT_NO_FATAL_FAILURE(SubmitBrightnessContrast(op, src, target, args, scalarArgs, mode == 0, mode == 2));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    std::vector<uint8_t> actual(bytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), inPlace ? srcData->basePtr() : dstData->basePtr(), bytes,
                                      cudaMemcpyDeviceToHost));
    if constexpr (std::is_same_v<T, double>)
        NormalizeBrightnessContrastNaNPayloads(expected, actual, layout);
    EXPECT_EQ(expected, actual);
}

template<typename T = int64_t>
static void RunBrightnessContrastObservedValues(int width, int channels, bool planar, bool padded,
                                                BrightnessContrastValuePattern pattern, int height = 3)
{
    static_assert(sizeof(T) == sizeof(int64_t));
    const int64_t                          rowBytes = width * (planar ? 1 : channels) * sizeof(T);
    const int64_t                          row      = padded ? ((rowBytes + 15) / 16 + 2) * 16 : rowBytes;
    const int64_t                          plane    = row * height;
    const int64_t                          sample   = (plane * (planar ? channels : 1) + 15) / 16 * 16;
    const BrightnessContrastObservedLayout layout{width, height, channels, planar, padded, row, plane, sample};
    const size_t                           bytes = sample * BrightnessContrastObservedLayout::samples;
    SCOPED_TRACE(testing::Message() << width << ":" << height << ":" << channels << ":" << planar << ":" << padded
                                    << ":" << static_cast<int>(pattern));
    NVCVByte *srcPtr{};
    NVCVByte *dstPtr{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, bytes));
    const auto           dtype = std::is_same_v<T, double> ? nvcv::TYPE_F64 : nvcv::TYPE_S64;
    auto                 src   = layout.wrap(srcPtr, dtype);
    auto                 dst   = layout.wrap(dstPtr, dtype);
    std::vector<uint8_t> input(bytes, 0xA5);
    std::vector<uint8_t> expected(bytes, 0xA5);
    FillBrightnessContrastObservedValues<T>(input, layout, pattern);
    const auto                  args = BrightnessContrastObservedArguments<T>();
    const std::array<double, 4> scalarArgs{1.75, 1.25, -0.5, std::is_same_v<T, double> ? 0.25 : 1024.5};
    auto tensorArgs = MakeBrightnessContrastArgumentTensors<double>(BrightnessContrastObservedLayout::samples);
    for (int column = 0; column < 4; ++column)
        ASSERT_NO_FATAL_FAILURE(UploadBrightnessContrastObservedArguments(tensorArgs[column], column, args));
    cvcuda::BrightnessContrast op;
    for (int mode : {0, 1, 2})
    {
        MakeBrightnessContrastObservedGold<T>(input, expected, layout, args, scalarArgs, mode);
        for (bool inPlace : {false, true})
            ASSERT_NO_FATAL_FAILURE(CheckBrightnessContrastObservedResult<T>(op, src, dst, tensorArgs, scalarArgs,
                                                                             layout, input, expected, mode, inPlace));
    }
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

TEST(OpBrightnessContrast, s64_observed_values_layouts_and_tails)
{
    for (int channels : {1, 3, 4})
        for (bool planar : {false, true})
            for (bool padded : {false, true})
                RunBrightnessContrastObservedValues(513, channels, planar, padded,
                                                    BrightnessContrastValuePattern::TwoKeys);
}

TEST(OpBrightnessContrast, s64_observed_values_mixed_hits_and_misses)
{
    for (int width : {127, 129, 191, 192, 193, 1025})
        for (bool planar : {false, true})
            for (auto pattern :
                 {BrightnessContrastValuePattern::TwoKeys, BrightnessContrastValuePattern::Unique,
                  BrightnessContrastValuePattern::MisleadingProbe, BrightnessContrastValuePattern::MixedWarps})
                RunBrightnessContrastObservedValues(width, 1, planar, true, pattern);
}

TEST(OpBrightnessContrast, s64_observed_values_stream_size_boundary)
{
    for (int width : {1048575, 1048576, 1048577})
        RunBrightnessContrastObservedValues(width, 1, false, false, BrightnessContrastValuePattern::TwoKeys, 1);
}

TEST(OpBrightnessContrast, s64_observed_values_large_streams)
{
    RunBrightnessContrastObservedValues(1048577, 1, true, false, BrightnessContrastValuePattern::Unique, 1);
    RunBrightnessContrastObservedValues(349527, 3, false, false, BrightnessContrastValuePattern::MixedWarps, 1);
    RunBrightnessContrastObservedValues(349527, 3, true, false, BrightnessContrastValuePattern::TwoKeys, 1);
    RunBrightnessContrastObservedValues(262145, 4, false, false, BrightnessContrastValuePattern::TwoKeys, 1);
    RunBrightnessContrastObservedValues(262145, 4, true, false, BrightnessContrastValuePattern::MisleadingProbe, 1);
}

TEST(OpBrightnessContrast, s64_observed_values_128_thread_admission_and_tails)
{
    constexpr int largeSegment = 1 << 20;
    for (int tail : {127, 128, 129, 511, 512, 513})
        RunBrightnessContrastObservedValues(largeSegment + tail, 1, false, false,
                                            BrightnessContrastValuePattern::MixedWarps128, 1);
    RunBrightnessContrastObservedValues(largeSegment + 129, 1, true, true,
                                        BrightnessContrastValuePattern::MixedWarps128, 1);
}

TEST(OpBrightnessContrast, s64_observed_values_full_tile_regions)
{
    RunBrightnessContrastObservedValues((1 << 20) + 512, 1, true, true, BrightnessContrastValuePattern::MixedWarps128,
                                        2);
    RunBrightnessContrastObservedValues((1 << 20) + 128, 1, true, true, BrightnessContrastValuePattern::MixedWarps128,
                                        4);
}

TEST(OpBrightnessContrast, f64_observed_values_layouts_and_tails)
{
    for (int channels : {1, 3, 4})
        for (bool planar : {false, true})
            for (bool padded : {false, true})
                RunBrightnessContrastObservedValues<double>(513, channels, planar, padded,
                                                            BrightnessContrastValuePattern::TwoKeys);
}

TEST(OpBrightnessContrast, f64_observed_values_stream_size_boundary)
{
    for (int width : {1048575, 1048576, 1048577})
        RunBrightnessContrastObservedValues<double>(width, 1, false, false, BrightnessContrastValuePattern::TwoKeys, 1);
}

TEST(OpBrightnessContrast, f64_observed_values_large_streams)
{
    RunBrightnessContrastObservedValues<double>(1048577, 1, true, false, BrightnessContrastValuePattern::Unique, 1);
    RunBrightnessContrastObservedValues<double>(349527, 3, false, true, BrightnessContrastValuePattern::MixedWarps, 1);
    RunBrightnessContrastObservedValues<double>(349527, 3, true, false, BrightnessContrastValuePattern::TwoKeys, 1);
    RunBrightnessContrastObservedValues<double>(262145, 4, false, false, BrightnessContrastValuePattern::TwoKeys, 1);
    RunBrightnessContrastObservedValues<double>(262145, 4, true, false, BrightnessContrastValuePattern::MisleadingProbe,
                                                1);
}

TEST(OpBrightnessContrast, f64_observed_values_ieee_edges)
{
    RunBrightnessContrastObservedValues<double>(1048577, 1, false, true, BrightnessContrastValuePattern::FloatingEdges,
                                                1);
    RunBrightnessContrastObservedValues<double>(349527, 3, true, false, BrightnessContrastValuePattern::FloatingEdges,
                                                1);
}

template<typename ArgT>
static void UploadBrightnessContrastScalarArguments(std::array<nvcv::Tensor, 4> &args)
{
    const std::array<ArgT, 4> values{2, 1, 1, 0};
    for (int column = 0; column < 4; ++column)
    {
        auto data = args[column].exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy(data->basePtr(), &values[column], sizeof(ArgT), cudaMemcpyHostToDevice));
    }
}

template<typename T, int channels>
static void RunBrightnessContrastScalarAlignedWideCase(nvcv::DataType dtype, int srcOffset, int dstOffset)
{
    constexpr int     width   = 33;
    constexpr int     height  = 3;
    constexpr int     samples = 2;
    constexpr int64_t row     = width * channels * sizeof(T) + 8;
    SCOPED_TRACE(testing::Message() << srcOffset << ":" << dstOffset);
    const size_t srcBytes = srcOffset + samples * height * row;
    const size_t dstBytes = dstOffset + samples * height * row;
    NVCVByte    *srcPtr{};
    NVCVByte    *dstPtr{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, srcBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, dstBytes));
    auto wrap = [=](NVCVByte *ptr, int offset)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr + offset;
        buffer.strides[0] = height * row;
        buffer.strides[1] = row;
        buffer.strides[2] = channels * sizeof(T);
        buffer.strides[3] = sizeof(T);
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{samples, height, width, channels}, "NHWC"},
            dtype, buffer
        });
    };
    auto                 src = wrap(srcPtr, srcOffset);
    auto                 dst = wrap(dstPtr, dstOffset);
    std::vector<uint8_t> input(srcBytes, 0xA5);
    std::vector<uint8_t> expected(dstBytes, 0xA5);
    std::vector<uint8_t> actual(dstBytes, 0xA5);
    for (int n = 0; n < samples; ++n)
        for (int y = 0; y < height; ++y)
            for (int i = 0; i < width * channels; ++i)
            {
                const auto   value = static_cast<T>(i - 50 + n * 200 + y * 5);
                const auto   gold  = static_cast<T>(2 * value + 1);
                const size_t off   = (n * height + y) * row + i * sizeof(T);
                std::memcpy(input.data() + srcOffset + off, &value, sizeof(value));
                std::memcpy(expected.data() + dstOffset + off, &gold, sizeof(gold));
            }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), srcBytes, cudaMemcpyHostToDevice));
    auto args = MakeBrightnessContrastArgumentTensors<double>(1);
    ASSERT_NO_FATAL_FAILURE(UploadBrightnessContrastScalarArguments<double>(args));
    cvcuda::BrightnessContrast op;
    for (bool tensorArgs : {false, true})
    {
        ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, dstBytes));
        ASSERT_NO_FATAL_FAILURE(SubmitBrightnessContrast(op, src, dst, args, {2., 1., 1., 0.}, tensorArgs, false));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstPtr, dstBytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected, actual);
    }
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

template<typename T, int channels>
static void RunBrightnessContrastScalarAlignedWide(nvcv::DataType dtype)
{
    for (int srcOffset : {0, 8})
        for (int dstOffset : {0, 8})
            ASSERT_NO_FATAL_FAILURE(
                (RunBrightnessContrastScalarAlignedWideCase<T, channels>(dtype, srcOffset, dstOffset)));
}

template<typename T>
static void RunBrightnessContrastRGBAAlignmentCase(nvcv::ImageFormat format, const std::array<nvcv::Tensor, 4> &args,
                                                   int width, int padding, int srcOffset, int dstOffset)
{
    constexpr int height = 3;
    SCOPED_TRACE(testing::Message() << padding << ":" << srcOffset << ":" << dstOffset);
    const int    row   = (width * 4 + padding) * sizeof(T);
    const size_t bytes = height * row + sizeof(T);
    NVCVByte    *srcPtr{};
    NVCVByte    *dstPtr{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, bytes));
    const auto wrap = [=](NVCVByte *ptr, int offset)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes = 1;
        buffer.planes[0] = {width, height, row, ptr + offset * sizeof(T)};
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, buffer});
    };
    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);
    src.pushBack(wrap(srcPtr, srcOffset));
    dst.pushBack(wrap(dstPtr, dstOffset));
    std::vector<uint8_t> input(bytes, 0xA5);
    std::vector<uint8_t> expected(bytes, 0xA5);
    std::vector<uint8_t> actual(bytes);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width * 4; ++x)
        {
            const auto value = static_cast<T>((x + y) % 128);
            const auto gold  = static_cast<T>(2 * ((x + y) % 128) + 1);
            std::memcpy(input.data() + srcOffset * sizeof(T) + y * row + x * sizeof(T), &value, sizeof(T));
            std::memcpy(expected.data() + dstOffset * sizeof(T) + y * row + x * sizeof(T), &gold, sizeof(T));
        }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), bytes, cudaMemcpyHostToDevice));
    cvcuda::BrightnessContrast op;
    for (bool tensorArgs : {false, true})
    {
        ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, bytes));
        ASSERT_NO_FATAL_FAILURE(SubmitBrightnessContrast(op, src, dst, args, {2., 1., 1., 0.}, tensorArgs, false));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstPtr, bytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected, actual);
    }
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

template<typename T>
static void RunBrightnessContrastRGBAAlignmentOffsets(nvcv::ImageFormat format, const std::array<nvcv::Tensor, 4> &args,
                                                      int width, int padding)
{
    for (int srcOffset : {0, 1})
        for (int dstOffset : {0, 1})
            ASSERT_NO_FATAL_FAILURE(
                RunBrightnessContrastRGBAAlignmentCase<T>(format, args, width, padding, srcOffset, dstOffset));
}

template<typename T>
static void RunBrightnessContrastRGBAAlignment(nvcv::ImageFormat format)
{
    auto args = MakeBrightnessContrastArgumentTensors<float>(1);
    ASSERT_NO_FATAL_FAILURE(UploadBrightnessContrastScalarArguments<float>(args));
    for (int width : {33, 131})
        for (int padding : {0, 1})
            ASSERT_NO_FATAL_FAILURE(RunBrightnessContrastRGBAAlignmentOffsets<T>(format, args, width, padding));
}

TEST(OpBrightnessContrast, rgba8_varshape_scalar_alignment)
{
    RunBrightnessContrastRGBAAlignment<uint8_t>(nvcv::FMT_RGBA8);
}

TEST(OpBrightnessContrast, rgbaf16_varshape_scalar_alignment)
{
    RunBrightnessContrastRGBAAlignment<__half>(nvcv::FMT_RGBAf16);
}

TEST(OpBrightnessContrast, s64_wide_scalar_alignment)
{
    RunBrightnessContrastScalarAlignedWide<int64_t, 4>(nvcv::TYPE_S64);
}

TEST(OpBrightnessContrast, f64_wide_scalar_alignment)
{
    RunBrightnessContrastScalarAlignedWide<double, 4>(nvcv::TYPE_F64);
}

TEST(OpBrightnessContrast, s64_paired_scalar_alignment)
{
    RunBrightnessContrastScalarAlignedWide<int64_t, 2>(nvcv::TYPE_S64);
}

TEST(OpBrightnessContrast, f64_paired_scalar_alignment)
{
    RunBrightnessContrastScalarAlignedWide<double, 2>(nvcv::TYPE_F64);
}

TEST(OpBrightnessContrastCoverage, tensor_padded_rows_reach_scalar_and_vector_kernels)
{
    constexpr int height = 5;

    // Odd-byte row pitches reject the element-stream alignment check while still leaving row 0
    // aligned in the scalar U8 case, so the vector kernel sees both its packed group and its tail.
    auto runCase = []<typename ValueType>(ValueType, int width, int rowStride)
    {
        constexpr int channels     = cuda::NumElements<ValueType>;
        const int     sampleStride = rowStride * height;
        const auto    bufferSize   = static_cast<size_t>(sampleStride);

        NVCVByte *srcAllocation{};
        NVCVByte *dstAllocation{};
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bufferSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), bufferSize));

        nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.basePtr    = srcAllocation;
        srcBuffer.strides[0] = sampleStride;
        srcBuffer.strides[1] = rowStride;
        srcBuffer.strides[2] = sizeof(ValueType);
        srcBuffer.strides[3] = sizeof(uchar);
        auto dstBuffer       = srcBuffer;
        dstBuffer.basePtr    = dstAllocation;

        const nvcv::TensorShape shape{
            {1, height, width, channels},
            "NHWC"
        };
        nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, srcBuffer});
        nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, dstBuffer});

        std::vector<uint8_t> srcHost(bufferSize, uint8_t{0xA5});
        std::vector<uint8_t> refHost(bufferSize, uint8_t{0xCD});
        std::vector<uint8_t> dstHost(bufferSize, uint8_t{0xCD});
        const long4_16a      strides{sampleStride, 0, rowStride, sizeof(ValueType)};
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                auto &pixel = test::ValueAt<ValueType>(srcHost, strides, int4{x, y, 0, 0});
                for (int c = 0; c < channels; ++c)
                {
                    cuda::GetElement(pixel, c) = static_cast<uchar>((17 * y + 13 * x + 29 * c + 5) % 251);
                }
            }
        }

        BrightnessContrast<ValueType, ValueType>(srcHost, refHost, strides, strides, int3{width, height, 1}, 1, 0, 1.f,
                                                 2.f, 0.f, 128.f);

        ASSERT_EQ(cudaSuccess, cudaMemcpy(srcAllocation, srcHost.data(), bufferSize, cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dstAllocation, dstHost.data(), bufferSize, cudaMemcpyHostToDevice));

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
        cvcuda::BrightnessContrast op;
        ASSERT_NO_THROW(op(stream, src, dst, 1.0, 2.0, 0.0, 128.0, true));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dstHost.data(), dstAllocation, bufferSize, cudaMemcpyDeviceToHost));

        EXPECT_EQ(refHost, dstHost);

        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
        ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
        ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
    };

    // On SM89, scalar U8 selects BrightnessContrastVec while RGB8 selects the scalar tensor kernel.
    runCase(uchar{}, 17, 18);
    runCase(uchar3{}, 5, 17);
}

TEST(OpBrightnessContrastCoverage, tensor_planar_noncontiguous_planes_reach_fallback)
{
    constexpr int    channels     = 3;
    constexpr int    width        = 17;
    constexpr int    height       = 5;
    constexpr int    rowStride    = width + 1;
    constexpr int    planeStride  = rowStride * height + 1;
    constexpr int    sampleStride = planeStride * channels;
    constexpr size_t bufferSize   = sampleStride;

    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bufferSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), bufferSize));

    auto wrapTensor = [&](NVCVByte *base)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = base;
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = planeStride;
        buffer.strides[2] = rowStride;
        buffer.strides[3] = sizeof(uchar);
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, channels, height, width}, "NCHW"},
            nvcv::TYPE_U8, buffer
        });
    };

    nvcv::Tensor src = wrapTensor(srcAllocation);
    nvcv::Tensor dst = wrapTensor(dstAllocation);

    std::vector<uint8_t> srcHost(bufferSize, uint8_t{0xA5});
    std::vector<uint8_t> refHost(bufferSize, uint8_t{0xCD});
    std::vector<uint8_t> dstHost(bufferSize, uint8_t{0xCD});
    const long4_16a      strides{sampleStride, planeStride, rowStride, sizeof(uchar)};
    for (int p = 0; p < channels; ++p)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                auto &pixel = test::ValueAt<uchar>(srcHost, strides, int4{x, y, p, 0});
                pixel       = static_cast<uchar>((17 * y + 13 * x + 29 * p + 5) % 251);
            }
        }
    }

    BrightnessContrast<uchar, uchar>(srcHost, refHost, strides, strides, int3{width, height, 1}, channels, 0, 1.f, 2.f,
                                     0.f, 128.f);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcAllocation, srcHost.data(), bufferSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstAllocation, dstHost.data(), bufferSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, src, dst, 1.0, 2.0, 0.0, 128.0, true));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstHost.data(), dstAllocation, bufferSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(refHost, dstHost);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpBrightnessContrastCoverage, tensor_planar_padded_rows_reach_batched_fallback)
{
    constexpr int    channels     = 3;
    constexpr int    width        = 17;
    constexpr int    height       = 5;
    constexpr int    rowStride    = width + 1;
    constexpr int    planeStride  = rowStride * height;
    constexpr int    sampleStride = planeStride * channels;
    constexpr size_t bufferSize   = sampleStride;

    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bufferSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), bufferSize));

    auto wrapTensor = [&](NVCVByte *base)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = base;
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = planeStride;
        buffer.strides[2] = rowStride;
        buffer.strides[3] = sizeof(uchar);
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, channels, height, width}, "NCHW"},
            nvcv::TYPE_U8, buffer
        });
    };

    nvcv::Tensor src = wrapTensor(srcAllocation);
    nvcv::Tensor dst = wrapTensor(dstAllocation);

    std::vector<uint8_t> srcHost(bufferSize, uint8_t{0xA5});
    std::vector<uint8_t> refHost(bufferSize, uint8_t{0xCD});
    std::vector<uint8_t> dstHost(bufferSize, uint8_t{0xCD});
    const long4_16a      strides{sampleStride, planeStride, rowStride, sizeof(uchar)};
    for (int p = 0; p < channels; ++p)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                auto &pixel = test::ValueAt<uchar>(srcHost, strides, int4{x, y, p, 0});
                pixel       = static_cast<uchar>((17 * y + 13 * x + 29 * p + 5) % 251);
            }
        }
    }

    BrightnessContrast<uchar, uchar>(srcHost, refHost, strides, strides, int3{width, height, 1}, channels, 0, 1.f, 2.f,
                                     0.f, 128.f);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcAllocation, srcHost.data(), bufferSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstAllocation, dstHost.data(), bufferSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, src, dst, 1.0, 2.0, 0.0, 128.0, true));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstHost.data(), dstAllocation, bufferSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(refHost, dstHost);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpBrightnessContrastCoverage, tensor_planar_tall_padded_rows_decline_stream)
{
    constexpr int    width        = 1;
    constexpr int    height       = 65536;
    constexpr int    rowStride    = 2;
    constexpr int    planeStride  = rowStride * height;
    constexpr int    sampleStride = planeStride;
    constexpr size_t bufferSize   = sampleStride;

    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bufferSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), bufferSize));

    auto wrapTensor = [&](NVCVByte *base)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = base;
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = planeStride;
        buffer.strides[2] = rowStride;
        buffer.strides[3] = sizeof(uchar);
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, 1, height, width}, "NCHW"},
            nvcv::TYPE_U8, buffer
        });
    };

    nvcv::Tensor src = wrapTensor(srcAllocation);
    nvcv::Tensor dst = wrapTensor(dstAllocation);

    std::vector<uint8_t> srcHost(bufferSize, uint8_t{0xA5});
    std::vector<uint8_t> refHost(bufferSize, uint8_t{0xCD});
    std::vector<uint8_t> dstHost(bufferSize, uint8_t{0xCD});
    const long4_16a      strides{sampleStride, planeStride, rowStride, sizeof(uchar)};
    for (int y = 0; y < height; ++y)
    {
        auto &pixel = test::ValueAt<uchar>(srcHost, strides, int4{0, y, 0, 0});
        pixel       = static_cast<uchar>((17 * y + 5) % 251);
    }

    BrightnessContrast<uchar, uchar>(srcHost, refHost, strides, strides, int3{width, height, 1}, 1, 0, 1.f, 2.f, 0.f,
                                     128.f);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcAllocation, srcHost.data(), bufferSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstAllocation, dstHost.data(), bufferSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, src, dst, 1.0, 2.0, 0.0, 128.0, true));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstHost.data(), dstAllocation, bufferSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(refHost, dstHost);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

// Parameters: width, height, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpBrightnessContrastPlanar,
    test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {64, 48, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {37, 29, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {41, 33, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {35, 31, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    // F16 parity is bit-exact: both layouts run the same float chain and round to half once.
    {41, 33, 2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
    {35, 31, 1, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

// clang-format on

// A double represents INT64_MAX as 2^63, outside the signed destination range.
TEST(OpBrightnessContrastScalar, s64_saturates_before_integer_conversion)
{
    constexpr int64_t          hi    = std::numeric_limits<int64_t>::max();
    constexpr int64_t          below = hi - 1023; // next representable double below 2^63
    const std::vector<int64_t> input{hi, below, 0};
    const nvcv::ImageFormat    format{NVCV_IMAGE_FORMAT_S64};
    nvcv::Tensor               src     = nvcv::util::CreateTensor(1, 3, 1, format);
    nvcv::Tensor               dst     = nvcv::util::CreateTensor(1, 3, 1, format);
    auto                       srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto                       dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    const size_t bytes = input.size() * sizeof(int64_t);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice));
    cvcuda::BrightnessContrast op;
    for (bool clamp : {false, true})
    {
        for (auto mode : {NVCV_ROUND_NEAREST, NVCV_ROUND_TRUNCATE})
        {
            for (double brightness : {1., 2.})
            {
                SCOPED_TRACE(testing::Message() << clamp << ":" << mode << ":" << brightness);
                op(nullptr, src, dst, brightness, 1., 0., 0., clamp, mode);
                ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
                std::vector<int64_t> actual(input.size());
                ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstData->basePtr(), bytes, cudaMemcpyDeviceToHost));
                const std::vector<int64_t> expected{hi, brightness == 1. ? below : hi, 0};
                EXPECT_EQ(expected, actual);
            }
        }
    }
}

TEST(OpBrightnessContrastScalar, legacy_c_submits_use_nearest_even_at_exact_ties)
{
    const std::array<uint8_t, 2> input{1, 3};
    const std::array<uint8_t, 2> nearestExpected{0, 2};
    const std::array<uint8_t, 2> truncatedExpected{0, 1};
    cvcuda::BrightnessContrast   op;

    nvcv::Tensor tensorIn      = nvcv::util::CreateTensor(1, 2, 1, nvcv::FMT_U8);
    nvcv::Tensor tensorOut     = nvcv::util::CreateTensor(1, 2, 1, nvcv::FMT_U8);
    auto         tensorInData  = tensorIn.exportData<nvcv::TensorDataStridedCuda>();
    auto         tensorOutData = tensorOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(tensorInData && tensorOutData);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(tensorInData->basePtr(), input.data(), input.size(), cudaMemcpyHostToDevice));

    ASSERT_EQ(NVCV_SUCCESS, cvcudaBrightnessContrastScalarSubmit(op.handle(), nullptr, tensorIn.handle(),
                                                                 tensorOut.handle(), 0.5, 1., 0., 0., false));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    std::array<uint8_t, 2> actual{};
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), tensorOutData->basePtr(), actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(nearestExpected, actual);

    ASSERT_EQ(NVCV_SUCCESS,
              cvcudaBrightnessContrastScalarRoundSubmit(op.handle(), nullptr, tensorIn.handle(), tensorOut.handle(),
                                                        0.5, 1., 0., 0., false, NVCV_ROUND_TRUNCATE));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), tensorOutData->basePtr(), actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(truncatedExpected, actual);

    nvcv::Image imageIn({2, 1}, nvcv::FMT_U8);
    nvcv::Image imageOut({2, 1}, nvcv::FMT_U8);
    auto        imageInData  = imageIn.exportData<nvcv::ImageDataStridedCuda>();
    auto        imageOutData = imageOut.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_TRUE(imageInData && imageOutData);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(imageInData->plane(0).basePtr, input.data(), input.size(), cudaMemcpyHostToDevice));
    nvcv::ImageBatchVarShape batchIn(1);
    nvcv::ImageBatchVarShape batchOut(1);
    batchIn.pushBack(imageIn);
    batchOut.pushBack(imageOut);

    ASSERT_EQ(NVCV_SUCCESS, cvcudaBrightnessContrastVarShapeScalarSubmit(op.handle(), nullptr, batchIn.handle(),
                                                                         batchOut.handle(), 0.5, 1., 0., 0., false));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), imageOutData->plane(0).basePtr, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(nearestExpected, actual);

    ASSERT_EQ(NVCV_SUCCESS, cvcudaBrightnessContrastVarShapeScalarRoundSubmit(op.handle(), nullptr, batchIn.handle(),
                                                                              batchOut.handle(), 0.5, 1., 0., 0., false,
                                                                              NVCV_ROUND_TRUNCATE));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), imageOutData->plane(0).basePtr, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(truncatedExpected, actual);
}

// u8 * 0.5 puts every odd input on a .5 boundary, so the two modes separate on half the values.
TEST(OpBrightnessContrastScalar, round_mode_selects_integer_conversion)
{
    const std::vector<uint8_t> srcVec{0, 1, 2, 3, 4, 5,   250, 251, 0, 1, 2, 3, 4, 5,   250, 251, 0,
                                      1, 2, 3, 4, 5, 250, 251, 0,   1, 2, 3, 4, 5, 250, 251, 0};
    const std::vector<uint8_t> expectedNearest{0, 0, 1, 2, 2, 2,   125, 126, 0, 0, 1, 2, 2, 2,   125, 126, 0,
                                               0, 1, 2, 2, 2, 125, 126, 0,   0, 1, 2, 2, 2, 125, 126, 0};
    const std::vector<uint8_t> expectedTruncate{0, 0, 1, 1, 2, 2,   125, 125, 0, 0, 1, 1, 2, 2,   125, 125, 0,
                                                0, 1, 1, 2, 2, 125, 125, 0,   0, 1, 1, 2, 2, 125, 125, 0};

    const auto   width       = static_cast<int>(srcVec.size());
    nvcv::Tensor src         = nvcv::util::CreateTensor(1, width, 1, nvcv::FMT_U8);
    nvcv::Tensor nearestDst  = nvcv::util::CreateTensor(1, width, 1, nvcv::FMT_U8);
    nvcv::Tensor truncateDst = nvcv::util::CreateTensor(1, width, 1, nvcv::FMT_U8);
    auto         srcData     = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         nearestData = nearestDst.exportData<nvcv::TensorDataStridedCuda>();
    auto         truncData   = truncateDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && nearestData && truncData);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, src, nearestDst, 0.5, 1., 0., 0., NVCV_ROUND_NEAREST));
    ASSERT_NO_THROW(op(stream, src, truncateDst, 0.5, 1., 0., 0., NVCV_ROUND_TRUNCATE));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> nearest(srcVec.size());
    std::vector<uint8_t> truncated(srcVec.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(nearest.data(), nearestData->basePtr(), nearest.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(truncated.data(), truncData->basePtr(), truncated.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expectedNearest, nearest);
    EXPECT_EQ(expectedTruncate, truncated);

    // Omitting the round mode has to keep the released behavior.
    ASSERT_NO_THROW(op(stream, src, nearestDst, 0.5, 1., 0., 0., true));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(nearest.data(), nearestData->basePtr(), nearest.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expectedNearest, nearest);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBrightnessContrastScalar, round_mode_does_not_change_float_output)
{
    const std::vector<float> srcVec{0.f, 0.25f, 0.75f, 1.f};

    nvcv::Tensor src         = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_F32);
    nvcv::Tensor nearestDst  = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_F32);
    nvcv::Tensor truncateDst = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_F32);
    auto         srcData     = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         nearestData = nearestDst.exportData<nvcv::TensorDataStridedCuda>();
    auto         truncData   = truncateDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && nearestData && truncData);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(srcData->basePtr(), srcVec.data(), srcVec.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, src, nearestDst, 0.5, 1., 0., 0., true, NVCV_ROUND_NEAREST));
    ASSERT_NO_THROW(op(stream, src, truncateDst, 0.5, 1., 0., 0., true, NVCV_ROUND_TRUNCATE));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<float> nearest(srcVec.size());
    std::vector<float> truncated(srcVec.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(nearest.data(), nearestData->basePtr(), nearest.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(truncated.data(), truncData->basePtr(), truncated.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
    EXPECT_EQ(nearest, truncated);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBrightnessContrastScalar, explicit_round_modes_planar_and_varshape)
{
    for (auto mode : {NVCV_ROUND_NEAREST, NVCV_ROUND_TRUNCATE})
    {
        const auto invoke = [mode](cudaStream_t stream, const auto &src, const auto &dst, nvcv::ImageFormat)
        {
            cvcuda::BrightnessContrast op;
            op(stream, src, dst, 0.5, 1., 0., 0., mode);
        };
        const auto input = [](int, int width, int height, int channels)
        {
            std::vector<uint8_t> values(width * height * channels);
            for (size_t i = 0; i < values.size(); ++i) values[i] = (i / channels) % 2 ? 3 : 1;
            return values;
        };
        const auto compare = [mode](const auto &interleaved, const auto &planar)
        {
            EXPECT_EQ(interleaved, planar);
            for (size_t i = 0; i < interleaved.size(); ++i)
            {
                EXPECT_EQ(interleaved[i], (i / 3) % 2 ? (mode == NVCV_ROUND_NEAREST ? 2 : 1) : 0);
            }
        };
        test::planar::RunTensorParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 4, 2, 4, 2, 2, invoke, input, compare);
        test::planar::RunVarShapeParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 4, 2, 4, 2, 2, invoke, input, compare);
    }
}

TEST(OpBrightnessContrastScalar, negative_half_ties_without_clamp)
{
    const std::vector<int16_t> input{-7, -5, -3, -1, 1, 3, 5, 7};
    nvcv::Tensor               src     = nvcv::util::CreateTensor(1, 8, 1, nvcv::FMT_S16);
    nvcv::Tensor               dst     = nvcv::util::CreateTensor(1, 8, 1, nvcv::FMT_S16);
    auto                       srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto                       dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(srcData->basePtr(), input.data(), input.size() * sizeof(int16_t), cudaMemcpyHostToDevice));
    cvcuda::BrightnessContrast op;
    for (auto mode : {NVCV_ROUND_NEAREST, NVCV_ROUND_TRUNCATE})
    {
        op(nullptr, src, dst, 0.5, 1., 0., 0., false, mode);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
        std::vector<int16_t> actual(input.size());
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstData->basePtr(), actual.size() * sizeof(int16_t),
                                          cudaMemcpyDeviceToHost));
        const std::vector<int16_t> expected = mode == NVCV_ROUND_NEAREST
                                                ? std::vector<int16_t>{-4, -2, -2, 0, 0, 2, 2, 4}
                                                : std::vector<int16_t>{-3, -2, -1, 0, 0, 1, 2, 3};
        EXPECT_EQ(expected, actual);
    }
}

TEST(OpBrightnessContrast_Negative, scalar_rejects_invalid_round_mode)
{
    nvcv::Tensor src = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_U8);
    nvcv::Tensor dst = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_U8);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BrightnessContrast op;
    nvcv::ImageBatchVarShape   srcBatch(1);
    nvcv::ImageBatchVarShape   dstBatch(1);
    srcBatch.pushBack(nvcv::Image({4, 1}, nvcv::FMT_U8));
    dstBatch.pushBack(nvcv::Image({4, 1}, nvcv::FMT_U8));

#ifndef ENABLE_SANITIZER
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, TestCAPIEnumBrightnessContrastScalarRoundSubmit(
                                               op.handle(), stream, src.handle(), dst.handle(), 0.5, 1., 0., 0., true));
#endif

    for (auto mode : {NVCV_ROUND_NEAREST_WRAP, NVCV_ROUND_TRUNCATE_WRAP})
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { op(stream, src, dst, 0.5, 1., 0., 0., true, mode); }));
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  cvcudaBrightnessContrastScalarRoundSubmit(op.handle(), stream, src.handle(), dst.handle(), 0.5, 1.,
                                                            0., 0., true, mode));
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  cvcudaBrightnessContrastVarShapeScalarRoundSubmit(op.handle(), stream, srcBatch.handle(),
                                                                    dstBatch.handle(), 0.5, 1., 0., 0., true, mode));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpBrightnessContrastPlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>());
}

TEST_P(OpBrightnessContrastPlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>());
}

// Declared hardening: a batch above the CUDA grid.z cap is now an API error rather than an opaque
// cudaErrorInvalidConfiguration, matching the other operators in the same-shape family.
TEST(OpBrightnessContrast_Negative, tensor_batch_exceeds_grid_z_limit)
{
    constexpr int64_t kSamples = 65536; // one past the grid.z cap
    constexpr int64_t kBufSize = kSamples + 16;

    NVCVByte *srcAlloc{};
    NVCVByte *dstAlloc{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAlloc), kBufSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAlloc), kBufSize));

    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    srcBuffer.basePtr    = srcAlloc;
    srcBuffer.strides[0] = 1;
    srcBuffer.strides[1] = 1;
    srcBuffer.strides[2] = 1;
    srcBuffer.strides[3] = 1;
    auto dstBuffer       = srcBuffer;
    dstBuffer.basePtr    = dstAlloc;

    const nvcv::TensorShape shape{
        {kSamples, 1, 1, 1},
        "NHWC"
    };
    nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, srcBuffer});
    nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, dstBuffer});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::BrightnessContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, 1.0, 1.0, 0.0, 0.0); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAlloc));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAlloc));
}

// Declared hardening, the var-shape half of tensor_batch_exceeds_grid_z_limit: a batch above the
// CUDA grid.z cap is an API error rather than an opaque cudaErrorInvalidConfiguration, matching the
// other operators in the same-shape family. The images wrap one allocation so the batch stays cheap.
TEST(OpBrightnessContrast_Negative, varshape_batch_exceeds_grid_z_limit)
{
    constexpr int kImages = 65536; // one past the grid.z cap

    NVCVByte *srcAlloc{};
    NVCVByte *dstAlloc{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAlloc), kImages));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAlloc), kImages));

    auto wrap1x1 = [](NVCVByte *base, int i)
    {
        nvcv::ImageDataStridedCuda::Buffer buf{};
        buf.numPlanes           = 1;
        buf.planes[0].width     = 1;
        buf.planes[0].height    = 1;
        buf.planes[0].rowStride = 1;
        buf.planes[0].basePtr   = base + i;
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_U8, buf});
    };

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    srcImages.reserve(kImages);
    dstImages.reserve(kImages);
    for (int i = 0; i < kImages; ++i)
    {
        srcImages.emplace_back(wrap1x1(srcAlloc, i));
        dstImages.emplace_back(wrap1x1(dstAlloc, i));
    }

    nvcv::ImageBatchVarShape src(kImages);
    nvcv::ImageBatchVarShape dst(kImages);
    src.pushBack(srcImages.begin(), srcImages.end());
    dst.pushBack(dstImages.begin(), dstImages.end());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::BrightnessContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, 1.0, 1.0, 0.0, 0.0); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAlloc));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAlloc));
}

TEST(OpBrightnessContrast_Negative, createWithNullHandle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaBrightnessContrastCreate(nullptr));
}

TEST(OpBrightnessContrast_Negative, two_channel_planar_tensor)
{
    nvcv::TensorShape shape{
        {1, 2, 8, 8},
        "NCHW"
    };
    nvcv::Tensor src(shape, nvcv::TYPE_U8);
    nvcv::Tensor dst(shape, nvcv::TYPE_U8);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BrightnessContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&op, stream, &src, &dst] {
                                                   op(stream, src, dst, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                                                      nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr});
                                               }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBrightnessContrast_Negative, tensor_layout_and_channel_validation)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BrightnessContrast op;
    auto                       expectInvalid = [&op, stream](const nvcv::Tensor &src, const nvcv::Tensor &dst)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, 1., 1., 0., 0.); }));
    };

    nvcv::Tensor invalidLayout(
        {
            {8, 8, 1},
            "WHC"
    },
        nvcv::TYPE_U8);
    expectInvalid(invalidLayout, invalidLayout);

    nvcv::Tensor srcThree(
        {
            {1, 8, 8, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dstFour(
        {
            {1, 8, 8, 4},
            "NHWC"
    },
        nvcv::TYPE_U8);
    expectInvalid(srcThree, dstFour);

    nvcv::Tensor fiveChannel(
        {
            {1, 8, 8, 5},
            "NHWC"
    },
        nvcv::TYPE_U8);
    expectInvalid(fiveChannel, fiveChannel);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBrightnessContrast_Negative, tensor_sample_count_mismatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src(
        {
            {2, 8, 8, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dst(
        {
            {1, 8, 8, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);

    cvcuda::BrightnessContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, 1., 1., 0., 0.); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBrightnessContrast_Negative, varshape_channel_validation)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BrightnessContrast op;
    auto                       expectInvalid = [&op, stream](nvcv::ImageFormat srcFmt, nvcv::ImageFormat dstFmt)
    {
        nvcv::Image              srcImage({8, 8}, srcFmt);
        nvcv::Image              dstImage({8, 8}, dstFmt);
        nvcv::ImageBatchVarShape src(1);
        nvcv::ImageBatchVarShape dst(1);
        src.pushBack(srcImage);
        dst.pushBack(dstImage);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, 1., 1., 0., 0.); }));
    };

    expectInvalid(nvcv::FMT_RGB8, nvcv::FMT_RGBA8);

    const nvcv::ImageFormat twoChannelPlanar{NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, X8, X8)};
    expectInvalid(twoChannelPlanar, twoChannelPlanar);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBrightnessContrast_Negative, varshape_rejects_multi_channel_planes)
{
    const nvcv::ImageFormat twoChannelPlanes{
        NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8, X8_Y8)};

    nvcv::Image              srcImage({8, 8}, twoChannelPlanes);
    nvcv::Image              dstImage({8, 8}, twoChannelPlanes);
    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);
    src.pushBack(srcImage);
    dst.pushBack(dstImage);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BrightnessContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, 1., 1., 0., 0.); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

#define NVCV_NEGATIVE_CASE(SrcShape, DstShape, SrcType, DstType, SrcImgFormat, DstImgFormat, ArgType, ArgCounts) \
    ttype::Types<ttype::Value<SrcShape>, ttype::Value<DstShape>, SrcType, DstType, ttype::Value<SrcImgFormat>,   \
                 ttype::Value<DstImgFormat>, ArgType, ttype::Value<ArgCounts>>

NVCV_TYPED_TEST_SUITE(
    OpBrightnessContrast_Negative,
    ttype::Types<
        // invalid data type
        NVCV_NEGATIVE_CASE(NVCV_SHAPE(32, 32, 1), NVCV_SHAPE(32, 32, 1), char, char, NVCV_IMAGE_FORMAT_S8,
                           NVCV_IMAGE_FORMAT_S8, float, NVCV_ARGS_COUNT(1, 1, 1, 1)),
        NVCV_NEGATIVE_CASE(NVCV_SHAPE(32, 32, 1), NVCV_SHAPE(32, 32, 1), uchar, uchar, NVCV_IMAGE_FORMAT_U8,
                           NVCV_IMAGE_FORMAT_U8, double, NVCV_ARGS_COUNT(1, 1, 1, 1)),
        NVCV_NEGATIVE_CASE(NVCV_SHAPE(32, 32, 1), NVCV_SHAPE(32, 32, 1), int, int, NVCV_IMAGE_FORMAT_S32,
                           NVCV_IMAGE_FORMAT_S32, float, NVCV_ARGS_COUNT(1, 1, 1, 1)),
        // invalid shape
        NVCV_NEGATIVE_CASE(NVCV_SHAPE(40, 39, 2), NVCV_SHAPE(40, 39, 1), uchar, uchar, NVCV_IMAGE_FORMAT_U8,
                           NVCV_IMAGE_FORMAT_U8, float, NVCV_ARGS_COUNT(0, 1, 1, 1)),
        NVCV_NEGATIVE_CASE(NVCV_SHAPE(41, 39, 1), NVCV_SHAPE(40, 39, 1), uchar, uchar, NVCV_IMAGE_FORMAT_U8,
                           NVCV_IMAGE_FORMAT_U8, float, NVCV_ARGS_COUNT(0, 1, 1, 1)),
        // invalid planes
        NVCV_NEGATIVE_CASE(NVCV_SHAPE(42, 41, 7), NVCV_SHAPE(42, 41, 7), uchar3, uchar3, NVCV_IMAGE_FORMAT_RGB8,
                           NVCV_IMAGE_FORMAT_RGB8p, float, NVCV_ARGS_COUNT(7, 7, 7, 7))>);

TYPED_TEST(OpBrightnessContrast_Negative, invalid_parameters_src_dst_tensor)
{
    const int3 shapeSrc = ttype::GetValue<TypeParam, 0>;
    const int3 shapeDst = ttype::GetValue<TypeParam, 1>;

    using SrcType = ttype::GetType<TypeParam, 2>;
    using DstType = ttype::GetType<TypeParam, 3>;
    using SrcBT   = cuda::BaseType<SrcType>;
    using DstBT   = cuda::BaseType<DstType>;

    const nvcv::ImageFormat srcImgFormat{ttype::GetValue<TypeParam, 4>};
    const nvcv::ImageFormat dstImgFormat{ttype::GetValue<TypeParam, 5>};

    using ArgType        = ttype::GetType<TypeParam, 6>;
    const int4 argCounts = ttype::GetValue<TypeParam, 7>;

    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(shapeSrc.z, shapeSrc.x, shapeSrc.y, srcImgFormat);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(shapeDst.z, shapeDst.x, shapeDst.y, dstImgFormat);

    auto            srcRand = MakeValueDistribution<SrcBT>();
    auto            dstRand = MakeValueDistribution<DstBT>();
    std::mt19937_64 rng(12345);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    BrightnessContrastArguments<ArgType> args;
    args.template populate<SrcBT, DstBT>(rng, argCounts, srcRand, dstRand);

    cvcuda::BrightnessContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&op, &stream, &srcTensor, &dstTensor, &args]
                                               {
                                                   op(stream, srcTensor, dstTensor, args.brightness.m_argTensor,
                                                      args.contrast.m_argTensor, args.brightnessShift.m_argTensor,
                                                      args.contrastCenter.m_argTensor);
                                               }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TYPED_TEST(OpBrightnessContrast_Negative, invalid_parameters_src_dst_varshape)
{
    const int3 shapeSrc = ttype::GetValue<TypeParam, 0>;
    const int3 shapeDst = ttype::GetValue<TypeParam, 1>;

    using SrcType = ttype::GetType<TypeParam, 2>;
    using DstType = ttype::GetType<TypeParam, 3>;
    using SrcBT   = cuda::BaseType<SrcType>;
    using DstBT   = cuda::BaseType<DstType>;

    const nvcv::ImageFormat srcImgFormat{ttype::GetValue<TypeParam, 4>};
    const nvcv::ImageFormat dstImgFormat{ttype::GetValue<TypeParam, 5>};

    using ArgType        = ttype::GetType<TypeParam, 6>;
    const int4 argCounts = ttype::GetValue<TypeParam, 7>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;

    auto            srcRand = MakeValueDistribution<SrcBT>();
    auto            dstRand = MakeValueDistribution<DstBT>();
    std::mt19937_64 rng(12345);

    for (int z = 0; z < shapeSrc.z; ++z)
    {
        nvcv::Size2D imgShapeSrc{shapeSrc.x, shapeSrc.y};
        imgSrc.emplace_back(imgShapeSrc, srcImgFormat);
    }

    for (int z = 0; z < shapeDst.z; ++z)
    {
        nvcv::Size2D imgShapeDst{shapeDst.x, shapeDst.y};
        imgDst.emplace_back(imgShapeDst, dstImgFormat);
    }

    nvcv::ImageBatchVarShape batchSrc(shapeSrc.z);
    nvcv::ImageBatchVarShape batchDst(shapeDst.z);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    BrightnessContrastArguments<ArgType> args;
    args.template populate<SrcBT, DstBT>(rng, argCounts, srcRand, dstRand);

    cvcuda::BrightnessContrast op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&op, &stream, &batchSrc, &batchDst, &args]
                                               {
                                                   op(stream, batchSrc, batchDst, args.brightness.m_argTensor,
                                                      args.contrast.m_argTensor, args.brightnessShift.m_argTensor,
                                                      args.contrastCenter.m_argTensor);
                                               }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBrightnessContrast_Negative, different_format_varshape)
{
    std::vector<std::pair<nvcv::ImageFormat, nvcv::ImageFormat>> extraFmts{
        { nvcv::FMT_RGB8, nvcv::FMT_RGBA8},
        {nvcv::FMT_RGBA8,  nvcv::FMT_RGB8}
    };

    for (const auto &[extraFmtSrc, extraFmtDst] : extraFmts)
    {
        const int               numSamples = 10;
        const int               x          = 32;
        const int3              shape{x, x, numSamples};
        const nvcv::ImageFormat imgFmt = nvcv::FMT_RGB8;
        nvcv::Size2D            imgShape2D{x, x};

        nvcv::Tensor srcTensor = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, imgFmt);
        nvcv::Tensor dstTensor = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, imgFmt);
        nvcv::Tensor brightness({{numSamples}, "W"}, nvcv::TYPE_F32);
        nvcv::Tensor contrast({{numSamples}, "W"}, nvcv::TYPE_F32);
        nvcv::Tensor brightnessShift({{numSamples}, "W"}, nvcv::TYPE_F32);
        nvcv::Tensor contrastCenter({{numSamples}, "W"}, nvcv::TYPE_F32);

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;

        for (int z = 0; z < numSamples - 1; ++z)
        {
            imgSrc.emplace_back(imgShape2D, imgFmt);
            imgDst.emplace_back(imgShape2D, imgFmt);
        }
        imgSrc.emplace_back(imgShape2D, extraFmtSrc);
        imgDst.emplace_back(imgShape2D, extraFmtDst);

        nvcv::ImageBatchVarShape batchSrc(numSamples);
        nvcv::ImageBatchVarShape batchDst(numSamples);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        cvcuda::BrightnessContrast op;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall(
                      [&op, &stream, &batchSrc, &batchDst, &brightness, &contrast, &brightnessShift, &contrastCenter]
                      { op(stream, batchSrc, batchDst, brightness, contrast, brightnessShift, contrastCenter); }));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
}

TEST(OpBrightnessContrast_Negative, invalid_parameters_tensors)
{
    const int               numSamples = 10;
    const int               x          = 32;
    const int3              shape{x, x, numSamples};
    const nvcv::ImageFormat imgFmt{NVCV_IMAGE_FORMAT_RGB8};

    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, imgFmt);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, imgFmt);
    nvcv::Tensor validBrightnessTensor({{numSamples}, "W"}, nvcv::TYPE_F32);
    nvcv::Tensor validContrastTensor({{numSamples}, "W"}, nvcv::TYPE_F32);
    nvcv::Tensor validBrightnessShiftTensor({{numSamples}, "W"}, nvcv::TYPE_F32);
    nvcv::Tensor validContrastCenterTensor({{numSamples}, "W"}, nvcv::TYPE_F32);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::BrightnessContrast op;

    auto runOp = [&op, &stream, &srcTensor, &dstTensor](const nvcv::Tensor &brightness, const nvcv::Tensor &contrast,
                                                        const nvcv::Tensor &brightnessShift,
                                                        const nvcv::Tensor &contrastCenter)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall(
                      [&op, &stream, &srcTensor, &dstTensor, &brightness, &contrast, &brightnessShift, &contrastCenter]
                      { op(stream, srcTensor, dstTensor, brightness, contrast, brightnessShift, contrastCenter); }));
    };
    auto runOpWithInvalidTensor
        = [&runOp, &validContrastTensor, &validBrightnessShiftTensor, &validContrastCenterTensor,
           &validBrightnessTensor](const nvcv::Tensor &invalidTensor)
    {
        runOp(invalidTensor, validContrastTensor, validBrightnessShiftTensor, validContrastCenterTensor);
        runOp(validBrightnessTensor, invalidTensor, validBrightnessShiftTensor, validContrastCenterTensor);
        runOp(validBrightnessTensor, validContrastTensor, invalidTensor, validContrastCenterTensor);
        runOp(validBrightnessTensor, validContrastTensor, validBrightnessShiftTensor, invalidTensor);
    };

    // not 1D tensor
    {
        nvcv::Tensor invalidTensor(
            {
                {numSamples, 2},
                "NW"
        },
            nvcv::TYPE_F32);
        runOpWithInvalidTensor(invalidTensor);
    }

    // length mismatch
    {
        nvcv::Tensor invalidTensor({{numSamples + 1}, "N"}, nvcv::TYPE_F32);
        runOpWithInvalidTensor(invalidTensor);
    }

    // invalid dtype
    {
        nvcv::Tensor invalidTensor({{numSamples}, "N"}, nvcv::TYPE_U32);
        runOpWithInvalidTensor(invalidTensor);
    }

    // brightness/contrast/brightness shift/contrast center dtype mismatch
    {
        nvcv::Tensor invalidTensor({{numSamples}, "N"}, nvcv::TYPE_F64);
        runOpWithInvalidTensor(invalidTensor);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Regression (bug, not hardening): a tensor whose *dtype* carries the channels while its C axis is
// 1 used to be dispatched as if each element were one base type, and only a fraction of every row
// was transformed. See the same regression in TestOpInvert.cpp for the trace; this operator has
// its own validator (it accepts a dst dtype differing from src), so it needed its own fix and
// tests both sides independently.
TEST(OpBrightnessContrast_Negative, tensor_multi_channel_dtype)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const nvcv::TensorShape shape{
        {1, 4, 4, 1},
        "NHWC"
    };

    auto run = [stream](const nvcv::Tensor &src, const nvcv::Tensor &dst)
    {
        cvcuda::BrightnessContrast op;
        return nvcv::ProtectCall([&op, stream, &src, &dst] { op(stream, src, dst, 1.0, 1.0, 0.0, 0.0); });
    };

    // Source side.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              run(nvcv::Tensor(shape, nvcv::TYPE_3U8), nvcv::Tensor(shape, nvcv::TYPE_3U8)));
    // Destination side alone: this operator allows the two dtypes to differ, so the source-side
    // test above cannot cover it.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              run(nvcv::Tensor(shape, nvcv::TYPE_U8), nvcv::Tensor(shape, nvcv::TYPE_3U8)));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
