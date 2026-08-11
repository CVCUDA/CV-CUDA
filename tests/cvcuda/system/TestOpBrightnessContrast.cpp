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

#include "PlanarParityUtils.hpp"

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpBrightnessContrast.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

using uchar = unsigned char;

template<typename T>
using uniform_distribution
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

template<typename ValueType>
void ComparePixel(std::vector<uint8_t> &dst, std::vector<uint8_t> &ref, const long4_16a &strides, const int4 &coord,
                  float tolerance, bool requireExact)
{
    auto dstPixel = test::ValueAt<ValueType>(dst, strides, coord);
    auto refPixel = test::ValueAt<ValueType>(ref, strides, coord);
    for (int k = 0; k < cuda::NumElements<ValueType>; ++k)
    {
        if (requireExact)
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
        ArgT v = cuda::GetElement(srcPixel, k);
        v      = brightness * (contrast * (v - contrastCenter) + contrastCenter) + brightnessShift;
        cuda::GetElement(outPixel, k) = cuda::SaturateCast<DstBT>(v);
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
#define NVCV_IMAGE_FORMAT_S16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X16)

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
                                             NVCV_IMAGE_FORMAT_RGBAf32, float, NVCV_ARGS_COUNT(0, 0, 0, 18))>);

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

    uniform_distribution<SrcBT> srcRand(SrcBT{0}, std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::max : SrcBT{1});
    uniform_distribution<DstBT> dstRand(DstBT{0}, std::is_integral_v<DstBT> ? cuda::TypeTraits<DstBT>::max : DstBT{1});
    std::mt19937_64             rng(12345);

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

    float      absTolerance = std::is_integral_v<DstBT> ? 1.f : 1e-5f;
    const bool requireExact = std::is_integral_v<SrcBT> && std::is_same_v<SrcType, DstType> && argCounts.x == 0
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
    uniform_distribution<SrcBT> srcRand(SrcBT{0}, std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::max : SrcBT{1});
    uniform_distribution<DstBT> dstRand(DstBT{0}, std::is_integral_v<DstBT> ? cuda::TypeTraits<DstBT>::max : DstBT{1});
    std::mt19937_64             rng(12345);

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

        float      absTolerance = std::is_integral_v<DstBT> ? 1.f : 1e-5f;
        const bool requireExact = std::is_integral_v<SrcBT> && std::is_same_v<SrcType, DstType> && argCounts.x == 0
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

// Parameters: width, height, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpBrightnessContrastPlanar,
    test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {64, 48, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {37, 29, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {41, 33, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {35, 31, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

// clang-format on

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

    uniform_distribution<SrcBT> srcRand(SrcBT{0}, std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::max : SrcBT{1});
    uniform_distribution<DstBT> dstRand(DstBT{0}, std::is_integral_v<DstBT> ? cuda::TypeTraits<DstBT>::max : DstBT{1});
    std::mt19937_64             rng(12345);

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

    uniform_distribution<SrcBT> srcRand(SrcBT{0}, std::is_integral_v<SrcBT> ? cuda::TypeTraits<SrcBT>::max : SrcBT{1});
    uniform_distribution<DstBT> dstRand(DstBT{0}, std::is_integral_v<DstBT> ? cuda::TypeTraits<DstBT>::max : DstBT{1});
    std::mt19937_64             rng(12345);

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
