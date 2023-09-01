/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/InterpUtils.hpp>
#include <common/TypedTests.hpp>
#include <cvcuda/OpBrightnessContrast.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <util/TensorDataUtils.hpp>

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
void CompareTensors(std::vector<uint8_t> &dst, std::vector<uint8_t> &ref, const long4 &strides, const int3 &shape,
                    int numPlanes, float tolerance)
{
    for (int z = 0; z < shape.z; ++z)
    {
        for (int p = 0; p < numPlanes; p++)
        {
            for (int y = 0; y < shape.y; ++y)
            {
                for (int x = 0; x < shape.x; ++x)
                {
                    for (int k = 0; k < cuda::NumElements<ValueType>; ++k)
                    {
                        auto val     = cuda::GetElement(test::ValueAt<ValueType>(dst, strides, int4{x, y, p, z}), k);
                        auto ref_val = cuda::GetElement(test::ValueAt<ValueType>(ref, strides, int4{x, y, p, z}), k);
                        EXPECT_NEAR(val, ref_val, tolerance);
                    }
                }
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
    if constexpr (std::is_same_v<ValueType, uchar>)
    {
        return 128.;
    }
    else if constexpr (std::is_same_v<ValueType, unsigned short>)
    {
        return 32768.;
    }
    else if constexpr (std::is_same_v<ValueType, short>)
    {
        return 16384.;
    }
    else if constexpr (std::is_same_v<ValueType, unsigned int>)
    {
        return 2147483648.;
    }
    else if constexpr (std::is_same_v<ValueType, int>)
    {
        return 1073741824;
    }
    else
    {
        static_assert(!std::is_integral_v<ValueType>);
        {
            return 0.5;
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
                auto &v = *reinterpret_cast<BT *>(&m_argVec[m_stride * z]);
                v       = coeffDist(rng);
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
            return *reinterpret_cast<const BT *>(&m_argVec[0]);
        }
        else
        {
            return *reinterpret_cast<const BT *>(&m_argVec[m_stride * idx]);
        }
    }

    BT                   m_default;
    int                  m_numSamples;
    nvcv::Tensor         m_argTensor;
    long                 m_stride = 0;
    std::vector<uint8_t> m_argVec = {};
};

template<typename SrcType, typename DstType, typename ArgT>
void BrightnessContrast(std::vector<uint8_t> &src, std::vector<uint8_t> &dst, const long4 &srcStrides,
                        const long4 &dstStrides, const int3 &shape, int numPlanes, int sampleIdx, ArgT brightness,
                        ArgT contrast, ArgT brightnessShift, ArgT contrastCenter)
{
    using DstBT               = cuda::BaseType<DstType>;
    constexpr int numChannels = cuda::NumElements<DstType>;
    static_assert(cuda::NumElements<SrcType> == numChannels);

    for (int p = 0; p < numPlanes; p++)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                int4  coord{x, y, p, sampleIdx};
                auto  srcPixel = test::ValueAt<SrcType>(src, srcStrides, coord);
                auto &outPixel = test::ValueAt<DstType>(dst, dstStrides, coord);
                for (int k = 0; k < numChannels; k++)
                {
                    ArgT v = cuda::GetElement(srcPixel, k);
                    v      = brightness * (contrast * (v - contrastCenter) + contrastCenter) + brightnessShift;
                    cuda::GetElement(outPixel, k) = cuda::SaturateCast<DstBT>(v);
                }
            }
        }
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

NVCV_TYPED_TEST_SUITE(OpBrightnessContrast,
                      ttype::Types<NVCV_CASE(NVCV_SHAPE(41, 39, 1), uchar, uchar, NVCV_IMAGE_FORMAT_U8,
                                             NVCV_IMAGE_FORMAT_U8, float, NVCV_ARGS_COUNT(0, 1, 1, 1)),
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
    int numSamples = srcAccess->numSamples();
    ASSERT_EQ(numSamples, dstAccess->numSamples());
    for (auto argCount : {argCounts.x, argCounts.y, argCounts.z, argCounts.w})
    {
        ASSERT_TRUE(argCount == 0 || argCount == 1 || argCount == numSamples);
    }
    int numRows = dstAccess->numRows();
    ASSERT_EQ(srcAccess->numRows(), numRows);
    ASSERT_TRUE(srcAccess->numCols() == dstAccess->numCols() && srcAccess->numChannels() == dstAccess->numChannels());
    long4 srcStrides{srcAccess->sampleStride(), srcAccess->planeStride(), srcAccess->rowStride(),
                     srcAccess->colStride()};
    long4 dstStrides{dstAccess->sampleStride(), dstAccess->planeStride(), dstAccess->rowStride(),
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

    for (int z = 0; z < shape.z; ++z)
    {
        for (int p = 0; p < numPlanes; p++)
        {
            for (int y = 0; y < shape.y; ++y)
            {
                for (int x = 0; x < shape.x; ++x)
                {
                    auto &pixel = test::ValueAt<SrcType>(srcVec, srcStrides, int4{x, y, p, z});
                    for (int k = 0; k < numChannels; ++k)
                    {
                        cuda::GetElement(pixel, k) = srcRand(rng);
                    }
                }
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), srcBufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    Argument<ArgType> brightness;
    Argument<ArgType> contrast;
    Argument<ArgType> brightnessShift;
    Argument<ArgType> contrastCenter;
    ArgType           normalizationFactor = GetHalfRange<DstBT, ArgType>() / GetHalfRange<SrcBT, ArgType>();
    brightness.populate(rng, argCounts.x, 0., 2. * normalizationFactor, 1.);
    contrast.populate(rng, argCounts.y, 0., 2., 1.);
    brightnessShift.populate(rng, argCounts.z, -dstRand(rng) / 2, dstRand(rng) / 2, 0.);
    contrastCenter.populate(rng, argCounts.w, 0, srcRand(rng), GetHalfRange<SrcBT, ArgType>());

    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, srcTensor, dstTensor, brightness.m_argTensor, contrast.m_argTensor,
                       brightnessShift.m_argTensor, contrastCenter.m_argTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstVec.data(), dstData->basePtr(), dstBufSize, cudaMemcpyDeviceToHost));

    for (int z = 0; z < shape.z; z++)
    {
        BrightnessContrast<SrcType, DstType>(srcVec, refVec, srcStrides, dstStrides, shape, numPlanes, z,
                                             brightness.GetHostElement(z), contrast.GetHostElement(z),
                                             brightnessShift.GetHostElement(z), contrastCenter.GetHostElement(z));
    }

    float absTolerance = std::is_integral_v<DstBT> ? 1 : 1e-5;
    CompareTensors<DstType>(dstVec, refVec, dstStrides, shape, numPlanes, absTolerance);

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

    std::uniform_int_distribution<int> randW(shape.x * 0.5, shape.x * 1.5);
    std::uniform_int_distribution<int> randH(shape.y * 0.5, shape.y * 1.5);
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

        for (int p = 0; p < numPlanes; p++)
        {
            for (int y = 0; y < imgSrc[z].size().h; ++y)
            {
                for (int x = 0; x < imgSrc[z].size().w; ++x)
                {
                    for (int k = 0; k < numChannels; ++k)
                    {
                        cuda::GetElement(test::ValueAt<SrcType>(srcVec[z], srcStrides, int3{x, y, p}), k)
                            = srcRand(rng);
                    }
                }
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

    Argument<ArgType> brightness;
    Argument<ArgType> contrast;
    Argument<ArgType> brightnessShift;
    Argument<ArgType> contrastCenter;
    ArgType           normalizationFactor = GetHalfRange<DstBT, ArgType>() / GetHalfRange<SrcBT, ArgType>();
    brightness.populate(rng, argCounts.x, 0., 2. * normalizationFactor, 1.);
    contrast.populate(rng, argCounts.y, 0., 2., 1.);
    brightnessShift.populate(rng, argCounts.z, -dstRand(rng) / 2, dstRand(rng) / 2, 0.);
    contrastCenter.populate(rng, argCounts.w, 0, srcRand(rng), GetHalfRange<SrcBT, ArgType>());

    cvcuda::BrightnessContrast op;
    ASSERT_NO_THROW(op(stream, batchSrc, batchDst, brightness.m_argTensor, contrast.m_argTensor,
                       brightnessShift.m_argTensor, contrastCenter.m_argTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int z = 0; z < shape.z; z++)
    {
        SCOPED_TRACE(z);

        const auto srcData = imgSrc[z].exportData<nvcv::ImageDataStridedCuda>();
        const auto dstData = imgDst[z].exportData<nvcv::ImageDataStridedCuda>();

        ASSERT_EQ(srcData->numPlanes(), dstData->numPlanes());

        int   srcRowStride = srcData->plane(0).rowStride;
        int   dstRowStride = dstData->plane(0).rowStride;
        long4 srcStrides{0, imgSrc[z].size().h * srcRowStride, srcRowStride, sizeof(SrcType)};
        long4 dstStrides{0, imgDst[z].size().h * dstRowStride, dstRowStride, sizeof(DstType)};

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
                                             brightness.GetHostElement(z), contrast.GetHostElement(z),
                                             brightnessShift.GetHostElement(z), contrastCenter.GetHostElement(z));

        float absTolerance = std::is_integral_v<DstBT> ? 1 : 1e-5;
        CompareTensors<DstType>(dstVec, refVec, dstStrides, sampleShape, numPlanes, absTolerance);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
