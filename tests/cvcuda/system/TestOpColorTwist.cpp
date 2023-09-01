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
#include <cvcuda/OpColorTwist.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/cuda/math/LinAlg.hpp>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

template<typename T, int N, int M>
using Mat = cuda::math::Matrix<T, N, M>;

template<typename T>
using uniform_distribution
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

template<typename ValueType, typename TwistValueType>
void ColorTwist(std::vector<uint8_t> &src, std::vector<uint8_t> &dst, std::vector<uint8_t> &twist, const long3 &strides,
                const long2 &twistStrides, const int3 &shape, bool usePerSampleTwist)
{
    using BT                  = cuda::BaseType<ValueType>;
    using TwistT              = cuda::BaseType<TwistValueType>;
    constexpr int numChannels = cuda::NumElements<ValueType>;
    static_assert(numChannels == 3 || numChannels == 4);

    Mat<TwistT, 3, 4> mix;
    for (int z = 0; z < shape.z; ++z)
    {
        int twistZ = usePerSampleTwist ? z : 0;
        for (int i = 0; i < 3; i++)
        {
            auto row = test::ValueAt<TwistValueType>(twist, twistStrides, int2{i, twistZ});
            for (int j = 0; j < 4; j++)
            {
                mix[i][j] = cuda::GetElement(row, j);
            }
        }
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                int3 coord{x, y, z};
                auto pixel = test::ValueAt<ValueType>(src, strides, coord);

                cuda::math::Vector<TwistT, 4> in;
                for (int k = 0; k < 3; k++)
                {
                    in[k] = cuda::GetElement(pixel, k);
                }
                in[3] = 1.;

                cuda::math::Vector<TwistT, 3> out = mix * in;
                for (int k = 0; k < 3; k++)
                {
                    cuda::GetElement(test::ValueAt<ValueType>(dst, strides, coord), k) = cuda::SaturateCast<BT>(out[k]);
                }
                for (int k = 3; k < numChannels; k++)
                {
                    cuda::GetElement(test::ValueAt<ValueType>(dst, strides, coord), k) = cuda::GetElement(pixel, k);
                }
            }
        }
    }
}

template<typename ValueType>
void CompareTensors(std::vector<uint8_t> &dst, std::vector<uint8_t> &ref, const long3 &strides, const int3 &shape,
                    float tolerance)
{
    for (int z = 0; z < shape.z; ++z)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                for (int k = 0; k < cuda::NumElements<ValueType>; ++k)
                {
                    auto val     = cuda::GetElement(test::ValueAt<ValueType>(dst, strides, int3{x, y, z}), k);
                    auto ref_val = cuda::GetElement(test::ValueAt<ValueType>(ref, strides, int3{x, y, z}), k);
                    EXPECT_NEAR(val, ref_val, tolerance);
                }
            }
        }
    }
}

template<typename TwistValueType>
inline nvcv::Tensor GetTwistTensor(bool usePerSampleArgs, int numSamples)
{
    static_assert(std::is_same_v<float4, TwistValueType> || std::is_same_v<double4, TwistValueType>);
    auto dType = std::is_same_v<float4, TwistValueType> ? nvcv::TYPE_4F32 : nvcv::TYPE_4F64;
    if (usePerSampleArgs)
    {
        nvcv::TensorShape shape{
            {numSamples, 3},
            "NH"
        };
        return nvcv::Tensor{shape, dType};
    }
    nvcv::TensorShape shape{{3}, "H"};
    return nvcv::Tensor{shape, dType};
}

template<typename TwistValueType_>
struct TwistMatrixArgument
{
    using TwistValueType = TwistValueType_;

    template<typename Rng>
    inline void populate(Rng &rng, bool usePerSampleArgs, int numSamples)
    {
        const int numRows = 3;
        const int numCols = 4;

        int numArgs    = usePerSampleArgs ? numSamples : 1;
        m_twistTensor  = GetTwistTensor<TwistValueType>(usePerSampleArgs, numArgs);
        auto twistData = m_twistTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(twistData);

        if (usePerSampleArgs)
        {
            m_twistStrides = {twistData->stride(0), twistData->stride(1)};
        }
        else
        {
            m_twistStrides = {numRows * twistData->stride(0), twistData->stride(0)};
        }
        size_t twistBufSize = m_twistStrides.x * numArgs;
        m_twistVec          = std::vector<uint8_t>(twistBufSize, uint8_t{0});

        std::uniform_real_distribution<float> coeffDist(-10., 10.);
        for (int y = 0; y < numArgs; ++y)
        {
            for (int x = 0; x < numRows; ++x)
            {
                for (int k = 0; k < numCols; ++k)
                {
                    cuda::GetElement(test::ValueAt<TwistValueType>(m_twistVec, m_twistStrides, int2{x, y}), k)
                        = coeffDist(rng);
                }
            }
        }

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(twistData->basePtr(), m_twistVec.data(), twistBufSize, cudaMemcpyHostToDevice));
    }

    nvcv::Tensor         m_twistTensor;
    long2                m_twistStrides;
    std::vector<uint8_t> m_twistVec;
};

#define NVCV_SHAPE(w, h, n) (int3{w, h, n})

#define NVCV_TEST_ROW(SrcDstShape, ValueType, ImgFormat, PerSampleArgs, ArgHelper) \
    ttype::Types<ttype::Value<SrcDstShape>, ValueType, ttype::Value<ImgFormat>, ttype::Value<PerSampleArgs>, ArgHelper>

#define NVCV_IMAGE_FORMAT_RGB16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGB16S NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGBA16S \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)
#define NVCV_IMAGE_FORMAT_RGB32U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X32_Y32_Z32)
#define NVCV_IMAGE_FORMAT_RGBA32U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X32_Y32_Z32_W32)
#define NVCV_IMAGE_FORMAT_RGB32S NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, SIGNED, XYZ1, ASSOCIATED, X32_Y32_Z32)

NVCV_TYPED_TEST_SUITE(
    OpColorTwist,
    ttype::Types<
        NVCV_TEST_ROW(NVCV_SHAPE(42, 60, 6), float3, NVCV_IMAGE_FORMAT_RGBf32, true, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(42, 60, 6), float4, NVCV_IMAGE_FORMAT_RGBAf32, true, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(41, 59, 1), float3, NVCV_IMAGE_FORMAT_RGBf32, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(349, 31, 2), uchar3, NVCV_IMAGE_FORMAT_RGB8, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(349, 31, 2), uchar4, NVCV_IMAGE_FORMAT_RGBA8, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(128, 32, 1), uchar3, NVCV_IMAGE_FORMAT_RGB8, true, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(79, 50, 3), ushort3, NVCV_IMAGE_FORMAT_RGB16U, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(101, 32, 5), short3, NVCV_IMAGE_FORMAT_RGB16S, true, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(101, 32, 5), short4, NVCV_IMAGE_FORMAT_RGBA16S, true, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(79, 50, 3), uint3, NVCV_IMAGE_FORMAT_RGB32U, true, TwistMatrixArgument<double4>),
        NVCV_TEST_ROW(NVCV_SHAPE(79, 50, 3), uint4, NVCV_IMAGE_FORMAT_RGBA32U, true, TwistMatrixArgument<double4>),
        NVCV_TEST_ROW(NVCV_SHAPE(101, 32, 5), int3, NVCV_IMAGE_FORMAT_RGB32S, false, TwistMatrixArgument<double4>)>);

TYPED_TEST(OpColorTwist, correct_output)
{
    const int3 shape = ttype::GetValue<TypeParam, 0>;
    using ValueType  = ttype::GetType<TypeParam, 1>;
    using BT         = cuda::BaseType<ValueType>;

    const nvcv::ImageFormat imgFormat{ttype::GetValue<TypeParam, 2>};
    const int               numChannels = cuda::NumElements<ValueType>;
    static_assert(numChannels == 3 || numChannels == 4);

    const bool usePerSampleArgs = ttype::GetValue<TypeParam, 3>;
    using ArgHelper             = ttype::GetType<TypeParam, 4>;
    using TwistValueType        = typename ArgHelper::TwistValueType;

    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, imgFormat);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, imgFormat);

    auto srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);
    int   numSamples = srcAccess->numSamples();
    long3 strides{srcAccess->sampleStride(), srcAccess->rowStride(), srcAccess->colStride()};
    // if tensor contains multiple samples, make sure x contains sample stride
    strides.x = (srcData->rank() == 3) ? srcAccess->numRows() * srcAccess->rowStride() : strides.x;

    size_t               bufSize = strides.x * numSamples;
    std::vector<uint8_t> srcVec(bufSize, uint8_t{0});
    std::vector<uint8_t> dstVec(bufSize, uint8_t{0});
    std::vector<uint8_t> refVec(bufSize, uint8_t{0});

    uniform_distribution<BT> rand(BT{0}, std::is_integral_v<BT> ? cuda::TypeTraits<BT>::max : BT{1});
    std::mt19937_64          rng(12345);

    for (int z = 0; z < shape.z; ++z)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                for (int k = 0; k < numChannels; ++k)
                {
                    cuda::GetElement(test::ValueAt<ValueType>(srcVec, strides, int3{x, y, z}), k) = rand(rng);
                }
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), bufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    ArgHelper arg;
    arg.populate(rng, usePerSampleArgs, numSamples);

    cvcuda::ColorTwist op;
    ASSERT_NO_THROW(op(stream, srcTensor, dstTensor, arg.m_twistTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstVec.data(), dstData->basePtr(), bufSize, cudaMemcpyDeviceToHost));

    ColorTwist<ValueType, TwistValueType>(srcVec, refVec, arg.m_twistVec, strides, arg.m_twistStrides, shape,
                                          usePerSampleArgs);

    float absTolerance = std::is_integral_v<BT> ? 1 : 1e-5;
    CompareTensors<ValueType>(dstVec, refVec, strides, shape, absTolerance);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TYPED_TEST(OpColorTwist, varshape_correct_output)
{
    const int3 shape = ttype::GetValue<TypeParam, 0>;
    using ValueType  = ttype::GetType<TypeParam, 1>;
    using BT         = cuda::BaseType<ValueType>;

    const nvcv::ImageFormat imgFormat{ttype::GetValue<TypeParam, 2>};
    const int               numChannels = cuda::NumElements<ValueType>;
    static_assert(numChannels == 3 || numChannels == 4);

    constexpr bool usePerSampleArgs = ttype::GetValue<TypeParam, 3>;
    using ArgHelper                 = ttype::GetType<TypeParam, 4>;
    using TwistValueType            = typename ArgHelper::TwistValueType;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image>          imgSrc;
    std::vector<nvcv::Image>          imgDst;
    std::vector<std::vector<uint8_t>> srcVec(shape.z);

    std::uniform_int_distribution<int> randW(shape.x * 0.5, shape.x * 1.5);
    std::uniform_int_distribution<int> randH(shape.y * 0.5, shape.y * 1.5);
    uniform_distribution<BT>           rand(BT{0}, std::is_integral_v<BT> ? cuda::TypeTraits<BT>::max : BT{1});
    std::mt19937_64                    rng(12345);

    ASSERT_EQ(sizeof(ValueType), imgFormat.planePixelStrideBytes(0));

    for (int z = 0; z < shape.z; ++z)
    {
        nvcv::Size2D imgShape{randW(rng), randH(rng)};
        imgSrc.emplace_back(imgShape, imgFormat);
        imgDst.emplace_back(imgShape, imgFormat);

        auto imgData = imgSrc[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        int   srcRowStride = imgData->plane(0).rowStride;
        long2 srcStrides   = long2{srcRowStride, sizeof(ValueType)};

        srcVec[z].resize(srcRowStride * imgSrc[z].size().h);

        for (int y = 0; y < imgSrc[z].size().h; ++y)
            for (int x = 0; x < imgSrc[z].size().w; ++x)
                for (int k = 0; k < numChannels; ++k)
                    cuda::GetElement(test::ValueAt<ValueType>(srcVec[z], srcStrides, int2{x, y}), k) = rand(rng);

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, srcRowStride, srcVec[z].data(), srcRowStride,
                                    srcRowStride, imgSrc[z].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(shape.z);
    nvcv::ImageBatchVarShape batchDst(shape.z);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    ArgHelper arg;
    arg.populate(rng, usePerSampleArgs, shape.z);

    cvcuda::ColorTwist op;
    ASSERT_NO_THROW(op(stream, batchSrc, batchDst, arg.m_twistTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int z = 0; z < shape.z; z++)
    {
        SCOPED_TRACE(z);

        const auto srcData = imgSrc[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        long3 sampleStrides{0, srcData->plane(0).rowStride, sizeof(ValueType)};
        int3  sampleShape{srcData->plane(0).width, srcData->plane(0).height, 1};

        std::vector<uint8_t> dstVec(sampleShape.y * sampleStrides.y);
        std::vector<uint8_t> refVec(sampleShape.y * sampleStrides.y);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dstVec.data(), sampleStrides.y, dstData->plane(0).basePtr, sampleStrides.y,
                                            sampleStrides.y, sampleShape.y, cudaMemcpyDeviceToHost));

        int  twistZ    = !usePerSampleArgs ? 0 : z;
        auto twistZbeg = arg.m_twistVec.begin() + twistZ * arg.m_twistStrides.x;
        auto twistZend = (!usePerSampleArgs || (twistZ == shape.z - 1))
                           ? arg.m_twistVec.end()
                           : arg.m_twistVec.begin() + (twistZ + 1) * arg.m_twistStrides.x;

        std::vector<uint8_t> twistVec(twistZbeg, twistZend);

        ColorTwist<ValueType, TwistValueType>(srcVec[z], refVec, twistVec, sampleStrides, arg.m_twistStrides,
                                              sampleShape, usePerSampleArgs);

        float absTolerance = std::is_integral_v<BT> ? 1 : 1e-5;
        CompareTensors<ValueType>(dstVec, refVec, sampleStrides, sampleShape, absTolerance);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
