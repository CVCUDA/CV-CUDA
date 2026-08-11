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
#include <cvcuda/OpColorTwist.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

template<typename T, int N, int M>
using Mat = cuda::math::Matrix<T, N, M>;

template<typename T>
using uniform_distribution
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

template<typename ValueType, typename TwistT>
cuda::math::Vector<TwistT, 4> LoadColorTwistInput(ValueType pixel)
{
    cuda::math::Vector<TwistT, 4> in;
    for (int k = 0; k < 3; k++)
    {
        in[k] = cuda::GetElement(pixel, k);
    }
    in[3] = 1.;
    return in;
}

template<typename ValueType, typename TwistT>
void StoreColorTwistOutput(std::vector<uint8_t> &dst, const long3 &strides, const int3 &coord, ValueType pixel,
                           const cuda::math::Vector<TwistT, 3> &out)
{
    using BT                  = cuda::BaseType<ValueType>;
    constexpr int numChannels = cuda::NumElements<ValueType>;

    ValueType &dstPixel = test::ValueAt<ValueType>(dst, strides, coord);
    for (int k = 0; k < 3; k++)
    {
        cuda::GetElement(dstPixel, k) = cuda::SaturateCast<BT>(out[k]);
    }
    for (int k = 3; k < numChannels; k++)
    {
        cuda::GetElement(dstPixel, k) = cuda::GetElement(pixel, k);
    }
}

template<typename TwistValueType>
Mat<cuda::BaseType<TwistValueType>, 3, 4> LoadTwistMatrix(std::vector<uint8_t> &twist, const long2 &twistStrides,
                                                          int twistZ)
{
    using TwistT = cuda::BaseType<TwistValueType>;

    Mat<TwistT, 3, 4> mix;
    for (int i = 0; i < 3; i++)
    {
        auto row = test::ValueAt<TwistValueType>(twist, twistStrides, int2{i, twistZ});
        for (int j = 0; j < 4; j++)
        {
            mix[i][j] = cuda::GetElement(row, j);
        }
    }
    return mix;
}

template<typename ValueType>
void CompareTensorPixel(std::vector<uint8_t> &dst, std::vector<uint8_t> &ref, const long3 &strides, const int3 &coord,
                        float tolerance)
{
    for (int k = 0; k < cuda::NumElements<ValueType>; ++k)
    {
        auto val     = cuda::GetElement(test::ValueAt<ValueType>(dst, strides, coord), k);
        auto ref_val = cuda::GetElement(test::ValueAt<ValueType>(ref, strides, coord), k);
        EXPECT_NEAR(val, ref_val, tolerance);
    }
}

template<typename ValueType, typename Distribution, typename Rng>
void FillRandomTensorPixel(std::vector<uint8_t> &src, const long3 &strides, const int3 &coord, int numChannels,
                           Distribution &rand, Rng &rng)
{
    ValueType &pixel = test::ValueAt<ValueType>(src, strides, coord);
    for (int k = 0; k < numChannels; ++k)
    {
        cuda::GetElement(pixel, k) = rand(rng);
    }
}

template<typename ValueType, typename Distribution, typename Rng>
void FillRandomImage(std::vector<uint8_t> &src, const long2 &strides, int2 shape, int numChannels, Distribution &rand,
                     Rng &rng)
{
    for (int y = 0; y < shape.y; ++y)
    {
        for (int x = 0; x < shape.x; ++x)
        {
            auto &pixel = test::ValueAt<ValueType>(src, strides, int2{x, y});
            for (int k = 0; k < numChannels; ++k)
            {
                cuda::GetElement(pixel, k) = rand(rng);
            }
        }
    }
}

template<typename TwistValueType, typename Distribution, typename Rng>
void FillTwistSample(std::vector<uint8_t> &twist, const long2 &twistStrides, int y, int numRows, int numCols,
                     Distribution &coeffDist, Rng &rng)
{
    for (int x = 0; x < numRows; ++x)
    {
        auto &row = test::ValueAt<TwistValueType>(twist, twistStrides, int2{x, y});
        for (int k = 0; k < numCols; ++k)
        {
            cuda::GetElement(row, k) = coeffDist(rng);
        }
    }
}

template<typename ValueType, typename TwistValueType>
void ColorTwist(std::vector<uint8_t> &src, std::vector<uint8_t> &dst, std::vector<uint8_t> &twist, const long3 &strides,
                const long2 &twistStrides, const int3 &shape, bool usePerSampleTwist)
{
    using TwistT              = cuda::BaseType<TwistValueType>;
    constexpr int numChannels = cuda::NumElements<ValueType>;
    static_assert(numChannels == 3 || numChannels == 4);

    for (int z = 0; z < shape.z; ++z)
    {
        int               twistZ = usePerSampleTwist ? z : 0;
        Mat<TwistT, 3, 4> mix    = LoadTwistMatrix<TwistValueType>(twist, twistStrides, twistZ);

        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                int3 coord{x, y, z};
                auto pixel = test::ValueAt<ValueType>(src, strides, coord);

                cuda::math::Vector<TwistT, 3> out = mix * LoadColorTwistInput<ValueType, TwistT>(pixel);
                StoreColorTwistOutput(dst, strides, coord, pixel, out);
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
                CompareTensorPixel<ValueType>(dst, ref, strides, int3{x, y, z}, tolerance);
            }
        }
    }
}

template<typename ValueType, typename TwistValueType, typename ArgHelper>
void RunColorTwistTensorPlanarParity(const int3 &shape, nvcv::ImageFormat interleavedFmt, nvcv::ImageFormat planarFmt,
                                     bool usePerSampleArgs)
{
    using BT = cuda::BaseType<ValueType>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int numChannels = cuda::NumElements<ValueType>;
    static_assert(numChannels == 3 || numChannels == 4);
    ASSERT_EQ(planarFmt.numChannels(), numChannels);

    const int elemSize     = sizeof(BT);
    const int srcRowStride = shape.x * sizeof(ValueType);
    const int srcSampleStr = shape.y * srcRowStride;

    nvcv::Tensor srcInterleaved = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, interleavedFmt);
    nvcv::Tensor dstInterleaved = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, interleavedFmt);
    nvcv::Tensor srcPlanar      = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, planarFmt);
    nvcv::Tensor dstPlanar      = nvcv::util::CreateTensor(shape.z, shape.x, shape.y, planarFmt);

    auto srcIData = srcInterleaved.exportData<nvcv::TensorDataStridedCuda>();
    auto dstIData = dstInterleaved.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPData = srcPlanar.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData = dstPlanar.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcIData && dstIData && srcPData && dstPData);

    auto srcIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcIData);
    auto dstIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstIData);
    auto srcPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPData);
    auto dstPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPData);
    ASSERT_TRUE(srcIAcc && dstIAcc && srcPAcc && dstPAcc);

    uniform_distribution<BT> rand(BT{0}, std::is_integral_v<BT> ? cuda::TypeTraits<BT>::max : BT{1});
    std::mt19937_64          rng(12345);
    const long3              hostStrides{srcSampleStr, srcRowStride, sizeof(ValueType)};

    for (int z = 0; z < shape.z; ++z)
    {
        std::vector<uint8_t> hwc(srcSampleStr, uint8_t{0});
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                FillRandomTensorPixel<ValueType>(hwc, hostStrides, int3{x, y, 0}, numChannels, rand, rng);
            }
        }

        test::planar::UploadInterleavedSample(*srcIAcc, z, hwc, shape.x, shape.y, srcRowStride);
        test::planar::UploadPlanarSample(
            *srcPAcc, z, test::planar::DeinterleaveToPlanes(hwc, shape.x, shape.y, numChannels, elemSize), shape.x,
            shape.y, numChannels, elemSize);
    }

    ArgHelper arg;
    arg.populate(rng, usePerSampleArgs, shape.z);

    cvcuda::ColorTwist op;
    ASSERT_NO_THROW(op(stream, srcInterleaved, dstInterleaved, arg.m_twistTensor));
    ASSERT_NO_THROW(op(stream, srcPlanar, dstPlanar, arg.m_twistTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int z = 0; z < shape.z; ++z)
    {
        SCOPED_TRACE(z);
        auto interleavedOut = test::planar::DownloadInterleavedSample(*dstIAcc, z, shape.x, shape.y, srcRowStride);
        auto planarOut      = test::planar::DownloadPlanarSample(*dstPAcc, z, shape.x, shape.y, numChannels, elemSize);
        auto planarAsHwc    = test::planar::InterleaveFromPlanes(planarOut, shape.x, shape.y, numChannels, elemSize);
        EXPECT_EQ(interleavedOut, planarAsHwc);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename ValueType, typename TwistValueType, typename ArgHelper>
void RunColorTwistVarShapePlanarParity(const int3 &shape, nvcv::ImageFormat interleavedFmt, nvcv::ImageFormat planarFmt,
                                       bool usePerSampleArgs)
{
    using BT = cuda::BaseType<ValueType>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int numChannels = cuda::NumElements<ValueType>;
    static_assert(numChannels == 3 || numChannels == 4);
    ASSERT_EQ(planarFmt.numChannels(), numChannels);

    const int elemSize = sizeof(BT);

    std::vector<nvcv::Image>          imgSrcI;
    std::vector<nvcv::Image>          imgDstI;
    std::vector<nvcv::Image>          imgSrcP;
    std::vector<nvcv::Image>          imgDstP;
    std::vector<std::vector<uint8_t>> srcHwc(shape.z);
    std::vector<nvcv::Size2D>         sampleSizes(shape.z);

    std::uniform_int_distribution randW(ScaledSize(shape.x, 0.5), ScaledSize(shape.x, 1.5));
    std::uniform_int_distribution randH(ScaledSize(shape.y, 0.5), ScaledSize(shape.y, 1.5));
    uniform_distribution<BT>      rand(BT{0}, std::is_integral_v<BT> ? cuda::TypeTraits<BT>::max : BT{1});
    std::mt19937_64               rng(12345);

    for (int z = 0; z < shape.z; ++z)
    {
        nvcv::Size2D imgShape{randW(rng), randH(rng)};
        sampleSizes[z] = imgShape;
        imgSrcI.emplace_back(imgShape, interleavedFmt);
        imgDstI.emplace_back(imgShape, interleavedFmt);
        imgSrcP.emplace_back(imgShape, planarFmt);
        imgDstP.emplace_back(imgShape, planarFmt);

        const int rowStride = imgShape.w * sizeof(ValueType);
        srcHwc[z].resize(rowStride * imgShape.h);
        FillRandomImage<ValueType>(srcHwc[z], long2{rowStride, sizeof(ValueType)}, int2{imgShape.w, imgShape.h},
                                   numChannels, rand, rng);

        auto srcIData = imgSrcI[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(srcIData, nvcv::NullOpt);
        ASSERT_EQ(srcIData->numPlanes(), 1);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(srcIData->plane(0).basePtr, srcIData->plane(0).rowStride, srcHwc[z].data(),
                                    rowStride, rowStride, imgShape.h, cudaMemcpyHostToDevice, stream));

        auto      planes = test::planar::DeinterleaveToPlanes(srcHwc[z], imgShape.w, imgShape.h, numChannels, elemSize);
        auto      srcPData   = imgSrcP[z].exportData<nvcv::ImageDataStridedCuda>();
        const int planeBytes = imgShape.w * imgShape.h * elemSize;
        ASSERT_NE(srcPData, nvcv::NullOpt);
        ASSERT_EQ(srcPData->numPlanes(), numChannels);
        for (int c = 0; c < numChannels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2DAsync(srcPData->plane(c).basePtr, srcPData->plane(c).rowStride,
                                        planes.data() + c * planeBytes, imgShape.w * elemSize, imgShape.w * elemSize,
                                        imgShape.h, cudaMemcpyHostToDevice, stream));
        }
    }

    nvcv::ImageBatchVarShape batchSrcI(shape.z);
    nvcv::ImageBatchVarShape batchDstI(shape.z);
    nvcv::ImageBatchVarShape batchSrcP(shape.z);
    nvcv::ImageBatchVarShape batchDstP(shape.z);
    batchSrcI.pushBack(imgSrcI.begin(), imgSrcI.end());
    batchDstI.pushBack(imgDstI.begin(), imgDstI.end());
    batchSrcP.pushBack(imgSrcP.begin(), imgSrcP.end());
    batchDstP.pushBack(imgDstP.begin(), imgDstP.end());

    ArgHelper arg;
    arg.populate(rng, usePerSampleArgs, shape.z);

    cvcuda::ColorTwist op;
    ASSERT_NO_THROW(op(stream, batchSrcI, batchDstI, arg.m_twistTensor));
    ASSERT_NO_THROW(op(stream, batchSrcP, batchDstP, arg.m_twistTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int z = 0; z < shape.z; ++z)
    {
        SCOPED_TRACE(z);
        const auto imgShape  = sampleSizes[z];
        const int  rowStride = imgShape.w * sizeof(ValueType);

        std::vector<uint8_t> interleavedOut(rowStride * imgShape.h);
        auto                 dstIData = imgDstI[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(dstIData, nvcv::NullOpt);
        EXPECT_EQ(cudaSuccess,
                  cudaMemcpy2D(interleavedOut.data(), rowStride, dstIData->plane(0).basePtr,
                               dstIData->plane(0).rowStride, rowStride, imgShape.h, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> planarOut(rowStride * imgShape.h);
        auto                 dstPData   = imgDstP[z].exportData<nvcv::ImageDataStridedCuda>();
        const int            planeBytes = imgShape.w * imgShape.h * elemSize;
        ASSERT_NE(dstPData, nvcv::NullOpt);
        for (int c = 0; c < numChannels; ++c)
        {
            EXPECT_EQ(cudaSuccess, cudaMemcpy2D(planarOut.data() + c * planeBytes, imgShape.w * elemSize,
                                                dstPData->plane(c).basePtr, dstPData->plane(c).rowStride,
                                                imgShape.w * elemSize, imgShape.h, cudaMemcpyDeviceToHost));
        }

        auto planarAsHwc = test::planar::InterleaveFromPlanes(planarOut, imgShape.w, imgShape.h, numChannels, elemSize);
        EXPECT_EQ(interleavedOut, planarAsHwc);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename TwistValueType>
inline nvcv::Tensor GetTwistTensor(bool usePerSampleArgs, int numSamples)
{
    static_assert(std::is_same_v<float4, TwistValueType> || std::is_same_v<double4_16a, TwistValueType>);
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

        std::uniform_real_distribution<float> coeffDist(-10.f, 10.f);
        for (int y = 0; y < numArgs; ++y)
        {
            FillTwistSample<TwistValueType>(m_twistVec, m_twistStrides, y, numRows, numCols, coeffDist, rng);
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

#define NVCV_PLANAR_TEST_ROW(SrcDstShape, ValueType, InterleavedFmt, PlanarFmt, PerSampleArgs, ArgHelper)     \
    ttype::Types<ttype::Value<SrcDstShape>, ValueType, ttype::Value<InterleavedFmt>, ttype::Value<PlanarFmt>, \
                 ttype::Value<PerSampleArgs>, ArgHelper>

#define NVCV_IMAGE_FORMAT_RGB16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGBA16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)
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
        NVCV_TEST_ROW(NVCV_SHAPE(55, 27, 1), float4, NVCV_IMAGE_FORMAT_RGBAf32, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(349, 31, 2), uchar3, NVCV_IMAGE_FORMAT_RGB8, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(349, 31, 2), uchar4, NVCV_IMAGE_FORMAT_RGBA8, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(128, 32, 1), uchar3, NVCV_IMAGE_FORMAT_RGB8, true, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(79, 50, 3), ushort3, NVCV_IMAGE_FORMAT_RGB16U, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(88, 57, 3), ushort4, NVCV_IMAGE_FORMAT_RGBA16U, false, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(101, 32, 5), short3, NVCV_IMAGE_FORMAT_RGB16S, true, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(101, 32, 5), short4, NVCV_IMAGE_FORMAT_RGBA16S, true, TwistMatrixArgument<float4>),
        NVCV_TEST_ROW(NVCV_SHAPE(79, 50, 3), uint3, NVCV_IMAGE_FORMAT_RGB32U, true, TwistMatrixArgument<double4_16a>),
        NVCV_TEST_ROW(NVCV_SHAPE(79, 50, 3), uint4, NVCV_IMAGE_FORMAT_RGBA32U, true, TwistMatrixArgument<double4_16a>),
        NVCV_TEST_ROW(NVCV_SHAPE(101, 32, 5), int3, NVCV_IMAGE_FORMAT_RGB32S, false,
                      TwistMatrixArgument<double4_16a>)>);

NVCV_TYPED_TEST_SUITE(
    OpColorTwistPlanarTensor,
    ttype::Types<NVCV_PLANAR_TEST_ROW(NVCV_SHAPE(42, 60, 3), uchar3, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p,
                                      true, TwistMatrixArgument<float4>),
                 NVCV_PLANAR_TEST_ROW(NVCV_SHAPE(35, 37, 2), uchar4, NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_RGBA8p,
                                      false, TwistMatrixArgument<float4>),
                 NVCV_PLANAR_TEST_ROW(NVCV_SHAPE(31, 29, 2), float3, NVCV_IMAGE_FORMAT_RGBf32,
                                      NVCV_IMAGE_FORMAT_RGBf32p, true, TwistMatrixArgument<float4>),
                 NVCV_PLANAR_TEST_ROW(NVCV_SHAPE(17, 19, 1), float4, NVCV_IMAGE_FORMAT_RGBAf32,
                                      NVCV_IMAGE_FORMAT_RGBAf32p, false, TwistMatrixArgument<float4>)>);

NVCV_TYPED_TEST_SUITE(
    OpColorTwistPlanarVarShape,
    ttype::Types<NVCV_PLANAR_TEST_ROW(NVCV_SHAPE(42, 60, 3), uchar3, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p,
                                      true, TwistMatrixArgument<float4>),
                 NVCV_PLANAR_TEST_ROW(NVCV_SHAPE(31, 29, 2), float3, NVCV_IMAGE_FORMAT_RGBf32,
                                      NVCV_IMAGE_FORMAT_RGBf32p, false, TwistMatrixArgument<float4>),
                 NVCV_PLANAR_TEST_ROW(NVCV_SHAPE(17, 19, 2), float4, NVCV_IMAGE_FORMAT_RGBAf32,
                                      NVCV_IMAGE_FORMAT_RGBAf32p, true, TwistMatrixArgument<float4>)>);

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
    auto  numSamples = static_cast<int>(srcAccess->numSamples());
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
                FillRandomTensorPixel<ValueType>(srcVec, strides, int3{x, y, z}, numChannels, rand, rng);
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

    float absTolerance = std::is_integral_v<BT> ? 1.f : 1e-5f;
    CompareTensors<ValueType>(dstVec, refVec, strides, shape, absTolerance);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TYPED_TEST(OpColorTwistPlanarTensor, tensor_matches_interleaved)
{
    const int3 shape = ttype::GetValue<TypeParam, 0>;
    using ValueType  = ttype::GetType<TypeParam, 1>;

    const nvcv::ImageFormat interleavedFmt{ttype::GetValue<TypeParam, 2>};
    const nvcv::ImageFormat planarFmt{ttype::GetValue<TypeParam, 3>};
    const bool              usePerSampleArgs = ttype::GetValue<TypeParam, 4>;
    using ArgHelper                          = ttype::GetType<TypeParam, 5>;
    using TwistValueType                     = typename ArgHelper::TwistValueType;

    RunColorTwistTensorPlanarParity<ValueType, TwistValueType, ArgHelper>(shape, interleavedFmt, planarFmt,
                                                                          usePerSampleArgs);
}

TYPED_TEST(OpColorTwistPlanarVarShape, varshape_matches_interleaved)
{
    const int3 shape = ttype::GetValue<TypeParam, 0>;
    using ValueType  = ttype::GetType<TypeParam, 1>;

    const nvcv::ImageFormat interleavedFmt{ttype::GetValue<TypeParam, 2>};
    const nvcv::ImageFormat planarFmt{ttype::GetValue<TypeParam, 3>};
    const bool              usePerSampleArgs = ttype::GetValue<TypeParam, 4>;
    using ArgHelper                          = ttype::GetType<TypeParam, 5>;
    using TwistValueType                     = typename ArgHelper::TwistValueType;

    RunColorTwistVarShapePlanarParity<ValueType, TwistValueType, ArgHelper>(shape, interleavedFmt, planarFmt,
                                                                            usePerSampleArgs);
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

    std::uniform_int_distribution randW(ScaledSize(shape.x, 0.5), ScaledSize(shape.x, 1.5));
    std::uniform_int_distribution randH(ScaledSize(shape.y, 0.5), ScaledSize(shape.y, 1.5));
    uniform_distribution<BT>      rand(BT{0}, std::is_integral_v<BT> ? cuda::TypeTraits<BT>::max : BT{1});
    std::mt19937_64               rng(12345);

    ASSERT_EQ(sizeof(ValueType), imgFormat.planePixelStrideBytes(0));

    for (int z = 0; z < shape.z; ++z)
    {
        nvcv::Size2D imgShape{randW(rng), randH(rng)};
        imgSrc.emplace_back(imgShape, imgFormat);
        imgDst.emplace_back(imgShape, imgFormat);

        auto imgData = imgSrc[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        int  srcRowStride = imgData->plane(0).rowStride;
        auto srcStrides   = long2{srcRowStride, sizeof(ValueType)};

        srcVec[z].resize(srcRowStride * imgSrc[z].size().h);

        FillRandomImage<ValueType>(srcVec[z], srcStrides, int2{imgSrc[z].size().w, imgSrc[z].size().h}, numChannels,
                                   rand, rng);

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

        float absTolerance = std::is_integral_v<BT> ? 1.f : 1e-5f;
        CompareTensors<ValueType>(dstVec, refVec, sampleStrides, sampleShape, absTolerance);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpColorTwistVarshape_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int>{
    // inFmt, outFmt, inputNumImages, outputNumImages
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 4, 3},
    {nvcv::FMT_U8, nvcv::FMT_U8, 3, 3},
});

NVCV_TEST_SUITE_P(OpColorTwist_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int , int, nvcv::DataType, std::string, int, int, int>{
    // inFmt, outFmt, inputSamples, outputSamples, twistDtype, layout, twistShapeSamples, twistShapeRows, twistShapeCols
    // Invalid src/dst tensors
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 4, 3, nvcv::TYPE_4F32, "H", 3, 3, 4},
    {nvcv::FMT_RGB8, nvcv::FMT_U8, 3, 3, nvcv::TYPE_4F32, "H", 3, 3, 4},
    {nvcv::FMT_U8, nvcv::FMT_U8, 3, 3, nvcv::TYPE_4F32, "H", 3, 3, 4},
    {nvcv::FMT_RGB8, nvcv::FMT_RGBf32, 3, 3, nvcv::TYPE_4F32, "H", 3, 3, 4},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 3, 3, nvcv::TYPE_4F32, "H", 3, 3, 4},
    // Invalid twist tensor
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 3, 3, nvcv::TYPE_2F32, "H", 3, 3, 4},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 3, 3, nvcv::TYPE_4F32, "NHW", 3, 3, 4},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 3, 3, nvcv::TYPE_F32, "NHW", 4, 3, 4}, // has per sample twist
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 3, 3, nvcv::TYPE_F32, "NHW", 3, 4, 4}, // has per sample twist
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 3, 3, nvcv::TYPE_F32, "NH", 3, 3, 5},
});

// clang-format on

TEST_P(OpColorTwist_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const nvcv::ImageFormat inFmt             = GetParamValue<0>();
    const nvcv::ImageFormat outFmt            = GetParamValue<1>();
    const int               inputSamples      = GetParamValue<2>();
    const int               outputSamples     = GetParamValue<3>();
    const nvcv::DataType    twistDtype        = GetParamValue<4>();
    const std::string       layout            = GetParamValue<5>();
    const int               twistShapeSamples = GetParamValue<6>();
    const int               twistShapeRows    = GetParamValue<7>();
    const int               twistShapeCols    = GetParamValue<8>();

    const int3   inShape{32, 32, inputSamples};
    const int3   outShape{32, 32, outputSamples};
    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(inShape.z, inShape.x, inShape.y, inFmt);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(outShape.z, outShape.x, outShape.y, outFmt);

    nvcv::Tensor twistTensor;
    if (layout.size() == 3)
    {
        twistTensor = nvcv::Tensor(
            {
                {twistShapeSamples, twistShapeRows, twistShapeCols},
                layout.c_str()
        },
            twistDtype);
    }
    else if (layout.size() == 2)
    {
        twistTensor = nvcv::Tensor(
            {
                {twistShapeRows, twistShapeCols},
                layout.c_str()
        },
            twistDtype);
    }
    else if (layout.size() == 1)
    {
        twistTensor = nvcv::Tensor({{twistShapeRows}, layout.c_str()}, twistDtype);
    }

    cvcuda::ColorTwist op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &srcTensor, &dstTensor, &twistTensor]
                                                             { op(stream, srcTensor, dstTensor, twistTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpColorTwistVarshape_Negative, op)
{
    const nvcv::ImageFormat inFmt           = GetParamValue<0>();
    const nvcv::ImageFormat outFmt          = GetParamValue<1>();
    const int               inputNumImages  = GetParamValue<2>();
    const int               outputNumImages = GetParamValue<3>();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;

    std::uniform_int_distribution randW(ScaledSize(24, 0.5), ScaledSize(24, 1.5));
    std::uniform_int_distribution randH(ScaledSize(24, 0.5), ScaledSize(24, 1.5));
    std::mt19937_64               rng(12345);

    for (int i = 0; i < inputNumImages; ++i)
    {
        nvcv::Size2D imgShape{randW(rng), randH(rng)};
        imgSrc.emplace_back(imgShape, inFmt);
    }
    for (int i = 0; i < outputNumImages; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), outFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(inputNumImages);
    nvcv::ImageBatchVarShape batchDst(outputNumImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor twistTensor{
        {{3}, "H"},
        nvcv::TYPE_4F32
    };

    cvcuda::ColorTwist op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &batchSrc, &batchDst, &twistTensor]
                                                             { op(stream, batchSrc, batchDst, twistTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpColorTwistVarshape_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt            = nvcv::FMT_RGB8;
    const int         numberOfImages = 5;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_RGBA8,             fmt},
        {            fmt, nvcv::FMT_RGBA8}
    };

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        std::vector<nvcv::Image>      imgSrc;
        std::vector<nvcv::Image>      imgDst;
        std::uniform_int_distribution randW(ScaledSize(24, 0.5), ScaledSize(24, 1.5));
        std::uniform_int_distribution randH(ScaledSize(24, 0.5), ScaledSize(24, 1.5));
        std::mt19937_64               rng(12345);

        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            nvcv::Size2D imgShape{randW(rng), randH(rng)};
            imgSrc.emplace_back(imgShape, fmt);
            imgDst.emplace_back(imgShape, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        nvcv::ImageBatchVarShape batchDst(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        nvcv::Tensor twistTensor{
            {{3}, "H"},
            nvcv::TYPE_4F32
        };

        cvcuda::ColorTwist op;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &batchSrc, &batchDst, &twistTensor]
                                                                 { op(stream, batchSrc, batchDst, twistTensor); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpColorTwist_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaColorTwistCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
