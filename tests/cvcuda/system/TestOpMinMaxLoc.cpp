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

#include "Definitions.hpp"

#include <common/InterpUtils.hpp>
#include <common/TypedTests.hpp>
#include <cvcuda/OpMinMaxLoc.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;
namespace type = nvcv::test::type;

static std::default_random_engine g_rng(std::random_device{}());

template<typename T>
using uniform_distribution
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

// Auxiliary functions to get the value type (for minVal or maxVal) for the given input type

template<typename T>
using OutputValueType = std::conditional_t<
    std::is_same_v<T, char1> || std::is_same_v<T, short1> || std::is_same_v<T, int1>, int1,
    std::conditional_t<std::is_same_v<T, uchar1> || std::is_same_v<T, ushort1> || std::is_same_v<T, uint1>, uint1, T>>;

inline nvcv::DataType GetValDataType(nvcv::DataType inDataType)
{
    if (inDataType == nvcv::TYPE_S8 || inDataType == nvcv::TYPE_S16 || inDataType == nvcv::TYPE_S32)
    {
        return nvcv::TYPE_S32;
    }
    else if (inDataType == nvcv::TYPE_U8 || inDataType == nvcv::TYPE_U16 || inDataType == nvcv::TYPE_U32)
    {
        return nvcv::TYPE_U32;
    }
    else if (inDataType == nvcv::TYPE_F32 || inDataType == nvcv::TYPE_F64)
    {
        return inDataType;
    }
    return nvcv::DataType();
}

// Compute reference (gold) output of operator MinMaxLoc

template<typename InVT, typename OutVT, class InContainerType, class InStridesType, class InShapeType>
inline void FindMinMax(InContainerType &in, InStridesType &inStrides, InShapeType &inShape,
                       std::vector<uint8_t> &minVal, std::vector<uint8_t> &maxVal, long1 valStrides,
                       std::vector<std::vector<int2>> &minLoc, std::vector<std::vector<int2>> &maxLoc, int capacity,
                       std::vector<uint8_t> &numMin, std::vector<uint8_t> &numMax, long1 &numStrides)
{
    constexpr bool InIsTensor = std::is_same_v<InContainerType, std::vector<uint8_t>>;

    int numSamples;
    if constexpr (InIsTensor)
    {
        numSamples = inShape.z;
    }
    else
    {
        numSamples = in.size();
    }

    using InBT  = cuda::BaseType<InVT>;
    using OutBT = cuda::BaseType<OutVT>;

    InBT val;

    for (int z = 0; z < numSamples; ++z)
    {
        OutBT min{cuda::TypeTraits<InBT>::max}, max{cuda::Lowest<InBT>};

        int2 inSize;
        if constexpr (InIsTensor)
        {
            inSize = cuda::DropCast<2>(inShape);
        }
        else
        {
            inSize = inShape[z];
        }

        for (int y = 0; y < inSize.y; ++y)
        {
            for (int x = 0; x < inSize.x; ++x)
            {
                if constexpr (InIsTensor)
                {
                    val = test::ValueAt<InVT>(in, inStrides, int3{x, y, z}).x;
                }
                else
                {
                    val = test::ValueAt<InVT>(in[z], inStrides[z], int2{x, y}).x;
                }

                min = std::min(min, static_cast<OutBT>(val));
                max = std::max(max, static_cast<OutBT>(val));
            }
        }

        test::ValueAt<OutVT>(minVal, valStrides, {z}).x = min;
        test::ValueAt<OutVT>(maxVal, valStrides, {z}).x = max;

        int nMin{0}, nMax{0};

        for (int y = 0; y < inSize.y; ++y)
        {
            for (int x = 0; x < inSize.x; ++x)
            {
                if constexpr (InIsTensor)
                {
                    val = test::ValueAt<InVT>(in, inStrides, int3{x, y, z}).x;
                }
                else
                {
                    val = test::ValueAt<InVT>(in[z], inStrides[z], int2{x, y}).x;
                }

                if (val == min)
                {
                    if (nMin < capacity)
                    {
                        minLoc[z].push_back(int2{x, y});
                    }
                    nMin++;
                }
                if (val == max)
                {
                    if (nMax < capacity)
                    {
                        maxLoc[z].push_back(int2{x, y});
                    }
                    nMax++;
                }
            }
        }

        test::ValueAt<int1>(numMin, numStrides, {z}).x = nMin;
        test::ValueAt<int1>(numMax, numStrides, {z}).x = nMax;
    }
}

// Sort min/max locations to be able to compare test vs. gold results

inline void LocSort(std::vector<std::vector<int2>> &minLocTest, std::vector<std::vector<int2>> &maxLocTest,
                    int capacity, std::vector<uint8_t> &minLocVec, std::vector<uint8_t> &maxLocVec, long2 &locStrides,
                    std::vector<uint8_t> &numMinVec, std::vector<uint8_t> &numMaxVec, long1 &numStrides)
{
    ASSERT_EQ(minLocTest.size(), maxLocTest.size());

    auto locLower = [](int2 loc1, int2 loc2)
    {
        return loc1.y == loc2.y ? loc1.x < loc2.x : loc1.y < loc2.y;
    };

    for (int z = 0; z < (int)minLocTest.size(); z++)
    {
        int nMin = test::ValueAt<int1>(numMinVec, numStrides, {z}).x;
        int nMax = test::ValueAt<int1>(numMaxVec, numStrides, {z}).x;

        for (int i = 0; i < nMin && i < capacity; i++)
        {
            minLocTest[z].push_back(test::ValueAt<int2>(minLocVec, locStrides, {i, z}));
        }
        for (int i = 0; i < nMax && i < capacity; i++)
        {
            maxLocTest[z].push_back(test::ValueAt<int2>(maxLocVec, locStrides, {i, z}));
        }

        std::sort(minLocTest[z].begin(), minLocTest[z].end(), locLower);
        std::sort(maxLocTest[z].begin(), maxLocTest[z].end(), locLower);
    }
}

// The full gold function includes preparing buffers, copying data and computing gold results

struct MinMaxResults
{
    std::vector<uint8_t> minValTest, numMinTest, maxValTest, numMaxTest, minLocTemp, maxLocTemp;
    std::vector<uint8_t> minValGold, numMinGold, maxValGold, numMaxGold;

    std::vector<std::vector<int2>> minLocTest, maxLocTest;
    std::vector<std::vector<int2>> minLocGold, maxLocGold;
};

template<typename InVT, typename OutVT, class InContainerType, class InStridesType, class InShapeType>
inline void GoldMinMaxLoc(const nvcv::Tensor &minVal, const nvcv::Tensor &minLoc, const nvcv::Tensor &numMin,
                          const nvcv::Tensor &maxVal, const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax,
                          InContainerType &inVec, InStridesType &inStrides, InShapeType &inShape,
                          MinMaxResults &outResults)
{
    auto minValData = minVal.exportData<nvcv::TensorDataStridedCuda>();
    auto minLocData = minLoc.exportData<nvcv::TensorDataStridedCuda>();
    auto numMinData = numMin.exportData<nvcv::TensorDataStridedCuda>();
    auto maxValData = maxVal.exportData<nvcv::TensorDataStridedCuda>();
    auto maxLocData = maxLoc.exportData<nvcv::TensorDataStridedCuda>();
    auto numMaxData = numMax.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(minValData && minLocData && numMinData && maxValData && maxLocData && numMaxData);

    int    capacity   = minLocData->shape(1);
    int    numSamples = minValData->shape(0);
    long1  valStrides = {minValData->stride(0)};
    long2  locStrides = {minLocData->stride(0), minLocData->stride(1)};
    long1  numStrides = {numMinData->stride(0)};
    size_t valBufSize = numSamples * valStrides.x;
    size_t locBufSize = numSamples * locStrides.x;
    size_t numBufSize = numSamples * numStrides.x;

    outResults.minValTest.resize(valBufSize);
    outResults.numMinTest.resize(numBufSize);
    outResults.maxValTest.resize(valBufSize);
    outResults.numMaxTest.resize(numBufSize);
    outResults.minLocTemp.resize(locBufSize);
    outResults.maxLocTemp.resize(locBufSize);

#define NVCV_TEST_CUDA_COPY(FROM, TO, SIZE) \
    ASSERT_EQ(cudaSuccess, cudaMemcpy(TO.data(), FROM->basePtr(), SIZE, cudaMemcpyDeviceToHost))

    NVCV_TEST_CUDA_COPY(minValData, outResults.minValTest, valBufSize);
    NVCV_TEST_CUDA_COPY(minLocData, outResults.minLocTemp, locBufSize);
    NVCV_TEST_CUDA_COPY(numMinData, outResults.numMinTest, numBufSize);
    NVCV_TEST_CUDA_COPY(maxValData, outResults.maxValTest, valBufSize);
    NVCV_TEST_CUDA_COPY(maxLocData, outResults.maxLocTemp, locBufSize);
    NVCV_TEST_CUDA_COPY(numMaxData, outResults.numMaxTest, numBufSize);

#undef NVCV_TEST_CUDA_COPY

    outResults.minValGold.resize(valBufSize);
    outResults.maxValGold.resize(valBufSize);
    outResults.numMinGold.resize(numBufSize);
    outResults.numMaxGold.resize(numBufSize);
    outResults.minLocGold.resize(numSamples);
    outResults.maxLocGold.resize(numSamples);
    outResults.minLocTest.resize(numSamples);
    outResults.maxLocTest.resize(numSamples);

    LocSort(outResults.minLocTest, outResults.maxLocTest, capacity, outResults.minLocTemp, outResults.maxLocTemp,
            locStrides, outResults.numMinTest, outResults.numMaxTest, numStrides);

    FindMinMax<InVT, OutVT>(inVec, inStrides, inShape, outResults.minValGold, outResults.maxValGold, valStrides,
                            outResults.minLocGold, outResults.maxLocGold, capacity, outResults.numMinGold,
                            outResults.numMaxGold, numStrides);
}

// clang-format off

typedef enum { MIN = 0b01, MAX = 0b10, MIN_MAX = 0b11 } RunChoice;

#define NVCV_SHAPE(w, h, n) (int3{w, h, n})

#define NVCV_TEST_ROW(InShape, ValueType, InFormat, MaxNumLocs, MinMaxChoice)                    \
    type::Types<type::Value<InShape>, ValueType, type::Value<InFormat>, type::Value<MaxNumLocs>, \
                type::Value<MinMaxChoice>>

NVCV_TYPED_TEST_SUITE(OpMinMaxLoc, type::Types<
    NVCV_TEST_ROW(NVCV_SHAPE(44, 33, 1), uchar1, NVCV_IMAGE_FORMAT_U8, 99, RunChoice::MIN_MAX),
    NVCV_TEST_ROW(NVCV_SHAPE(43, 32, 4), ushort1, NVCV_IMAGE_FORMAT_U16, 202, RunChoice::MIN),
    NVCV_TEST_ROW(NVCV_SHAPE(42, 30, 5), int1, NVCV_IMAGE_FORMAT_S32, 320, RunChoice::MAX),
    NVCV_TEST_ROW(NVCV_SHAPE(421, 292, 2), char1, NVCV_IMAGE_FORMAT_S8, 9855, RunChoice::MIN),
    NVCV_TEST_ROW(NVCV_SHAPE(98, 39, 3), short1, NVCV_IMAGE_FORMAT_S16, 644, RunChoice::MAX),
    NVCV_TEST_ROW(NVCV_SHAPE(13, 11, 11), uint1, NVCV_IMAGE_FORMAT_U32, 166, RunChoice::MIN_MAX),
    NVCV_TEST_ROW(NVCV_SHAPE(41, 20, 6), float1, NVCV_IMAGE_FORMAT_F32, 330, RunChoice::MAX),
    NVCV_TEST_ROW(NVCV_SHAPE(40, 19, 7), double1, NVCV_IMAGE_FORMAT_F64, 240, RunChoice::MIN),
    NVCV_TEST_ROW(NVCV_SHAPE(39, 18, 8), float1, NVCV_IMAGE_FORMAT_F32, 150, RunChoice::MIN_MAX),
    NVCV_TEST_ROW(NVCV_SHAPE(38, 17, 9), double1, NVCV_IMAGE_FORMAT_F64, 260, RunChoice::MIN_MAX)
>);

// clang-format on

TYPED_TEST(OpMinMaxLoc, tensor_correct_output)
{
    int3 inShape = type::GetValue<TypeParam, 0>;

    using InVT  = type::GetType<TypeParam, 1>;
    using InBT  = cuda::BaseType<InVT>;
    using OutVT = OutputValueType<InVT>;

    nvcv::ImageFormat inFormat{type::GetValue<TypeParam, 2>};

    int       capacity = type::GetValue<TypeParam, 3>;
    RunChoice run      = type::GetValue<TypeParam, 4>;

    nvcv::DataType inDataType  = inFormat.planeDataType(0);
    nvcv::DataType valDataType = GetValDataType(inDataType);
    ASSERT_EQ(inFormat.numPlanes(), 1);
    ASSERT_EQ(inDataType.numChannels(), 1);

    nvcv::Tensor in = nvcv::util::CreateTensor(inShape.z, inShape.x, inShape.y, inFormat);

    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inData);
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    long3 inStrides{inAccess->sampleStride(), inAccess->rowStride(), inAccess->colStride()};
    inStrides.x = (inData->rank() == 3) ? inAccess->numRows() * inAccess->rowStride() : inStrides.x;

    uniform_distribution<InBT> rg(std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::min : InBT{0},
                                  std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::max : InBT{1});

    size_t inBufSize = inStrides.x * inAccess->numSamples();

    std::vector<uint8_t> inVec(inBufSize, uint8_t{0});

    for (int z = 0; z < inShape.z; ++z)
        for (int y = 0; y < inShape.y; ++y)
            for (int x = 0; x < inShape.x; ++x) test::ValueAt<InVT>(inVec, inStrides, int3{x, y, z}).x = rg(g_rng);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // clang-format off

    nvcv::Tensor minVal({{inShape.z}, "N"}, valDataType);
    nvcv::Tensor minLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{inShape.z}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor maxVal({{inShape.z}, "N"}, valDataType);
    nvcv::Tensor maxLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMax({{inShape.z}, "N"}, nvcv::TYPE_S32);

    // clang-format on

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinMaxLoc op;
    switch (run)
    {
    case RunChoice::MIN:
        EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, nullptr, nullptr, nullptr));
        break;

    case RunChoice::MAX:
        EXPECT_NO_THROW(op(stream, in, nullptr, nullptr, nullptr, maxVal, maxLoc, numMax));
        break;

    case RunChoice::MIN_MAX:
        EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, maxVal, maxLoc, numMax));
        break;
    };

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    MinMaxResults res;

    GoldMinMaxLoc<InVT, OutVT>(minVal, minLoc, numMin, maxVal, maxLoc, numMax, inVec, inStrides, inShape, res);

    if (run & RunChoice::MIN)
    {
        EXPECT_EQ(res.minValTest, res.minValGold);
        EXPECT_EQ(res.minLocTest, res.minLocGold);
        EXPECT_EQ(res.numMinTest, res.numMinGold);
    }
    if (run & RunChoice::MAX)
    {
        EXPECT_EQ(res.maxValTest, res.maxValGold);
        EXPECT_EQ(res.maxLocTest, res.maxLocGold);
        EXPECT_EQ(res.numMaxTest, res.numMaxGold);
    }
}

TYPED_TEST(OpMinMaxLoc, varshape_correct_output)
{
    int3 inShape = type::GetValue<TypeParam, 0>;

    using InVT  = type::GetType<TypeParam, 1>;
    using InBT  = cuda::BaseType<InVT>;
    using OutVT = OutputValueType<InVT>;

    nvcv::ImageFormat inFormat{type::GetValue<TypeParam, 2>};

    int       capacity = type::GetValue<TypeParam, 3>;
    RunChoice run      = type::GetValue<TypeParam, 4>;

    nvcv::DataType inDataType  = inFormat.planeDataType(0);
    nvcv::DataType valDataType = GetValDataType(inDataType);
    ASSERT_EQ(inFormat.numPlanes(), 1);
    ASSERT_EQ(inDataType.numChannels(), 1);

    std::vector<nvcv::Image> inImg;

    std::vector<std::vector<uint8_t>> inVec(inShape.z);

    std::vector<long2> inStrides(inShape.z);
    std::vector<int2>  inShape2(inShape.z);

    uniform_distribution<InBT> rg(std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::min : InBT{0},
                                  std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::max : InBT{1});

    std::uniform_int_distribution<int> rgW(inShape.x * 0.8, inShape.x * 1.2);
    std::uniform_int_distribution<int> rgH(inShape.y * 0.8, inShape.y * 1.2);

    for (int z = 0; z < inShape.z; ++z)
    {
        inImg.emplace_back(nvcv::Size2D{rgW(g_rng), rgH(g_rng)}, inFormat);

        auto inData = inImg[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(inData);

        inStrides[z] = long2{inData->plane(0).rowStride, sizeof(InVT)};
        inShape2[z]  = int2{inData->plane(0).width, inData->plane(0).height};

        inVec[z].resize(inStrides[z].x * inShape2[z].y);

        for (int y = 0; y < inShape2[z].y; ++y)
            for (int x = 0; x < inShape2[z].x; ++x)
                test::ValueAt<InVT>(inVec[z], inStrides[z], int2{x, y}).x = rg(g_rng);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inData->plane(0).basePtr, inStrides[z].x, inVec[z].data(), inStrides[z].x,
                                            inStrides[z].x, inImg[z].size().h, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape in(inShape.z);
    in.pushBack(inImg.begin(), inImg.end());

    // clang-format off

    nvcv::Tensor minVal({{inShape.z}, "N"}, valDataType);
    nvcv::Tensor minLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{inShape.z}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor maxVal({{inShape.z}, "N"}, valDataType);
    nvcv::Tensor maxLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMax({{inShape.z}, "N"}, nvcv::TYPE_S32);

    // clang-format on

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinMaxLoc op;
    switch (run)
    {
    case RunChoice::MIN:
        EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, nullptr, nullptr, nullptr));
        break;

    case RunChoice::MAX:
        EXPECT_NO_THROW(op(stream, in, nullptr, nullptr, nullptr, maxVal, maxLoc, numMax));
        break;

    case RunChoice::MIN_MAX:
        EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, maxVal, maxLoc, numMax));
        break;
    };

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    MinMaxResults res;

    GoldMinMaxLoc<InVT, OutVT>(minVal, minLoc, numMin, maxVal, maxLoc, numMax, inVec, inStrides, inShape2, res);

    if (run & RunChoice::MIN)
    {
        EXPECT_EQ(res.minValTest, res.minValGold);
        EXPECT_EQ(res.minLocTest, res.minLocGold);
        EXPECT_EQ(res.numMinTest, res.numMinGold);
    }
    if (run & RunChoice::MAX)
    {
        EXPECT_EQ(res.maxValTest, res.maxValGold);
        EXPECT_EQ(res.maxLocTest, res.maxLocGold);
        EXPECT_EQ(res.numMaxTest, res.numMaxGold);
    }
}
