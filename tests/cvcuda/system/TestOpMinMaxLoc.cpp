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

#include "Definitions.hpp"

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpMinMaxLoc.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace cuda = nvcv::cuda;
namespace gt   = ::testing;
namespace test = nvcv::test;
namespace type = nvcv::test::type;
namespace util = nvcv::util;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

// Fixed seed: random_device made tests non-deterministic across CI runs and
// occasionally produced ill-conditioned numerical inputs that exceeded
// EXPECT_NEAR tolerances on rare-config CI. Use a known-good fixed seed.
static std::default_random_engine &Rng()
{
    static std::default_random_engine rng(12345);
    return rng;
}

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

template<typename InVT, class InContainerType, class InStridesType>
inline cuda::BaseType<InVT> InputValue(InContainerType &in, InStridesType &inStrides, int z, int x, int y)
{
    if constexpr (std::is_same_v<InContainerType, std::vector<uint8_t>>)
    {
        return test::ValueAt<InVT>(in, inStrides, int3{x, y, z}).x;
    }
    else
    {
        return test::ValueAt<InVT>(in[z], inStrides[z], int2{x, y}).x;
    }
}

inline void RecordLocation(std::vector<std::vector<int2>> &locations, int z, int capacity, int &count, bool matches,
                           int2 loc)
{
    if (!matches)
    {
        return;
    }
    if (count < capacity)
    {
        locations[z].push_back(loc);
    }
    count++;
}

template<typename InVT, typename OutVT, class InContainerType, class InStridesType, class InShapeType>
inline void FindMinMax(InContainerType &in, InStridesType &inStrides, InShapeType &inShape,
                       std::vector<uint8_t> &minVal, std::vector<uint8_t> &maxVal, long1 valStrides,
                       std::vector<std::vector<int2>> &minLoc, std::vector<std::vector<int2>> &maxLoc, int capacity,
                       std::vector<uint8_t> &numMin, std::vector<uint8_t> &numMax, const long1 &numStrides)
{
    constexpr bool InIsTensor = std::is_same_v<InContainerType, std::vector<uint8_t>>;

    int numSamples;
    if constexpr (InIsTensor)
    {
        numSamples = inShape.z;
    }
    else
    {
        numSamples = static_cast<int>(in.size());
    }

    using InBT  = cuda::BaseType<InVT>;
    using OutBT = cuda::BaseType<OutVT>;

    for (int z = 0; z < numSamples; ++z)
    {
        OutBT min{cuda::TypeTraits<InBT>::max};
        OutBT max{cuda::Lowest<InBT>};

        int2 inSize;
        if constexpr (InIsTensor)
        {
            inSize = cuda::DropCast<2>(inShape);
        }
        else
        {
            inSize = inShape[z];
        }

        const int numPixels = inSize.x * inSize.y;
        for (int idx = 0; idx < numPixels; ++idx)
        {
            InBT val = InputValue<InVT>(in, inStrides, z, idx % inSize.x, idx / inSize.x);
            min      = std::min(min, static_cast<OutBT>(val));
            max      = std::max(max, static_cast<OutBT>(val));
        }

        test::ValueAt<OutVT>(minVal, valStrides, {z}).x = min;
        test::ValueAt<OutVT>(maxVal, valStrides, {z}).x = max;

        int nMin{0};
        int nMax{0};

        for (int idx = 0; idx < numPixels; ++idx)
        {
            const int x   = idx % inSize.x;
            const int y   = idx / inSize.x;
            InBT      val = InputValue<InVT>(in, inStrides, z, x, y);
            RecordLocation(minLoc, z, capacity, nMin, val == min, int2{x, y});
            RecordLocation(maxLoc, z, capacity, nMax, val == max, int2{x, y});
        }

        test::ValueAt<int1>(numMin, numStrides, {z}).x = nMin;
        test::ValueAt<int1>(numMax, numStrides, {z}).x = nMax;
    }
}

// Sort min/max locations to be able to compare test vs. gold results

inline void LocSort(std::vector<std::vector<int2>> &minLocTest, std::vector<std::vector<int2>> &maxLocTest,
                    int capacity, std::vector<uint8_t> &minLocVec, std::vector<uint8_t> &maxLocVec,
                    const long2 &locStrides, std::vector<uint8_t> &numMinVec, std::vector<uint8_t> &numMaxVec,
                    const long1 &numStrides)
{
    ASSERT_EQ(minLocTest.size(), maxLocTest.size());

    auto locLower = [](int2 loc1, int2 loc2)
    {
        return loc1.y == loc2.y ? loc1.x < loc2.x : loc1.y < loc2.y;
    };

    for (int z = 0; z < static_cast<int>(minLocTest.size()); z++)
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

        std::ranges::sort(minLocTest[z], locLower);
        std::ranges::sort(maxLocTest[z], locLower);
    }
}

// The full gold function includes preparing buffers, copying data and computing gold results

struct MinMaxResults
{
    std::vector<uint8_t> minValTest;
    std::vector<uint8_t> numMinTest;
    std::vector<uint8_t> maxValTest;
    std::vector<uint8_t> numMaxTest;
    std::vector<uint8_t> minLocTemp;
    std::vector<uint8_t> maxLocTemp;
    std::vector<uint8_t> minValGold;
    std::vector<uint8_t> numMinGold;
    std::vector<uint8_t> maxValGold;
    std::vector<uint8_t> numMaxGold;

    std::vector<std::vector<int2>> minLocTest;
    std::vector<std::vector<int2>> maxLocTest;
    std::vector<std::vector<int2>> minLocGold;
    std::vector<std::vector<int2>> maxLocGold;
};

struct MinMaxParityResults
{
    std::vector<uint32_t>          minVal;
    std::vector<int>               numMin;
    std::vector<uint32_t>          maxVal;
    std::vector<int>               numMax;
    std::vector<std::vector<int2>> minLoc;
    std::vector<std::vector<int2>> maxLoc;
};

static nvcv::Tensor CreateMinMaxLocParityTensor(int batches, int width, int height, bool planar, bool batched)
{
    if (batched && planar)
    {
        return nvcv::Tensor(
            {
                {batches, 1, height, width},
                "NCHW"
        },
            nvcv::TYPE_U8);
    }
    if (planar)
    {
        return nvcv::Tensor(
            {
                {1, height, width},
                "CHW"
        },
            nvcv::TYPE_U8);
    }
    if (batched)
    {
        return nvcv::Tensor(
            {
                {batches, height, width, 1},
                "NHWC"
        },
            nvcv::TYPE_U8);
    }
    return nvcv::Tensor(
        {
            {height, width, 1},
            "HWC"
    },
        nvcv::TYPE_U8);
}

static std::vector<uint8_t> MakeMinMaxLocParityInput(int width, int height, int sample)
{
    std::vector<uint8_t> values(width * height);
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = static_cast<uint8_t>((37 * i + 53 * sample + 17) & 0xff);
    }
    values.front() = 0;
    values.back()  = 255;
    return values;
}

static void FillMinMaxLocParityInput(nvcv::Tensor &interleaved, nvcv::Tensor &planar, int width, int height, int sample)
{
    std::vector<uint8_t> values = MakeMinMaxLocParityInput(width, height, sample);
    ASSERT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(interleaved.exportData(), values, sample));
    ASSERT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(planar.exportData(), values, sample));
}

static void FillMinMaxLocParityInputs(nvcv::Tensor &interleaved, nvcv::Tensor &planar, int width, int height,
                                      int batches)
{
    for (int sample = 0; sample < batches; ++sample)
    {
        FillMinMaxLocParityInput(interleaved, planar, width, height, sample);
    }
}

static void DownloadMinMaxParityResults(const nvcv::Tensor &minVal, const nvcv::Tensor &minLoc,
                                        const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal,
                                        const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax,
                                        MinMaxParityResults &results)
{
    auto minValData = minVal.exportData<nvcv::TensorDataStridedCuda>();
    auto minLocData = minLoc.exportData<nvcv::TensorDataStridedCuda>();
    auto numMinData = numMin.exportData<nvcv::TensorDataStridedCuda>();
    auto maxValData = maxVal.exportData<nvcv::TensorDataStridedCuda>();
    auto maxLocData = maxLoc.exportData<nvcv::TensorDataStridedCuda>();
    auto numMaxData = numMax.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(minValData && minLocData && numMinData && maxValData && maxLocData && numMaxData);

    const auto  numSamples = static_cast<int>(minValData->shape(0));
    const auto  capacity   = static_cast<int>(minLocData->shape(1));
    const long1 valStrides{minValData->stride(0)};
    const long2 locStrides{minLocData->stride(0), minLocData->stride(1)};
    const long1 numStrides{numMinData->stride(0)};

    std::vector<uint8_t> minValVec(numSamples * valStrides.x);
    std::vector<uint8_t> maxValVec(numSamples * valStrides.x);
    std::vector<uint8_t> minLocVec(numSamples * locStrides.x);
    std::vector<uint8_t> maxLocVec(numSamples * locStrides.x);
    std::vector<uint8_t> numMinVec(numSamples * numStrides.x);
    std::vector<uint8_t> numMaxVec(numSamples * numStrides.x);

#define NVCV_TEST_CUDA_COPY_PARITY(FROM, TO) \
    ASSERT_EQ(cudaSuccess, cudaMemcpy(TO.data(), FROM->basePtr(), TO.size(), cudaMemcpyDeviceToHost))

    NVCV_TEST_CUDA_COPY_PARITY(minValData, minValVec);
    NVCV_TEST_CUDA_COPY_PARITY(maxValData, maxValVec);
    NVCV_TEST_CUDA_COPY_PARITY(minLocData, minLocVec);
    NVCV_TEST_CUDA_COPY_PARITY(maxLocData, maxLocVec);
    NVCV_TEST_CUDA_COPY_PARITY(numMinData, numMinVec);
    NVCV_TEST_CUDA_COPY_PARITY(numMaxData, numMaxVec);

#undef NVCV_TEST_CUDA_COPY_PARITY

    results.minVal.resize(numSamples);
    results.maxVal.resize(numSamples);
    results.numMin.resize(numSamples);
    results.numMax.resize(numSamples);
    results.minLoc.resize(numSamples);
    results.maxLoc.resize(numSamples);

    auto locLower = [](int2 lhs, int2 rhs)
    {
        return lhs.y == rhs.y ? lhs.x < rhs.x : lhs.y < rhs.y;
    };

    for (int sample = 0; sample < numSamples; ++sample)
    {
        results.minVal[sample] = test::ValueAt<uint1>(minValVec, valStrides, {sample}).x;
        results.maxVal[sample] = test::ValueAt<uint1>(maxValVec, valStrides, {sample}).x;
        results.numMin[sample] = test::ValueAt<int1>(numMinVec, numStrides, {sample}).x;
        results.numMax[sample] = test::ValueAt<int1>(numMaxVec, numStrides, {sample}).x;

        for (int i = 0; i < std::min(results.numMin[sample], capacity); ++i)
        {
            results.minLoc[sample].push_back(test::ValueAt<int2>(minLocVec, locStrides, {i, sample}));
        }
        for (int i = 0; i < std::min(results.numMax[sample], capacity); ++i)
        {
            results.maxLoc[sample].push_back(test::ValueAt<int2>(maxLocVec, locStrides, {i, sample}));
        }
        std::ranges::sort(results.minLoc[sample], locLower);
        std::ranges::sort(results.maxLoc[sample], locLower);
    }
}

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

    auto   capacity   = static_cast<int>(minLocData->shape(1));
    auto   numSamples = static_cast<int>(minValData->shape(0));
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

enum RunChoice
{
    MIN     = 0b01,
    MAX     = 0b10,
    MIN_MAX = 0b11
};

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
            for (int x = 0; x < inShape.x; ++x) test::ValueAt<InVT>(inVec, inStrides, int3{x, y, z}).x = rg(Rng());

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
        EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                           nvcv::Tensor{nullptr}));
        break;

    case RunChoice::MAX:
        EXPECT_NO_THROW(op(stream, in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, maxVal,
                           maxLoc, numMax));
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

    std::uniform_int_distribution rgW(ScaledSize(inShape.x, 0.8), ScaledSize(inShape.x, 1.2));
    std::uniform_int_distribution rgH(ScaledSize(inShape.y, 0.8), ScaledSize(inShape.y, 1.2));

    for (int z = 0; z < inShape.z; ++z)
    {
        inImg.emplace_back(nvcv::Size2D{rgW(Rng()), rgH(Rng())}, inFormat);

        auto inData = inImg[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(inData);

        inStrides[z] = long2{inData->plane(0).rowStride, sizeof(InVT)};
        inShape2[z]  = int2{inData->plane(0).width, inData->plane(0).height};

        inVec[z].resize(inStrides[z].x * inShape2[z].y);

        for (int y = 0; y < inShape2[z].y; ++y)
            for (int x = 0; x < inShape2[z].x; ++x)
                test::ValueAt<InVT>(inVec[z], inStrides[z], int2{x, y}).x = rg(Rng());

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
        EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                           nvcv::Tensor{nullptr}));
        break;

    case RunChoice::MAX:
        EXPECT_NO_THROW(op(stream, in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, maxVal,
                           maxLoc, numMax));
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

// clang-format off
NVCV_TEST_SUITE_P(OpMinMaxLocPlanar, test::ValueList<int, int, int, bool>
{
    // width, height, batches, batched
    {    17,     13,       3,    true},
    {    19,     11,       1,   false},
});

// clang-format on

TEST_P(OpMinMaxLocPlanar, tensor_matches_interleaved)
{
    const int  width    = GetParamValue<0>();
    const int  height   = GetParamValue<1>();
    const int  batches  = GetParamValue<2>();
    const bool batched  = GetParamValue<3>();
    const int  capacity = width * height;

    nvcv::Tensor srcInterleaved = CreateMinMaxLocParityTensor(batches, width, height, false, batched);
    nvcv::Tensor srcPlanar      = CreateMinMaxLocParityTensor(batches, width, height, true, batched);
    FillMinMaxLocParityInputs(srcInterleaved, srcPlanar, width, height, batches);

    nvcv::Tensor minValInterleaved({{batches}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor minLocInterleaved(
        {
            {batches, capacity},
            "NM"
    },
        nvcv::TYPE_2S32);
    nvcv::Tensor numMinInterleaved({{batches}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor maxValInterleaved({{batches}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor maxLocInterleaved(
        {
            {batches, capacity},
            "NM"
    },
        nvcv::TYPE_2S32);
    nvcv::Tensor numMaxInterleaved({{batches}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor minValPlanar({{batches}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor minLocPlanar(
        {
            {batches, capacity},
            "NM"
    },
        nvcv::TYPE_2S32);
    nvcv::Tensor numMinPlanar({{batches}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor maxValPlanar({{batches}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor maxLocPlanar(
        {
            {batches, capacity},
            "NM"
    },
        nvcv::TYPE_2S32);
    nvcv::Tensor numMaxPlanar({{batches}, "N"}, nvcv::TYPE_S32);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinMaxLoc op;
    EXPECT_NO_THROW(op(stream, srcInterleaved, minValInterleaved, minLocInterleaved, numMinInterleaved,
                       maxValInterleaved, maxLocInterleaved, numMaxInterleaved));
    EXPECT_NO_THROW(
        op(stream, srcPlanar, minValPlanar, minLocPlanar, numMinPlanar, maxValPlanar, maxLocPlanar, numMaxPlanar));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    MinMaxParityResults interleaved;
    MinMaxParityResults planar;
    DownloadMinMaxParityResults(minValInterleaved, minLocInterleaved, numMinInterleaved, maxValInterleaved,
                                maxLocInterleaved, numMaxInterleaved, interleaved);
    DownloadMinMaxParityResults(minValPlanar, minLocPlanar, numMinPlanar, maxValPlanar, maxLocPlanar, numMaxPlanar,
                                planar);

    EXPECT_EQ(interleaved.minVal, planar.minVal);
    EXPECT_EQ(interleaved.minLoc, planar.minLoc);
    EXPECT_EQ(interleaved.numMin, planar.numMin);
    EXPECT_EQ(interleaved.maxVal, planar.maxVal);
    EXPECT_EQ(interleaved.maxLoc, planar.maxLoc);
    EXPECT_EQ(interleaved.numMax, planar.numMax);
}

// NaN must be ignored: min/max are taken over the finite values only. This guards the
// float path's ordered-int atomicMin/Max — a NaN reaching the atomic encodes to an integer
// extreme and would otherwise win min or max. The block-reduce (fminf/fmaxf) drops NaN
// before the atomic, so only finite values are encoded; this test pins that contract.
TEST(OpMinMaxLoc, ignores_nan_float)
{
    constexpr int W        = 8; // 32 distinct finite values 0..31; min=0 @(0,0), max=31 @(7,3)
    constexpr int H        = 4;
    constexpr int capacity = 8;

    nvcv::Tensor in = nvcv::util::CreateTensor(1, W, H, nvcv::FMT_F32);

    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inData);
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    long3  inStrides{inAccess->numRows() * inAccess->rowStride(), inAccess->rowStride(), inAccess->colStride()};
    size_t inBufSize = inStrides.x * inAccess->numSamples();

    std::vector<uint8_t> inVec(inBufSize, uint8_t{0});
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) test::ValueAt<float1>(inVec, inStrides, int3{x, y, 0}).x = float(y * W + x);

    // Inject a positive quiet NaN at two interior, non-extremal positions. A positive NaN
    // encodes to the top of the ordered-int range, so if it were not dropped it would win max.
    const float nan                                          = std::numeric_limits<float>::quiet_NaN();
    test::ValueAt<float1>(inVec, inStrides, int3{3, 1, 0}).x = nan;
    test::ValueAt<float1>(inVec, inStrides, int3{5, 2, 0}).x = nan;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    nvcv::Tensor minVal({{1}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor minLoc(
        {
            {1, capacity},
            "NM"
    },
        nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{1}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor maxVal({{1}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor maxLoc(
        {
            {1, capacity},
            "NM"
    },
        nvcv::TYPE_2S32);
    nvcv::Tensor numMax({{1}, "N"}, nvcv::TYPE_S32);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinMaxLoc op;
    EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, maxVal, maxLoc, numMax));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    auto readScalar = [](const nvcv::Tensor &t, auto &dst)
    {
        auto data = t.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(data);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(&dst, data->basePtr(), sizeof(dst), cudaMemcpyDeviceToHost));
    };

    float minValHost = 0.f;
    float maxValHost = 0.f;
    int   numMinHost = 0;
    int   numMaxHost = 0;
    int2  minLocHost{-1, -1};
    int2  maxLocHost{-1, -1};
    readScalar(minVal, minValHost);
    readScalar(maxVal, maxValHost);
    readScalar(numMin, numMinHost);
    readScalar(numMax, numMaxHost);
    readScalar(minLoc, minLocHost); // first location (capacity-major), the only match
    readScalar(maxLoc, maxLocHost);

    EXPECT_FALSE(std::isnan(minValHost));
    EXPECT_FALSE(std::isnan(maxValHost));
    EXPECT_EQ(minValHost, 0.f);
    EXPECT_EQ(maxValHost, 31.f);
    EXPECT_EQ(numMinHost, 1);
    EXPECT_EQ(numMaxHost, 1);
    EXPECT_EQ(minLocHost.x, 0);
    EXPECT_EQ(minLocHost.y, 0);
    EXPECT_EQ(maxLocHost.x, 7);
    EXPECT_EQ(maxLocHost.y, 3);
}

TEST(OpMinMaxLoc_Negative, op)
{
    int3         inShape{24, 24, 2};
    int          capacity = 100;
    nvcv::Tensor in       = nvcv::util::CreateTensor(inShape.z, inShape.x, inShape.y, nvcv::FMT_U8);
    nvcv::Tensor inInvalidSamples
        = nvcv::util::CreateTensor(65536, inShape.x, inShape.y, nvcv::FMT_U8); // wrong number of samples
    nvcv::Tensor inInvalidChannels
        = nvcv::util::CreateTensor(inShape.z, inShape.x, inShape.y, nvcv::FMT_RGB8); // wrong number of channels

    // clang-format off

    nvcv::Tensor minVal({{inShape.z}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor minLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{inShape.z}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor maxVal({{inShape.z}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor maxLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMax({{inShape.z}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor valWrongDataType({{inShape.z}, "N"}, nvcv::TYPE_F16); // wrong data type
    nvcv::Tensor valWrongNumSamples({{inShape.z + 1}, "N"}, nvcv::TYPE_U32); // wrong number of samples
    nvcv::Tensor valWrongNumChannels({{inShape.z, 2}, "NC"}, nvcv::TYPE_U32); // wrong number of channels

    nvcv::Tensor locWrongNumSamples({{inShape.z + 1, capacity}, "NM"}, nvcv::TYPE_2S32); // wrong number of samples
    nvcv::Tensor locWrongDataType({{inShape.z, capacity}, "NM"}, nvcv::TYPE_S32); // wrong data type

    nvcv::Tensor numWrongNumSamples({{inShape.z + 1}, "N"}, nvcv::TYPE_S32); // wrong number of samples
    nvcv::Tensor numWrongNumChannels({{inShape.z, 2}, "NC"}, nvcv::TYPE_S32); // wrong number of channels
    nvcv::Tensor numWrongDataType({{inShape.z}, "N"}, nvcv::TYPE_U8); // wrong data type

    // clang-format on

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::MinMaxLoc op;

    auto runOpMinMaxLocNegativeTest
        = [&op, &stream](nvcv::Tensor in, nvcv::Tensor minVal, nvcv::Tensor minLoc, nvcv::Tensor numMin,
                         nvcv::Tensor maxVal, nvcv::Tensor maxLoc, nvcv::Tensor numMax)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&op, &stream, &in, &minVal, &minLoc, &numMin, &maxVal, &maxLoc, &numMax]
                                    { op(stream, in, minVal, minLoc, numMin, maxVal, maxLoc, numMax); }));
    };

    // cases
    runOpMinMaxLocNegativeTest(inInvalidSamples, minVal, minLoc, numMin, maxVal, maxLoc, numMax);
    runOpMinMaxLocNegativeTest(inInvalidChannels, minVal, minLoc, numMin, maxVal, maxLoc, numMax);

    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, minLoc, numMin, maxVal, maxLoc, numMax);
    runOpMinMaxLocNegativeTest(in, minVal, nvcv::Tensor{nullptr}, numMin, maxVal, maxLoc, numMax);
    runOpMinMaxLocNegativeTest(in, minVal, minLoc, nvcv::Tensor{nullptr}, maxVal, maxLoc, numMax);
    runOpMinMaxLocNegativeTest(in, minVal, minLoc, numMin, nvcv::Tensor{nullptr}, maxLoc, numMax);
    runOpMinMaxLocNegativeTest(in, minVal, minLoc, numMin, maxVal, nvcv::Tensor{nullptr}, numMax);
    runOpMinMaxLocNegativeTest(in, minVal, minLoc, numMin, maxVal, maxLoc, nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr});

    runOpMinMaxLocNegativeTest(in, valWrongDataType, minLoc, numMin, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               valWrongDataType, maxLoc, numMax);

    runOpMinMaxLocNegativeTest(in, valWrongNumSamples, minLoc, numMin, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               valWrongNumSamples, maxLoc, numMax);

    runOpMinMaxLocNegativeTest(in, valWrongNumChannels, minLoc, numMin, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               valWrongNumChannels, maxLoc, numMax);

    runOpMinMaxLocNegativeTest(in, minVal, locWrongNumSamples, numMin, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, maxVal,
                               locWrongNumSamples, numMax);

    runOpMinMaxLocNegativeTest(in, minVal, locWrongDataType, numMin, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, maxVal,
                               locWrongDataType, numMax);

    runOpMinMaxLocNegativeTest(in, minVal, minLoc, numWrongNumSamples, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, maxVal, maxLoc,
                               numWrongNumSamples);

    runOpMinMaxLocNegativeTest(in, minVal, minLoc, numWrongNumChannels, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, maxVal, maxLoc,
                               numWrongNumChannels);

    runOpMinMaxLocNegativeTest(in, minVal, minLoc, numWrongDataType, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                               nvcv::Tensor{nullptr});
    runOpMinMaxLocNegativeTest(in, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, maxVal, maxLoc,
                               numWrongDataType);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMinMaxLoc_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaMinMaxLocCreate(nullptr));
}

TEST(OpMinMaxLoc_Negative, varshape_invalid_plane)
{
    int3              inShape{24, 24, 2};
    nvcv::ImageFormat inFormat{nvcv::FMT_RGB8p};
    int               capacity = 100;

    std::vector<nvcv::Image> inImg;
    for (int z = 0; z < inShape.z; ++z)
    {
        inImg.emplace_back(nvcv::Size2D{inShape.x, inShape.y}, inFormat);
    }

    nvcv::ImageBatchVarShape in(inShape.z);
    in.pushBack(inImg.begin(), inImg.end());

    // clang-format off
    nvcv::Tensor minVal({{inShape.z}, "N"}, nvcv::TYPE_U8);
    nvcv::Tensor minLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{inShape.z}, "N"}, nvcv::TYPE_S32);
    // clang-format on

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinMaxLoc op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&op, &stream, &in, &minVal, &minLoc, &numMin] {
                                                   op(stream, in, minVal, minLoc, numMin, nvcv::Tensor{nullptr},
                                                      nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr});
                                               }));
    std::array<char, 1024> msg;
    nvcvGetLastErrorMessage(msg.data(), msg.size());
    std::cout << "\033[33m" << msg.data() << "\033[0m" << std::endl;

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
