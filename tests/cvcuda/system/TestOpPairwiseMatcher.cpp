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

#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <cvcuda/OpPairwiseMatcher.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

// ----------------------- Basic utility definitions ---------------------------

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;
namespace type = nvcv::test::type;

using RawBufferType = std::vector<uint8_t>;

template<typename T>
using uniform_distribution
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

template<typename T>
constexpr nvcv::DataType ToDataType()
{
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        return nvcv::TYPE_U8;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        return nvcv::TYPE_U32;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return nvcv::TYPE_F32;
    }
}

// --------------------- Reference (gold) computations -------------------------

namespace ref {

template<typename T>
T absdiff(T a, T b)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return std::abs(a - b);
    }
    else
    {
        return a < b ? b - a : a - b;
    }
}

template<typename DT, typename ST>
void ComputeDistance(DT &dist, ST p1, ST p2, NVCVNormType normType)
{
    if (normType == NVCV_NORM_HAMMING)
    {
        if constexpr (!std::is_floating_point_v<ST>)
        {
            dist = static_cast<DT>(dist + static_cast<DT>(std::bitset<sizeof(ST) * 8>(p1 ^ p2).count()));
        }
    }
    else if (normType == NVCV_NORM_L1)
    {
        dist = static_cast<DT>(dist + static_cast<DT>(absdiff(p1, p2)));
    }
    else if (normType == NVCV_NORM_L2)
    {
        dist = static_cast<DT>(dist + std::pow(absdiff(p1, p2), 2));
    }
}

template<typename ST>
float DescriptorDistance(const RawBufferType &set1Vec, const RawBufferType &set2Vec, const long3 &set1Strides,
                         const long3 &set2Strides, int sampleIdx, int set1Idx, int set2Idx, int numDim,
                         NVCVNormType normType)
{
    float dist = 0.f;

    for (int coordIdx = 0; coordIdx < numDim; coordIdx++)
    {
        ST p1 = util::ValueAt<ST>(set1Vec, set1Strides, long3{sampleIdx, set1Idx, coordIdx});
        ST p2 = util::ValueAt<ST>(set2Vec, set2Strides, long3{sampleIdx, set2Idx, coordIdx});

        ComputeDistance(dist, p1, p2, normType);
    }

    return normType == NVCV_NORM_L2 ? std::sqrt(dist) : dist;
}

template<typename ST>
void ComputeSortedDistances(std::vector<std::tuple<float, int>> &distIdx, const RawBufferType &set1Vec,
                            const RawBufferType &set2Vec, const long3 &set1Strides, const long3 &set2Strides,
                            int sampleIdx, int set1Idx, int numDim, NVCVNormType normType)
{
    for (int set2Idx = 0; set2Idx < static_cast<int>(distIdx.size()); set2Idx++)
    {
        distIdx[set2Idx] = std::make_tuple(DescriptorDistance<ST>(set1Vec, set2Vec, set1Strides, set2Strides, sampleIdx,
                                                                  set1Idx, set2Idx, numDim, normType),
                                           set2Idx);
    }

    std::ranges::sort(distIdx);
}

template<typename ST>
int FindCrossCheckMatch(std::vector<std::tuple<float, int>> &cckDistIdx, const RawBufferType &set1Vec,
                        const RawBufferType &set2Vec, const long3 &set1Strides, const long3 &set2Strides, int sampleIdx,
                        int set2Idx, int numDim, NVCVNormType normType)
{
    for (int cck1Idx = 0; cck1Idx < static_cast<int>(cckDistIdx.size()); cck1Idx++)
    {
        cckDistIdx[cck1Idx] = std::make_tuple(DescriptorDistance<ST>(set1Vec, set2Vec, set1Strides, set2Strides,
                                                                     sampleIdx, cck1Idx, set2Idx, numDim, normType),
                                              cck1Idx);
    }

    std::ranges::sort(cckDistIdx);
    return std::get<1>(cckDistIdx[0]);
}

inline void StoreMatch(RawBufferType &mchVec, RawBufferType &nmVec, RawBufferType &dVec, const long3 &mchStrides,
                       const long1 &nmStrides, const long2 &dStrides, int sampleIdx, int &mchIdx, int set1Idx,
                       int set2Idx, float distance)
{
    util::ValueAt<int>(mchVec, mchStrides, long3{sampleIdx, mchIdx, 0}) = set1Idx;
    util::ValueAt<int>(mchVec, mchStrides, long3{sampleIdx, mchIdx, 1}) = set2Idx;

    if (dStrides.x > 0)
    {
        util::ValueAt<float>(dVec, dStrides, long2{sampleIdx, mchIdx}) = distance;
    }

    mchIdx++;

    if (nmStrides.x > 0)
    {
        util::ValueAt<int>(nmVec, nmStrides, long1{sampleIdx}) = mchIdx;
    }
}

template<typename ST>
void BruteForceMatcher(RawBufferType &mchVec, RawBufferType &nmVec, RawBufferType &dVec, const RawBufferType &set1Vec,
                       const RawBufferType &set2Vec, const long3 &mchStrides, const long1 &nmStrides,
                       const long2 &dStrides, const long3 &set1Strides, const long3 &set2Strides, int numSamples,
                       int numDim, int set1Size, int set2Size, bool crossCheck, int matchesPerPoint,
                       NVCVNormType normType)
{
    std::vector<std::tuple<float, int>> distIdx(set2Size);
    std::vector<std::tuple<float, int>> cckDistIdx(set1Size);

    auto storeCrossCheckMatch
        = [&cckDistIdx, &dStrides, &dVec, &mchStrides, &mchVec, &nmStrides, &nmVec, &set1Strides, &set1Vec,
           &set2Strides, &set2Vec, numDim, normType](int sampleIdx, int &mchIdx, int set1Idx,
                                                     const std::vector<std::tuple<float, int>> &sortedDistances)
    {
        int set2Idx = std::get<1>(sortedDistances[0]);

        if (FindCrossCheckMatch<ST>(cckDistIdx, set1Vec, set2Vec, set1Strides, set2Strides, sampleIdx, set2Idx, numDim,
                                    normType)
            == set1Idx)
        {
            StoreMatch(mchVec, nmVec, dVec, mchStrides, nmStrides, dStrides, sampleIdx, mchIdx, set1Idx, set2Idx,
                       std::get<0>(sortedDistances[0]));
        }
    };

    auto storeBestMatches
        = [&dStrides, &dVec, &mchStrides, &mchVec, &nmStrides, &nmVec, matchesPerPoint](
              int sampleIdx, int &mchIdx, int set1Idx, const std::vector<std::tuple<float, int>> &sortedDistances)
    {
        for (int m = 0; m < matchesPerPoint; m++)
        {
            StoreMatch(mchVec, nmVec, dVec, mchStrides, nmStrides, dStrides, sampleIdx, mchIdx, set1Idx,
                       std::get<1>(sortedDistances[m]), std::get<0>(sortedDistances[m]));
        }
    };

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        int mchIdx = 0;

        for (int set1Idx = 0; set1Idx < set1Size; set1Idx++)
        {
            ComputeSortedDistances<ST>(distIdx, set1Vec, set2Vec, set1Strides, set2Strides, sampleIdx, set1Idx, numDim,
                                       normType);

            if (crossCheck)
            {
                storeCrossCheckMatch(sampleIdx, mchIdx, set1Idx, distIdx);
            }
            else
            {
                storeBestMatches(sampleIdx, mchIdx, set1Idx, distIdx);
            }
        }
    }
}

template<typename ST>
void PairwiseMatcher(NVCVPairwiseMatcherType algoChoice, RawBufferType &mchVec, RawBufferType &nmVec,
                     RawBufferType &dVec, const RawBufferType &set1Vec, const RawBufferType &set2Vec,
                     const long3 &mchStrides, const long1 &nmStrides, const long2 &dStrides, const long3 &set1Strides,
                     const long3 &set2Strides, int numSamples, int numDim, int set1Size, int set2Size, bool crossCheck,
                     int matchesPerPoint, NVCVNormType normType)
{
    if (algoChoice == NVCV_BRUTE_FORCE)
    {
        BruteForceMatcher<ST>(mchVec, nmVec, dVec, set1Vec, set2Vec, mchStrides, nmStrides, dStrides, set1Strides,
                              set2Strides, numSamples, numDim, set1Size, set2Size, crossCheck, matchesPerPoint,
                              normType);
    }
}

inline void SortOutput(std::vector<std::tuple<int, int, int, float>> &outIdsDist, const RawBufferType &mchVec,
                       const RawBufferType &nmVec, const RawBufferType &dVec, const long3 &mchStrides,
                       const long1 &nmStrides, const long2 &dStrides, int numSamples, int set1Size, int matchesPerPoint,
                       int maxMatches)
{
    int totalMatches = set1Size * matchesPerPoint;

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        if (nmStrides.x > 0)
        {
            totalMatches = util::ValueAt<int>(nmVec, nmStrides, long1{sampleIdx});
        }

        for (int matchIdx = 0; matchIdx < totalMatches && matchIdx < maxMatches; matchIdx++)
        {
            int   set1Idx  = util::ValueAt<int>(mchVec, mchStrides, long3{sampleIdx, matchIdx, 0});
            int   set2Idx  = util::ValueAt<int>(mchVec, mchStrides, long3{sampleIdx, matchIdx, 1});
            float distance = (dStrides.x > 0) ? util::ValueAt<float>(dVec, dStrides, long2{sampleIdx, matchIdx}) : 0.f;

            outIdsDist.emplace_back(sampleIdx, set1Idx, set2Idx, distance);
        }
    }

    std::ranges::sort(outIdsDist);
}

} // namespace ref

// ----------------------------- Start tests -----------------------------------

// clang-format off

#define NVCV_TEST_ROW(NumSamples, Set1Size, Set2Size, NumDim, MatchesPerPoint, CrossCheck, StoreDistances,  \
                      AlgoChoice, NormType, Type)                                                           \
    type::Types<type::Value<NumSamples>, type::Value<Set1Size>, type::Value<Set2Size>, type::Value<NumDim>, \
                type::Value<MatchesPerPoint>, type::Value<CrossCheck>, type::Value<StoreDistances>,         \
                type::Value<AlgoChoice>, type::Value<NormType>, Type>

NVCV_TYPED_TEST_SUITE(OpPairwiseMatcher, type::Types<
    NVCV_TEST_ROW(1, 2, 2, 1, 1, false, false, NVCV_BRUTE_FORCE, NVCV_NORM_HAMMING, uint8_t),
    NVCV_TEST_ROW(2, 3, 4, 5, 1, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_HAMMING, uint8_t),
    NVCV_TEST_ROW(2, 13, 14, 32, 1, true, true, NVCV_BRUTE_FORCE, NVCV_NORM_HAMMING, uint8_t),
    NVCV_TEST_ROW(3, 4, 3, 32, 1, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_HAMMING, uint32_t),
    NVCV_TEST_ROW(4, 11, 12, 128, 2, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_HAMMING, uint8_t),
    NVCV_TEST_ROW(3, 17, 16, 128, 3, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_HAMMING, uint8_t),
    NVCV_TEST_ROW(2, 3, 4, 32, 1, true, false, NVCV_BRUTE_FORCE, NVCV_NORM_HAMMING, uint32_t),
    NVCV_TEST_ROW(1, 5, 6, 7, 1, false, false, NVCV_BRUTE_FORCE, NVCV_NORM_L1, uint8_t),
    NVCV_TEST_ROW(2, 18, 19, 17, 1, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_L1, uint32_t),
    NVCV_TEST_ROW(3, 98, 17, 32, 1, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_L1, float),
    NVCV_TEST_ROW(2, 54, 65, 32, 2, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_L1, uint8_t),
    NVCV_TEST_ROW(3, 68, 37, 1025, 1, true, true, NVCV_BRUTE_FORCE, NVCV_NORM_L1, float),
    NVCV_TEST_ROW(2, 14, 24, 32, 3, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_L1, uint8_t),
    NVCV_TEST_ROW(3, 48, 37, 8, 1, true, false, NVCV_BRUTE_FORCE, NVCV_NORM_L1, float),
    NVCV_TEST_ROW(4, 8, 9, 1025, 1, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_L2, uint8_t),
    NVCV_TEST_ROW(3, 27, 16, 8, 1, false, false, NVCV_BRUTE_FORCE, NVCV_NORM_L2, uint32_t),
    NVCV_TEST_ROW(2, 73, 132, 64, 1, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_L2, float),
    NVCV_TEST_ROW(3, 87, 98, 19, 2, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_L2, uint8_t),
    NVCV_TEST_ROW(4, 43, 32, 26, 1, true, true, NVCV_BRUTE_FORCE, NVCV_NORM_L2, float),
    NVCV_TEST_ROW(3, 67, 58, 32, 3, false, true, NVCV_BRUTE_FORCE, NVCV_NORM_L2, uint8_t),
    NVCV_TEST_ROW(2, 73, 62, 8, 1, true, false, NVCV_BRUTE_FORCE, NVCV_NORM_L2, float),
    NVCV_TEST_ROW(1, 3, 4, 128, 1, false, false, NVCV_BRUTE_FORCE, NVCV_NORM_HAMMING, uint8_t)
>);

// clang-format on

TYPED_TEST(OpPairwiseMatcher, CorrectOutput)
{
    int  numSamples      = type::GetValue<TypeParam, 0>;
    int  set1Size        = type::GetValue<TypeParam, 1>;
    int  set2Size        = type::GetValue<TypeParam, 2>;
    int  numDim          = type::GetValue<TypeParam, 3>;
    int  matchesPerPoint = type::GetValue<TypeParam, 4>;
    bool crossCheck      = type::GetValue<TypeParam, 5>;
    bool storeDistances  = type::GetValue<TypeParam, 6>;

    NVCVPairwiseMatcherType algoChoice{type::GetValue<TypeParam, 7>};

    NVCVNormType normType{type::GetValue<TypeParam, 8>};

    using SrcT = type::GetType<TypeParam, 9>;

    constexpr nvcv::DataType srcDT{ToDataType<SrcT>()};

    int maxSet1    = set1Size + 12;
    int maxSet2    = set2Size + 23; // adding extra sizes to test different capacities on set 1 and 2
    int maxMatches = maxSet1 * matchesPerPoint;

    // clang-format off

    nvcv::Tensor set1({{numSamples, maxSet1, numDim}, "NMD"}, srcDT);
    nvcv::Tensor set2({{numSamples, maxSet2, numDim}, "NMD"}, srcDT);

    nvcv::Tensor numSet1({{numSamples}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor numSet2({{numSamples}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor matches({{numSamples, maxMatches, 2}, "NMD"}, nvcv::TYPE_S32);

    nvcv::Tensor numMatches;
    nvcv::Optional<nvcv::TensorDataStridedCuda> nmData;

    nvcv::Tensor distances;
    nvcv::Optional<nvcv::TensorDataStridedCuda> dData;

    numMatches = nvcv::Tensor({{numSamples}, "N"}, nvcv::TYPE_S32);
    nmData = numMatches.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(nmData);

    if (storeDistances)
    {
        distances = nvcv::Tensor({{numSamples, maxMatches}, "NM"}, nvcv::TYPE_F32);

        dData = distances.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(dData);
    }

    // clang-format on

    auto set1Data = set1.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(set1Data);

    auto set2Data = set2.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(set2Data);

    auto ns1Data = numSet1.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(ns1Data);

    auto ns2Data = numSet2.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(ns2Data);

    auto mchData = matches.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(mchData);

    long3 set1Strides{set1Data->stride(0), set1Data->stride(1), set1Data->stride(2)};
    long3 set2Strides{set2Data->stride(0), set2Data->stride(1), set2Data->stride(2)};
    long1 ns1Strides{ns1Data->stride(0)};
    long1 ns2Strides{ns2Data->stride(0)};
    long3 mchStrides{mchData->stride(0), mchData->stride(1), mchData->stride(2)};
    long1 nmStrides = numMatches ? long1{nmData->stride(0)} : long1{0};
    long2 dStrides  = distances ? long2{dData->stride(0), dData->stride(1)} : long2{0, 0};

    long set1BufSize = set1Strides.x * numSamples;
    long set2BufSize = set2Strides.x * numSamples;
    long ns1BufSize  = ns1Strides.x * numSamples;
    long ns2BufSize  = ns2Strides.x * numSamples;
    long mchBufSize  = mchStrides.x * numSamples;
    long nmBufSize   = nmStrides.x * numSamples;
    long dBufSize    = dStrides.x * numSamples;

    RawBufferType set1Vec(set1BufSize);
    RawBufferType set2Vec(set2BufSize);
    RawBufferType ns1Vec(ns1BufSize);
    RawBufferType ns2Vec(ns2BufSize);

    std::default_random_engine rng(12345u);

    SrcT minV = std::is_integral_v<SrcT> ? cuda::TypeTraits<SrcT>::min : -1;
    SrcT maxV = std::is_integral_v<SrcT> ? cuda::TypeTraits<SrcT>::max : +1;

    uniform_distribution<SrcT> rand(minV, maxV);

    for (int x = 0; x < numSamples; ++x)
    {
        for (int z = 0; z < numDim; ++z)
        {
            for (int y = 0; y < set1Size; ++y)
            {
                util::ValueAt<SrcT>(set1Vec, set1Strides, long3{x, y, z}) = rand(rng);
            }
            for (int y = 0; y < set2Size; ++y)
            {
                util::ValueAt<SrcT>(set2Vec, set2Strides, long3{x, y, z}) = rand(rng);
            }
        }

        util::ValueAt<int>(ns1Vec, ns1Strides, long1{x}) = set1Size;
        util::ValueAt<int>(ns2Vec, ns2Strides, long1{x}) = set2Size;
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(set1Data->basePtr(), set1Vec.data(), set1BufSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(set2Data->basePtr(), set2Vec.data(), set2BufSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(ns1Data->basePtr(), ns1Vec.data(), ns1BufSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(ns2Data->basePtr(), ns2Vec.data(), ns2BufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::PairwiseMatcher op(algoChoice);

    op(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, crossCheck, matchesPerPoint, normType);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    RawBufferType nmTestVec(nmBufSize, 0);
    RawBufferType nmGoldVec(nmBufSize, 0);
    RawBufferType mchTestVec(mchBufSize, 0);
    RawBufferType mchGoldVec(mchBufSize, 0);
    RawBufferType dTestVec(dBufSize, 0);
    RawBufferType dGoldVec(dBufSize, 0);

    // Treated output is a vector of (sampleIdx, set1Idx, set2Idx, distance)
    std::vector<std::tuple<int, int, int, float>> testIdsDist;
    std::vector<std::tuple<int, int, int, float>> goldIdsDist;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(mchTestVec.data(), mchData->basePtr(), mchBufSize, cudaMemcpyDeviceToHost));

    if (numMatches)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(nmTestVec.data(), nmData->basePtr(), nmBufSize, cudaMemcpyDeviceToHost));
    }
    if (distances)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dTestVec.data(), dData->basePtr(), dBufSize, cudaMemcpyDeviceToHost));
    }

    ref::SortOutput(testIdsDist, mchTestVec, nmTestVec, dTestVec, mchStrides, nmStrides, dStrides, numSamples, set1Size,
                    matchesPerPoint, maxMatches);

    ref::PairwiseMatcher<SrcT>(algoChoice, mchGoldVec, nmGoldVec, dGoldVec, set1Vec, set2Vec, mchStrides, nmStrides,
                               dStrides, set1Strides, set2Strides, numSamples, numDim, set1Size, set2Size, crossCheck,
                               matchesPerPoint, normType);

    ref::SortOutput(goldIdsDist, mchGoldVec, nmGoldVec, dGoldVec, mchStrides, nmStrides, dStrides, numSamples, set1Size,
                    matchesPerPoint, maxMatches);

    EXPECT_EQ(testIdsDist, goldIdsDist);
}

TEST(OpPairwiseMatcher, optional_counts_and_output_capacity)
{
    constexpr int numSamples      = 1;
    constexpr int set1Capacity    = 2;
    constexpr int set2Capacity    = 3;
    constexpr int numDim          = 7;
    constexpr int matchesPerPoint = 1;
    constexpr int outCapacity     = 1;

    // clang-format off
    nvcv::Tensor set1({{numSamples, set1Capacity, numDim}, "NMD"}, nvcv::TYPE_U8);
    nvcv::Tensor set2({{numSamples, set2Capacity, numDim}, "NMD"}, nvcv::TYPE_U8);
    // Null count tensors force the kernels to use the input capacities.
    nvcv::Tensor numSet1;
    nvcv::Tensor numSet2;
    nvcv::Tensor matches({{numSamples, outCapacity, 2}, "NMD"}, nvcv::TYPE_S32);
    nvcv::Tensor numMatches({{numSamples}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor distances;
    // clang-format on

    auto set1Data = set1.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(set1Data);
    auto set2Data = set2.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(set2Data);
    auto mchData = matches.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(mchData);
    auto nmData = numMatches.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(nmData);

    long3 set1Strides{set1Data->stride(0), set1Data->stride(1), set1Data->stride(2)};
    long3 set2Strides{set2Data->stride(0), set2Data->stride(1), set2Data->stride(2)};
    long3 mchStrides{mchData->stride(0), mchData->stride(1), mchData->stride(2)};
    long1 nmStrides{nmData->stride(0)};
    long2 noDistanceStrides{0, 0};

    long set1BufSize = set1Strides.x * numSamples;
    long set2BufSize = set2Strides.x * numSamples;
    long mchBufSize  = mchStrides.x * numSamples;
    long nmBufSize   = nmStrides.x * numSamples;

    RawBufferType set1Vec(set1BufSize, 0);
    RawBufferType set2Vec(set2BufSize, 0);

    for (int i = 0; i < numDim; ++i)
    {
        util::ValueAt<uint8_t>(set1Vec, set1Strides, long3{0, 0, i}) = 0;
        util::ValueAt<uint8_t>(set1Vec, set1Strides, long3{0, 1, i}) = 100;
        util::ValueAt<uint8_t>(set2Vec, set2Strides, long3{0, 0, i}) = 0;
        util::ValueAt<uint8_t>(set2Vec, set2Strides, long3{0, 1, i}) = 100;
        util::ValueAt<uint8_t>(set2Vec, set2Strides, long3{0, 2, i}) = 200;
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(set1Data->basePtr(), set1Vec.data(), set1BufSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(set2Data->basePtr(), set2Vec.data(), set2BufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::PairwiseMatcher op(NVCV_BRUTE_FORCE);
    op(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, false, matchesPerPoint, NVCV_NORM_L1);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    RawBufferType mchTestVec(mchBufSize, 0);
    RawBufferType nmTestVec(nmBufSize, 0);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(mchTestVec.data(), mchData->basePtr(), mchBufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(nmTestVec.data(), nmData->basePtr(), nmBufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Keep the reference at full capacity so the device output limit is modeled explicitly.
    const long3   goldMchStrides{set1Capacity * 2 * sizeof(int), 2 * sizeof(int), sizeof(int)};
    const long1   goldNmStrides{sizeof(int)};
    RawBufferType goldMchVec(goldMchStrides.x * numSamples, 0);
    RawBufferType goldNmVec(goldNmStrides.x * numSamples, 0);
    RawBufferType noDistanceVec;

    ref::PairwiseMatcher<uint8_t>(NVCV_BRUTE_FORCE, goldMchVec, goldNmVec, noDistanceVec, set1Vec, set2Vec,
                                  goldMchStrides, goldNmStrides, noDistanceStrides, set1Strides, set2Strides,
                                  numSamples, numDim, set1Capacity, set2Capacity, false, matchesPerPoint, NVCV_NORM_L1);

    std::vector<std::tuple<int, int, int, float>> testIdsDist;
    std::vector<std::tuple<int, int, int, float>> goldIdsDist;
    ref::SortOutput(testIdsDist, mchTestVec, nmTestVec, noDistanceVec, mchStrides, nmStrides, noDistanceStrides,
                    numSamples, set1Capacity, matchesPerPoint, outCapacity);
    ref::SortOutput(goldIdsDist, goldMchVec, goldNmVec, noDistanceVec, goldMchStrides, goldNmStrides, noDistanceStrides,
                    numSamples, set1Capacity, matchesPerPoint, outCapacity);

    EXPECT_EQ(testIdsDist, goldIdsDist);
    EXPECT_EQ(set1Capacity * matchesPerPoint, util::ValueAt<int>(nmTestVec, nmStrides, long1{0}));
}

TEST(OpPairwiseMatcher, rank_two_optional_shapes)
{
    constexpr int numSamples      = 2;
    constexpr int set1Size        = 2;
    constexpr int set2Size        = 3;
    constexpr int set1Capacity    = 3;
    constexpr int set2Capacity    = 4;
    constexpr int numDim          = 5;
    constexpr int matchesPerPoint = 1;
    constexpr int outCapacity     = set1Size * matchesPerPoint;

    // clang-format off
    nvcv::Tensor set1({{numSamples, set1Capacity, numDim}, "NMD"}, nvcv::TYPE_U8);
    nvcv::Tensor set2({{numSamples, set2Capacity, numDim}, "NMD"}, nvcv::TYPE_U8);
    nvcv::Tensor numSet1({{numSamples, 1}, "NC"}, nvcv::TYPE_S32);
    nvcv::Tensor numSet2({{numSamples, 1}, "NC"}, nvcv::TYPE_S32);
    nvcv::Tensor matches({{numSamples, outCapacity, 2}, "NMD"}, nvcv::TYPE_S32);
    nvcv::Tensor numMatches({{numSamples, 1}, "NC"}, nvcv::TYPE_S32);
    nvcv::Tensor distances({{numSamples, outCapacity, 1}, "NMC"}, nvcv::TYPE_F32);
    // clang-format on

    auto set1Data       = set1.exportData<nvcv::TensorDataStridedCuda>();
    auto set2Data       = set2.exportData<nvcv::TensorDataStridedCuda>();
    auto numSet1Data    = numSet1.exportData<nvcv::TensorDataStridedCuda>();
    auto numSet2Data    = numSet2.exportData<nvcv::TensorDataStridedCuda>();
    auto matchesData    = matches.exportData<nvcv::TensorDataStridedCuda>();
    auto numMatchesData = numMatches.exportData<nvcv::TensorDataStridedCuda>();
    auto distancesData  = distances.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(set1Data && set2Data && numSet1Data && numSet2Data && matchesData && numMatchesData && distancesData);

    long3 set1Strides{set1Data->stride(0), set1Data->stride(1), set1Data->stride(2)};
    long3 set2Strides{set2Data->stride(0), set2Data->stride(1), set2Data->stride(2)};
    long1 numSet1Strides{numSet1Data->stride(0)};
    long1 numSet2Strides{numSet2Data->stride(0)};
    long3 mchStrides{matchesData->stride(0), matchesData->stride(1), matchesData->stride(2)};
    long1 nmStrides{numMatchesData->stride(0)};
    long2 dStrides{distancesData->stride(0), distancesData->stride(1)};

    RawBufferType set1Vec(set1Strides.x * numSamples, 0);
    RawBufferType set2Vec(set2Strides.x * numSamples, 0);
    for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
    {
        for (int set1Idx = 0; set1Idx < set1Size; ++set1Idx)
        {
            for (int dimIdx = 0; dimIdx < numDim; ++dimIdx)
            {
                util::ValueAt<uint8_t>(set1Vec, set1Strides, long3{sampleIdx, set1Idx, dimIdx})
                    = static_cast<uint8_t>(10 * sampleIdx + 7 * set1Idx + dimIdx);
            }
        }
        for (int set2Idx = 0; set2Idx < set2Size; ++set2Idx)
        {
            for (int dimIdx = 0; dimIdx < numDim; ++dimIdx)
            {
                util::ValueAt<uint8_t>(set2Vec, set2Strides, long3{sampleIdx, set2Idx, dimIdx})
                    = static_cast<uint8_t>(10 * sampleIdx + 7 * set2Idx + dimIdx + (set2Idx == 0 ? 0 : 3));
            }
        }
    }

    RawBufferType numSet1Vec(numSet1Strides.x * numSamples, 0);
    RawBufferType numSet2Vec(numSet2Strides.x * numSamples, 0);
    for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
    {
        util::ValueAt<int>(numSet1Vec, numSet1Strides, long1{sampleIdx}) = set1Size;
        util::ValueAt<int>(numSet2Vec, numSet2Strides, long1{sampleIdx}) = set2Size;
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(set1Data->basePtr(), set1Vec.data(), set1Vec.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(set2Data->basePtr(), set2Vec.data(), set2Vec.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(numSet1Data->basePtr(), numSet1Vec.data(), numSet1Vec.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(numSet2Data->basePtr(), numSet2Vec.data(), numSet2Vec.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::PairwiseMatcher op(NVCV_BRUTE_FORCE);
    op(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, false, matchesPerPoint, NVCV_NORM_L1);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    long          mchBufSize = mchStrides.x * numSamples;
    long          nmBufSize  = nmStrides.x * numSamples;
    long          dBufSize   = dStrides.x * numSamples;
    RawBufferType mchTestVec(mchBufSize, 0);
    RawBufferType nmTestVec(nmBufSize, 0);
    RawBufferType dTestVec(dBufSize, 0);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(mchTestVec.data(), matchesData->basePtr(), mchBufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(nmTestVec.data(), numMatchesData->basePtr(), nmBufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dTestVec.data(), distancesData->basePtr(), dBufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    RawBufferType mchGoldVec(mchBufSize, 0);
    RawBufferType nmGoldVec(nmBufSize, 0);
    RawBufferType dGoldVec(dBufSize, 0);
    ref::PairwiseMatcher<uint8_t>(NVCV_BRUTE_FORCE, mchGoldVec, nmGoldVec, dGoldVec, set1Vec, set2Vec, mchStrides,
                                  nmStrides, dStrides, set1Strides, set2Strides, numSamples, numDim, set1Size, set2Size,
                                  false, matchesPerPoint, NVCV_NORM_L1);

    std::vector<std::tuple<int, int, int, float>> testIdsDist;
    std::vector<std::tuple<int, int, int, float>> goldIdsDist;
    ref::SortOutput(testIdsDist, mchTestVec, nmTestVec, dTestVec, mchStrides, nmStrides, dStrides, numSamples, set1Size,
                    matchesPerPoint, outCapacity);
    ref::SortOutput(goldIdsDist, mchGoldVec, nmGoldVec, dGoldVec, mchStrides, nmStrides, dStrides, numSamples, set1Size,
                    matchesPerPoint, outCapacity);

    EXPECT_EQ(testIdsDist, goldIdsDist);
}

TEST(OpPairwiseMatcher, padded_optional_tensor_strides_regression)
{
    auto runCrossCheck = [&](bool crossCheck)
    {
        SCOPED_TRACE(crossCheck);
        constexpr int numSamples      = 2;
        constexpr int set1Size        = 2;
        constexpr int set2Size        = 3;
        constexpr int set1Capacity    = 3;
        constexpr int set2Capacity    = 4;
        constexpr int numDim          = 5;
        constexpr int matchesPerPoint = 1;
        constexpr int outCapacity     = set1Size * matchesPerPoint;

        // clang-format off
        nvcv::Tensor set1({{numSamples, set1Capacity, numDim}, "NMD"}, nvcv::TYPE_U8);
        nvcv::Tensor set2({{numSamples, set2Capacity, numDim}, "NMD"}, nvcv::TYPE_U8);
        nvcv::Tensor numSet1({{numSamples, 1}, "NC"}, nvcv::TYPE_S32);
        nvcv::Tensor numSet2({{numSamples, 1}, "NC"}, nvcv::TYPE_S32);
        nvcv::Tensor matches({{numSamples, outCapacity, 2}, "NMD"}, nvcv::TYPE_S32);
        nvcv::Tensor numMatches({{numSamples, 1}, "NC"}, nvcv::TYPE_S32);
        nvcv::Tensor distances({{numSamples, outCapacity, 1}, "NMC"}, nvcv::TYPE_F32);
        // clang-format on

        auto set1Data       = set1.exportData<nvcv::TensorDataStridedCuda>();
        auto set2Data       = set2.exportData<nvcv::TensorDataStridedCuda>();
        auto numSet1Data    = numSet1.exportData<nvcv::TensorDataStridedCuda>();
        auto numSet2Data    = numSet2.exportData<nvcv::TensorDataStridedCuda>();
        auto matchesData    = matches.exportData<nvcv::TensorDataStridedCuda>();
        auto numMatchesData = numMatches.exportData<nvcv::TensorDataStridedCuda>();
        auto distancesData  = distances.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(set1Data && set2Data && numSet1Data && numSet2Data && matchesData && numMatchesData
                    && distancesData);

        long3 set1Strides{set1Data->stride(0), set1Data->stride(1), set1Data->stride(2)};
        long3 set2Strides{set2Data->stride(0), set2Data->stride(1), set2Data->stride(2)};
        long1 numSet1Strides{numSet1Data->stride(0)};
        long1 numSet2Strides{numSet2Data->stride(0)};
        long3 mchStrides{matchesData->stride(0), matchesData->stride(1), matchesData->stride(2)};
        long1 nmStrides{numMatchesData->stride(0)};
        long2 dStrides{distancesData->stride(0), distancesData->stride(1)};

        RawBufferType set1Vec(set1Strides.x * numSamples, 0);
        RawBufferType set2Vec(set2Strides.x * numSamples, 0);
        for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
        {
            for (int set1Idx = 0; set1Idx < set1Size; ++set1Idx)
            {
                for (int dimIdx = 0; dimIdx < numDim; ++dimIdx)
                {
                    util::ValueAt<uint8_t>(set1Vec, set1Strides, long3{sampleIdx, set1Idx, dimIdx})
                        = static_cast<uint8_t>(10 * sampleIdx + 7 * set1Idx + dimIdx);
                }
            }
            for (int set2Idx = 0; set2Idx < set2Size; ++set2Idx)
            {
                for (int dimIdx = 0; dimIdx < numDim; ++dimIdx)
                {
                    util::ValueAt<uint8_t>(set2Vec, set2Strides, long3{sampleIdx, set2Idx, dimIdx})
                        = static_cast<uint8_t>(10 * sampleIdx + 7 * set2Idx + dimIdx + (set2Idx == 0 ? 0 : 3));
                }
            }
        }

        RawBufferType numSet1Vec(numSet1Strides.x * numSamples, 0);
        RawBufferType numSet2Vec(numSet2Strides.x * numSamples, 0);
        for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
        {
            util::ValueAt<int>(numSet1Vec, numSet1Strides, long1{sampleIdx}) = set1Size;
            util::ValueAt<int>(numSet2Vec, numSet2Strides, long1{sampleIdx}) = set2Size;
        }

        ASSERT_EQ(cudaSuccess, cudaMemcpy(set1Data->basePtr(), set1Vec.data(), set1Vec.size(), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(set2Data->basePtr(), set2Vec.data(), set2Vec.size(), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(numSet1Data->basePtr(), numSet1Vec.data(), numSet1Vec.size(), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(numSet2Data->basePtr(), numSet2Vec.data(), numSet2Vec.size(), cudaMemcpyHostToDevice));

        // Seed counts so cross-check must reset every sample through its actual stride.
        RawBufferType initialCounts(nmStrides.x * numSamples, 0);
        for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
        {
            util::ValueAt<int>(initialCounts, nmStrides, long1{sampleIdx}) = 1;
        }
        ASSERT_EQ(cudaSuccess, cudaMemcpy(numMatchesData->basePtr(), initialCounts.data(), initialCounts.size(),
                                          cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemset(matchesData->basePtr(), 0, mchStrides.x * numSamples));
        ASSERT_EQ(cudaSuccess, cudaMemset(distancesData->basePtr(), 0, dStrides.x * numSamples));

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        cvcuda::PairwiseMatcher op(NVCV_BRUTE_FORCE);
        op(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, crossCheck, matchesPerPoint,
           NVCV_NORM_L1);

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        long          mchBufSize = mchStrides.x * numSamples;
        long          nmBufSize  = nmStrides.x * numSamples;
        long          dBufSize   = dStrides.x * numSamples;
        RawBufferType mchTestVec(mchBufSize, 0);
        RawBufferType nmTestVec(nmBufSize, 0);
        RawBufferType dTestVec(dBufSize, 0);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(mchTestVec.data(), matchesData->basePtr(), mchBufSize, cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(nmTestVec.data(), numMatchesData->basePtr(), nmBufSize, cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dTestVec.data(), distancesData->basePtr(), dBufSize, cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

        std::vector<std::tuple<int, int, int, float>>       testIdsDist;
        const std::vector<std::tuple<int, int, int, float>> goldIdsDist{
            {0, 0, 0,  0.f},
            {0, 1, 1, 15.f},
            {1, 0, 0,  0.f},
            {1, 1, 1, 15.f}
        };
        for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
        {
            EXPECT_EQ(util::ValueAt<int>(nmTestVec, nmStrides, long1{sampleIdx}), 2);
        }
        ref::SortOutput(testIdsDist, mchTestVec, nmTestVec, dTestVec, mchStrides, nmStrides, dStrides, numSamples,
                        set1Size, matchesPerPoint, outCapacity);
        EXPECT_EQ(testIdsDist, goldIdsDist);
    };

    for (bool crossCheck : {false, true})
    {
        ASSERT_NO_FATAL_FAILURE(runCrossCheck(crossCheck));
    }
}

TEST(OpPairwiseMatcher, packed_hamming_and_cached_loop_edges)
{
    constexpr int numSamples      = 1;
    constexpr int set1Capacity    = 1;
    constexpr int set2Capacity    = 2;
    constexpr int matchesPerPoint = 1;
    constexpr int outCapacity     = 1;

    auto runCase = [&]<typename SrcT>(SrcT, int numDim)
    {
        nvcv::Tensor set1(
            {
                {numSamples, set1Capacity, numDim},
                "NMD"
        },
            ToDataType<SrcT>());
        nvcv::Tensor set2(
            {
                {numSamples, set2Capacity, numDim},
                "NMD"
        },
            ToDataType<SrcT>());
        nvcv::Tensor matches(
            {
                {numSamples, outCapacity, 2},
                "NMD"
        },
            nvcv::TYPE_S32);
        nvcv::Tensor numMatches({{numSamples}, "N"}, nvcv::TYPE_S32);
        nvcv::Tensor numSet1;
        nvcv::Tensor numSet2;
        nvcv::Tensor distances;

        auto set1Data = set1.exportData<nvcv::TensorDataStridedCuda>();
        auto set2Data = set2.exportData<nvcv::TensorDataStridedCuda>();
        auto mchData  = matches.exportData<nvcv::TensorDataStridedCuda>();
        auto nmData   = numMatches.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(set1Data);
        ASSERT_TRUE(set2Data);
        ASSERT_TRUE(mchData);
        ASSERT_TRUE(nmData);

        long3 set1Strides{set1Data->stride(0), set1Data->stride(1), set1Data->stride(2)};
        long3 set2Strides{set2Data->stride(0), set2Data->stride(1), set2Data->stride(2)};
        long3 mchStrides{mchData->stride(0), mchData->stride(1), mchData->stride(2)};
        long1 nmStrides{nmData->stride(0)};

        RawBufferType set1Vec(set1Strides.x * numSamples, 0);
        RawBufferType set2Vec(set2Strides.x * numSamples, 0);
        for (int i = 0; i < numDim; ++i)
        {
            util::ValueAt<SrcT>(set1Vec, set1Strides, long3{0, 0, i}) = 0;
            util::ValueAt<SrcT>(set2Vec, set2Strides, long3{0, 0, i}) = 0;
            util::ValueAt<SrcT>(set2Vec, set2Strides, long3{0, 1, i}) = 1;
        }

        ASSERT_EQ(cudaSuccess, cudaMemcpy(set1Data->basePtr(), set1Vec.data(), set1Vec.size(), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(set2Data->basePtr(), set2Vec.data(), set2Vec.size(), cudaMemcpyHostToDevice));

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        cvcuda::PairwiseMatcher op(NVCV_BRUTE_FORCE);
        op(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, false, matchesPerPoint,
           NVCV_NORM_HAMMING);

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        RawBufferType matchesVec(mchStrides.x * numSamples, 0);
        RawBufferType numMatchesVec(nmStrides.x * numSamples, 0);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(matchesVec.data(), mchData->basePtr(), matchesVec.size(), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(numMatchesVec.data(), nmData->basePtr(), numMatchesVec.size(), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

        EXPECT_EQ(0, util::ValueAt<int>(matchesVec, mchStrides, long3{0, 0, 0}));
        EXPECT_EQ(0, util::ValueAt<int>(matchesVec, mchStrides, long3{0, 0, 1}));
        EXPECT_EQ(1, util::ValueAt<int>(numMatchesVec, nmStrides, long1{0}));
    };

    runCase(uint8_t{}, 32);
    runCase(uint32_t{}, 8);
}

TEST(OpPairwiseMatcher, packed_and_cached_hamming_full_outputs)
{
    constexpr int numSamples      = 2;
    constexpr int set1Capacity    = 2;
    constexpr int set2Capacity    = 3;
    constexpr int numDim          = 32;
    constexpr int matchesPerPoint = 1;
    constexpr int outCapacity     = set1Capacity * matchesPerPoint;

    auto runCase = [&]<typename SrcT>(SrcT, int descriptorDims)
    {
        nvcv::Tensor set1(
            {
                {numSamples, set1Capacity, descriptorDims},
                "NMD"
        },
            ToDataType<SrcT>());
        nvcv::Tensor set2(
            {
                {numSamples, set2Capacity, descriptorDims},
                "NMD"
        },
            ToDataType<SrcT>());
        nvcv::Tensor matches(
            {
                {numSamples, outCapacity, 2},
                "NMD"
        },
            nvcv::TYPE_S32);
        nvcv::Tensor numMatches({{numSamples}, "N"}, nvcv::TYPE_S32);
        nvcv::Tensor distances(
            {
                {numSamples, outCapacity},
                "NM"
        },
            nvcv::TYPE_F32);
        nvcv::Tensor numSet1;
        nvcv::Tensor numSet2;

        auto set1Data       = set1.exportData<nvcv::TensorDataStridedCuda>();
        auto set2Data       = set2.exportData<nvcv::TensorDataStridedCuda>();
        auto matchesData    = matches.exportData<nvcv::TensorDataStridedCuda>();
        auto numMatchesData = numMatches.exportData<nvcv::TensorDataStridedCuda>();
        auto distancesData  = distances.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(set1Data && set2Data && matchesData && numMatchesData && distancesData);

        long3 set1Strides{set1Data->stride(0), set1Data->stride(1), set1Data->stride(2)};
        long3 set2Strides{set2Data->stride(0), set2Data->stride(1), set2Data->stride(2)};
        long3 mchStrides{matchesData->stride(0), matchesData->stride(1), matchesData->stride(2)};
        long1 nmStrides{numMatchesData->stride(0)};
        long2 dStrides{distancesData->stride(0), distancesData->stride(1)};

        RawBufferType set1Vec(set1Strides.x * numSamples, 0);
        RawBufferType set2Vec(set2Strides.x * numSamples, 0);
        for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
        {
            for (int dimIdx = 0; dimIdx < descriptorDims; ++dimIdx)
            {
                const auto first  = static_cast<SrcT>(1 + sampleIdx * 3 + dimIdx);
                const auto second = static_cast<SrcT>(80 + sampleIdx * 5 + dimIdx);

                util::ValueAt<SrcT>(set1Vec, set1Strides, long3{sampleIdx, 0, dimIdx}) = first;
                util::ValueAt<SrcT>(set1Vec, set1Strides, long3{sampleIdx, 1, dimIdx}) = second;
                util::ValueAt<SrcT>(set2Vec, set2Strides, long3{sampleIdx, 0, dimIdx}) = first;
                util::ValueAt<SrcT>(set2Vec, set2Strides, long3{sampleIdx, 1, dimIdx}) = second;
                util::ValueAt<SrcT>(set2Vec, set2Strides, long3{sampleIdx, 2, dimIdx}) = static_cast<SrcT>(-1);
            }
        }

        ASSERT_EQ(cudaSuccess, cudaMemcpy(set1Data->basePtr(), set1Vec.data(), set1Vec.size(), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(set2Data->basePtr(), set2Vec.data(), set2Vec.size(), cudaMemcpyHostToDevice));

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        cvcuda::PairwiseMatcher op(NVCV_BRUTE_FORCE);
        op(stream, set1, set2, numSet1, numSet2, matches, numMatches, distances, false, matchesPerPoint,
           NVCV_NORM_HAMMING);

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        RawBufferType matchesVec(mchStrides.x * numSamples, 0);
        RawBufferType numMatchesVec(nmStrides.x * numSamples, 0);
        RawBufferType distancesVec(dStrides.x * numSamples, 0);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(matchesVec.data(), matchesData->basePtr(), matchesVec.size(), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(numMatchesVec.data(), numMatchesData->basePtr(), numMatchesVec.size(),
                                          cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(distancesVec.data(), distancesData->basePtr(), distancesVec.size(),
                                          cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

        for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
        {
            EXPECT_EQ(set1Capacity * matchesPerPoint, util::ValueAt<int>(numMatchesVec, nmStrides, long1{sampleIdx}));
            for (int set1Idx = 0; set1Idx < set1Capacity; ++set1Idx)
            {
                EXPECT_EQ(set1Idx, util::ValueAt<int>(matchesVec, mchStrides, long3{sampleIdx, set1Idx, 0}));
                EXPECT_EQ(set1Idx, util::ValueAt<int>(matchesVec, mchStrides, long3{sampleIdx, set1Idx, 1}));
                EXPECT_EQ(0.f, util::ValueAt<float>(distancesVec, dStrides, long2{sampleIdx, set1Idx}));
            }
        }
    };

    runCase(uint8_t{}, numDim);
    runCase(uint32_t{}, 8);
}

static void pairwiseMatcherNegative(const nvcv::Tensor &set1, const nvcv::Tensor &set2, const nvcv::Tensor &numSet1,
                                    const nvcv::Tensor &numSet2, const nvcv::Tensor &matches,
                                    const nvcv::Tensor &numMatches, const nvcv::Tensor &distances, bool crossCheck,
                                    int matchesPerPoint, NVCVNormType normType, NVCVPairwiseMatcherType algoChoice)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::PairwiseMatcher op(algoChoice);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&op, &stream, &set1, &set2, &numSet1, &numSet2, &matches, &numMatches,
                                                &distances, &crossCheck, &matchesPerPoint, &normType] {
                                                   op(stream, set1, set2, numSet1, numSet2, matches, numMatches,
                                                      distances, crossCheck, matchesPerPoint, normType);
                                               }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPairwiseMatcher_Negative, shape_operand_short_circuits)
{
    constexpr int numSamples   = 2;
    constexpr int set1Capacity = 2;
    constexpr int set2Capacity = 3;
    constexpr int numDim       = 4;
    constexpr int maxMatches   = set1Capacity;

    nvcv::Tensor set1(
        {
            {numSamples, set1Capacity, numDim},
            "NMD"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor validSet2(
        {
            {numSamples, set2Capacity, numDim},
            "NMD"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor invalidSet2Width(
        {
            {numSamples, set2Capacity, numDim + 1},
            "NMD"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor validMatches(
        {
            {numSamples, maxMatches, 2},
            "NMD"
    },
        nvcv::TYPE_S32);
    nvcv::Tensor invalidMatchesAttribute(
        {
            {numSamples, maxMatches, 3},
            "NMD"
    },
        nvcv::TYPE_S32);
    nvcv::Tensor nullTensor;

    pairwiseMatcherNegative(set1, invalidSet2Width, nullTensor, nullTensor, validMatches, nullTensor, nullTensor, false,
                            1, NVCV_NORM_L1, NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, validSet2, nullTensor, nullTensor, invalidMatchesAttribute, nullTensor, nullTensor,
                            false, 1, NVCV_NORM_L1, NVCV_BRUTE_FORCE);
}

TEST(OpPairwiseMatcher_Negative, invalid_inputs)
{
    int numSamples      = 2;
    int maxSet1         = 2 + 12;
    int maxSet2         = 2 + 23;
    int numDim          = 3;
    int matchesPerPoint = 1;
    int maxMatches      = maxSet1 * matchesPerPoint;
    // clang-format off

    // null inputs
    nvcv::Tensor nullTensor;

    // invalid set1
    nvcv::Tensor invalidRankSet1({{numSamples}, "N"}, nvcv::TYPE_U8); // invalid rank
    nvcv::Tensor f32Set1({{numSamples, maxSet1, numDim}, "NMD"}, nvcv::TYPE_F32); // different data type
    nvcv::Tensor f16Set1({{numSamples, maxSet1, numDim}, "NMD"}, nvcv::TYPE_F16);

    // invalid set2
    nvcv::Tensor invalidRankSet2({{numSamples}, "N"}, nvcv::TYPE_U8); // invalid rank
    nvcv::Tensor f32Set2({{numSamples, maxSet2, numDim}, "NMD"}, nvcv::TYPE_F32); // different data type
    nvcv::Tensor f16Set2({{numSamples, maxSet2, numDim}, "NMD"}, nvcv::TYPE_F16);
    nvcv::Tensor invalidShapeSet2({{numSamples + 1, maxSet2, numDim}, "NMD"}, nvcv::TYPE_U8); // invalid shape 0

    // invalid numSet
    nvcv::Tensor invalidRankNumSet({{numSamples, maxSet2, numDim}, "NMD"}, nvcv::TYPE_S32);
    nvcv::Tensor invalidShapeCNumSet({{numSamples, 2}, "NC"}, nvcv::TYPE_S32); // the shape C should be 1

    // invalid matches
    nvcv::Tensor invalidRankMatches({{numSamples}, "N"}, nvcv::TYPE_S32);

    // invalid distances
    nvcv::Tensor invalidRankDistances({{numSamples}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor invalidShapeCDistances({{numSamples, maxMatches, 2}, "NMC"}, nvcv::TYPE_F32);

    // valid inputs
    nvcv::Tensor set1({{numSamples, maxSet1, numDim}, "NMD"}, nvcv::TYPE_U8);
    nvcv::Tensor set2({{numSamples, maxSet2, numDim}, "NMD"}, nvcv::TYPE_U8);

    nvcv::Tensor numSet1({{numSamples}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor numSet2({{numSamples}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor matches({{numSamples, maxMatches, 2}, "NMD"}, nvcv::TYPE_S32);
    nvcv::Tensor numMatches({{numSamples}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor distances({{numSamples, maxMatches}, "NM"}, nvcv::TYPE_F32);

    // clang-format on

    // null inputs
    pairwiseMatcherNegative(nullTensor, set2, numSet1, numSet2, matches, numMatches, distances, false, 1, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, nullTensor, numSet1, numSet2, matches, numMatches, distances, false, 1, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, nullTensor, numMatches, distances, false, 1, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, matches, nullTensor, distances, true, 1, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);

    // invalid set1
    pairwiseMatcherNegative(invalidRankSet1, set2, numSet1, numSet2, matches, numMatches, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);

    // invalid set2
    pairwiseMatcherNegative(set1, invalidRankSet2, numSet1, numSet2, matches, numMatches, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, f32Set2, numSet1, numSet2, matches, numMatches, distances, false, 1, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, invalidShapeSet2, numSet1, numSet2, matches, numMatches, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);

    // invalid numSet1
    pairwiseMatcherNegative(set1, set2, invalidRankNumSet, numSet2, matches, numMatches, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, set2, invalidShapeCNumSet, numSet2, matches, numMatches, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);

    // invalid numSet2
    pairwiseMatcherNegative(set1, set2, numSet1, invalidRankNumSet, matches, numMatches, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, set2, numSet1, invalidShapeCNumSet, matches, numMatches, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);

    // invalid matches
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, invalidRankMatches, numMatches, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);

    // invalid numMatches
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, matches, invalidRankNumSet, distances, false, 1, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, matches, invalidShapeCNumSet, distances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);

    // invalid distances
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, matches, numMatches, invalidRankDistances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, matches, numMatches, invalidShapeCDistances, false, 1,
                            NVCV_NORM_L1, NVCV_BRUTE_FORCE);

    // invalid matchesPerPoint
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, matches, numMatches, distances, false, -1, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, matches, numMatches, distances, true, 2, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);

    // invalid type
    pairwiseMatcherNegative(f16Set1, f16Set2, numSet1, numSet2, matches, numMatches, distances, false, 1, NVCV_NORM_L1,
                            NVCV_BRUTE_FORCE);
    pairwiseMatcherNegative(f32Set1, f32Set2, numSet1, numSet2, matches, numMatches, distances, false, 1,
                            NVCV_NORM_HAMMING, NVCV_BRUTE_FORCE);
#ifndef ENABLE_SANITIZER
    pairwiseMatcherNegative(set1, set2, numSet1, numSet2, matches, numMatches, distances, false, 1,
                            static_cast<NVCVNormType>(255), NVCV_BRUTE_FORCE);
#endif
}

#ifndef ENABLE_SANITIZER
TEST(OpPairwiseMatcher_Negative, create_invalid_algo)
{
    NVCVOperatorHandle handle;
    EXPECT_EQ(cvcudaPairwiseMatcherCreate(&handle, static_cast<NVCVPairwiseMatcherType>(255)),
              NVCV_ERROR_INVALID_ARGUMENT);
}
#endif

TEST(OpPairwiseMatcher_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaPairwiseMatcherCreate(nullptr, NVCV_BRUTE_FORCE), NVCV_ERROR_INVALID_ARGUMENT);
}
