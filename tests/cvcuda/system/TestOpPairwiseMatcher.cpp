/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            dist += std::bitset<sizeof(ST) * 8>(p1 ^ p2).count();
        }
    }
    else if (normType == NVCV_NORM_L1)
    {
        dist += absdiff(p1, p2);
    }
    else if (normType == NVCV_NORM_L2)
    {
        dist += std::pow(absdiff(p1, p2), 2);
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

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        int mchIdx = 0;

        for (int set1Idx = 0; set1Idx < set1Size; set1Idx++)
        {
            for (int set2Idx = 0; set2Idx < set2Size; set2Idx++)
            {
                float dist = 0.f;

                for (int coordIdx = 0; coordIdx < numDim; coordIdx++)
                {
                    ST p1 = util::ValueAt<ST>(set1Vec, set1Strides, long3{sampleIdx, set1Idx, coordIdx});
                    ST p2 = util::ValueAt<ST>(set2Vec, set2Strides, long3{sampleIdx, set2Idx, coordIdx});

                    ComputeDistance(dist, p1, p2, normType);
                }
                if (normType == NVCV_NORM_L2)
                {
                    dist = std::sqrt(dist);
                }

                distIdx[set2Idx] = std::tie(dist, set2Idx);
            }

            std::sort(distIdx.begin(), distIdx.end());

            if (crossCheck)
            {
                int set2Idx = std::get<1>(distIdx[0]);

                for (int cck1Idx = 0; cck1Idx < set1Size; cck1Idx++)
                {
                    float dist = 0.f;

                    for (int coordIdx = 0; coordIdx < numDim; coordIdx++)
                    {
                        ST p1 = util::ValueAt<ST>(set1Vec, set1Strides, long3{sampleIdx, cck1Idx, coordIdx});
                        ST p2 = util::ValueAt<ST>(set2Vec, set2Strides, long3{sampleIdx, set2Idx, coordIdx});

                        ComputeDistance(dist, p1, p2, normType);
                    }
                    if (normType == NVCV_NORM_L2)
                    {
                        dist = std::sqrt(dist);
                    }

                    cckDistIdx[cck1Idx] = std::tie(dist, cck1Idx);
                }

                std::sort(cckDistIdx.begin(), cckDistIdx.end());

                if (std::get<1>(cckDistIdx[0]) == set1Idx)
                {
                    util::ValueAt<int>(mchVec, mchStrides, long3{sampleIdx, mchIdx, 0}) = set1Idx;
                    util::ValueAt<int>(mchVec, mchStrides, long3{sampleIdx, mchIdx, 1}) = std::get<1>(distIdx[0]);
                    if (dStrides.x > 0)
                    {
                        util::ValueAt<float>(dVec, dStrides, long2{sampleIdx, mchIdx}) = std::get<0>(distIdx[0]);
                    }

                    mchIdx++;
                    if (nmStrides.x > 0)
                    {
                        util::ValueAt<int>(nmVec, nmStrides, long1{sampleIdx}) = mchIdx;
                    }
                }
            }
            else
            {
                for (int m = 0; m < matchesPerPoint; m++)
                {
                    util::ValueAt<int>(mchVec, mchStrides, long3{sampleIdx, mchIdx, 0}) = set1Idx;
                    util::ValueAt<int>(mchVec, mchStrides, long3{sampleIdx, mchIdx, 1}) = std::get<1>(distIdx[m]);
                    if (dStrides.x > 0)
                    {
                        util::ValueAt<float>(dVec, dStrides, long2{sampleIdx, mchIdx}) = std::get<0>(distIdx[m]);
                    }

                    mchIdx++;
                    if (nmStrides.x > 0)
                    {
                        util::ValueAt<int>(nmVec, nmStrides, long1{sampleIdx}) = mchIdx;
                    }
                }
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

    std::sort(outIdsDist.begin(), outIdsDist.end());
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
    NVCV_TEST_ROW(2, 73, 62, 8, 1, true, false, NVCV_BRUTE_FORCE, NVCV_NORM_L2, float)
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
    long1 nmStrides = (numMatches) ? long1{nmData->stride(0)} : long1{0};
    long2 dStrides  = (distances) ? long2{dData->stride(0), dData->stride(1)} : long2{0, 0};

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

static void pairwiseMatcherNegative(nvcv::Tensor &set1, nvcv::Tensor &set2, nvcv::Tensor &numSet1,
                                    nvcv::Tensor &numSet2, nvcv::Tensor &matches, nvcv::Tensor &numMatches,
                                    nvcv::Tensor &distances, bool crossCheck, int matchesPerPoint,
                                    NVCVNormType normType, NVCVPairwiseMatcherType algoChoice)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::PairwiseMatcher op(algoChoice);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&] {
                                                   op(stream, set1, set2, numSet1, numSet2, matches, numMatches,
                                                      distances, crossCheck, matchesPerPoint, normType);
                                               }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
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
