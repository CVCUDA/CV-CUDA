/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TEST_COMMON_UTILS_HPP
#define NVCV_TEST_COMMON_UTILS_HPP

#include <random>
#include <vector>

using RandEng = std::default_random_engine;

template<typename T>
using RandInt = std::uniform_int_distribution<T>;

template<typename T>
using RandFlt = std::uniform_real_distribution<T>;

//--------------------------------------------------------------------------------------------------------------------//
template<typename T>
void generateRandVec(T *dst, size_t size, RandEng &eng);

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
inline void generateRandVec(std::vector<T> &dst, RandEng &eng)
{
    generateRandVec<T>(dst.data(), dst.size(), eng);
}

//--------------------------------------------------------------------------------------------------------------------//
template<typename T>
void generateRandTestRGB(T *dst, size_t size, RandEng &eng, bool rgba = false, bool bga = false);

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
inline void generateRandTestRGB(std::vector<T> &dst, RandEng &eng, bool rgba = false, bool bga = false)
{
    generateRandTestRGB<T>(dst.data(), dst.size(), eng, rgba, bga);
}

//--------------------------------------------------------------------------------------------------------------------//
template<typename T>
void generateAllRGB(T *dst, uint wdth, uint hght, uint num, bool rgba = false, bool bga = false);

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
inline void generateAllRGB(std::vector<T> &dst, uint wdth, uint hght, uint num, bool rgba = false, bool bga = false)
{
    ASSERT_GE(dst.size(), (size_t)num * (size_t)hght * (size_t)wdth * (size_t)(3 + rgba));
    generateAllRGB<T>(dst.data(), wdth, hght, num, rgba, bga);
}

//--------------------------------------------------------------------------------------------------------------------//
template<typename T, bool FullRange>
void generateRandHSV(T *dst, size_t size, RandEng &eng, double minHueMult = 0.0, double maxHueMult = 1.0);

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T, bool FullRange>
inline void generateRandHSV(std::vector<T> &dst, RandEng &eng, double minHueMult = 0.0, double maxHueMult = 1.0)
{
    generateRandHSV<T, FullRange>(dst.data(), dst.size(), eng, minHueMult, maxHueMult);
}

//--------------------------------------------------------------------------------------------------------------------//
template<typename T, bool FullRange>
void generateAllHSV(T *dst, uint wdth, uint hght, uint num);

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T, bool FullRange>
inline void generateAllHSV(std::vector<T> &dst, uint wdth, uint hght, uint num)
{
    ASSERT_EQ(dst.size() % 3, 0);
    ASSERT_GE(dst.size(), (size_t)num * (size_t)hght * (size_t)wdth * (size_t)3);
    generateAllHSV<T, FullRange>(dst.data(), wdth, hght, num);
}

//--------------------------------------------------------------------------------------------------------------------//

// NOTE: the "do {" ... "} while (false)" statements in the macros below add scope context to multi-statement macro
//       expansions so they can be nested inside non-scoped statements (e.g., "if", "for", etc. statements that don't
//       have braces) and still be treated like a single statement that can be terminated with a semicolon (";").
//       For example, the "do-while" construct allows for:
//
//           if (<condition>)
//               EXPECT_NEAR_VEC_CNT(vec1, vec2, maxDiff, maxCnt, passes);
//           else
//               std::cout << "Test condition not satisfied.\n";
//
//       without the problems that would otherwise occur from multi-statement macro expansion.
//--------------------------------------------------------------------------------------------------------------------//
#define EXPECT_NEAR_ARR_CNT(data1, data2, size, maxDiff, maxCnt, passes)                                        \
    do                                                                                                          \
    {                                                                                                           \
        uint cnt = 0;                                                                                           \
        for (size_t i = 0; i < size && cnt < maxCnt; i++)                                                       \
        {                                                                                                       \
            EXPECT_NEAR(data1[i], data2[i], maxDiff) << "At index " << i << " (error count = " << ++cnt << ")"; \
        }                                                                                                       \
        passes = (cnt == 0);                                                                                    \
    }                                                                                                           \
    while (false)

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define EXPECT_NEAR_VEC_CNT(vec1, vec2, maxDiff, maxCnt, passes)                             \
    do                                                                                       \
    {                                                                                        \
        ASSERT_EQ(vec1.size(), vec2.size());                                                 \
        EXPECT_NEAR_ARR_CNT(vec1.data(), vec2.data(), vec1.size(), maxDiff, maxCnt, passes); \
    }                                                                                        \
    while (false)

//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
// clang-format off
#define EXPECT_NEAR_HSV_ARR_CNT(data1, data2, size, range, maxDiff, maxCnt, passes)                                   \
    do                                                                                                                \
    {                                                                                                                 \
        ASSERT_EQ(size % 3, 0);                                                                                       \
        uint   cnt  = 0;                                                                                              \
        double half = range * 0.5;                                                                                    \
        for (size_t i = 0; i < size && cnt < maxCnt; i += 3)                                                          \
        {                                                                                                             \
            double val1 = static_cast<double>(data1[i]);                                                              \
            double val2 = static_cast<double>(data2[i]);                                                              \
            if (val2 >= val1 && val2 - val1 > half)                                                                   \
                EXPECT_NEAR(data1[i] + range, data2[i], maxDiff) << "At index " << i                                  \
                                                                 << " (error count = " << ++cnt << ")";               \
            else if (val1 - val2 > half)                                                                              \
                EXPECT_NEAR(data1[i], data2[i] + range, maxDiff) << "At index " << i                                  \
                                                                 << " (error count = " << ++cnt << ")";               \
            else                                                                                                      \
                EXPECT_NEAR(data1[i], data2[i], maxDiff) << "At index " << i   << " (error count = " << ++cnt << ")"; \
            EXPECT_NEAR(data1[i+1], data2[i+1], maxDiff) << "At index " << i+1 << " (error count = " << ++cnt << ")"; \
            EXPECT_NEAR(data1[i+2], data2[i+2], maxDiff) << "At index " << i+2 << " (error count = " << ++cnt << ")"; \
        }                                                                                                             \
        passes = (cnt == 0);                                                                                          \
    }                                                                                                                 \
    while (false)

// clang-format on
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define EXPECT_NEAR_HSV_VEC_CNT(vec1, vec2, range, maxDiff, maxCnt, passes)                             \
    do                                                                                                  \
    {                                                                                                   \
        ASSERT_EQ(vec1.size(), vec2.size());                                                            \
        EXPECT_NEAR_HSV_ARR_CNT(vec1.data(), vec2.data(), vec1.size(), range, maxDiff, maxCnt, passes); \
    }                                                                                                   \
    while (false)

//--------------------------------------------------------------------------------------------------------------------//

/*
FYI: gtest expands the following macro statement:

    EXPECT_NEAR(data1[i], data2[i], maxDiff) << "At index " << i << " (error count = " << ++cnt << ")";

to:

    switch (0)
    case 0:
    default:
        if (const ::testing::AssertionResult gtest_ar
                = ::testing::internal::DoubleNearPredFormat("refVec.data()[i]", "dstVec.data()[i]", "maxDiff",
                                                             refVec.data()[i] ,  dstVec.data()[i] ,  maxDiff)) ;
        else
            ::testing::internal::AssertHelper(::testing::TestPartResult::kNonFatalFailure,
                                              __FILE__, __LINE__, gtest_ar.failure_message())
                = ::testing::Message() << "At index " << i << " (error count = " << ++cnt << ")";

The switch statement is to disambiguate the else clause if the macro is expanded in a nested if without braces.
*/

#endif // NVCV_TEST_COMMON_UTILS_HPP
