/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/ValueTests.hpp>
#include <nvcv/util/Version.hpp>

namespace t    = ::testing;
namespace util = nvcv::util;
namespace test = nvcv::test;

#define MAKE_VERSION(major, minor, patch, tweak) ((major)*1000000 + (minor)*10000 + (patch)*100 + (tweak))

class VersionTests
    : public t::TestWithParam<std::tuple<test::Param<"code", uint32_t>,             // 0
                                         test::Param<"major", int>,                 // 1
                                         test::Param<"minor", int>,                 // 2
                                         test::Param<"patch", int>,                 // 3
                                         test::Param<"tweak", int>,                 // 4
                                         test::Param<"ctor_result", NVCVStatus>,    // 5
                                         test::Param<"values_result", NVCVStatus>>> // 6
{
protected:
    uint32_t paramCode() const
    {
        return m_paramCode;
    }

    int paramMajor() const
    {
        return m_paramMajor;
    }

    int paramMinor() const
    {
        return m_paramMinor;
    }

    int paramPatch() const
    {
        return m_paramPatch;
    }

    int paramTweak() const
    {
        return m_paramTweak;
    }

    NVCVStatus goldCtorResult() const
    {
        return m_goldCtorResult;
    }

    NVCVStatus goldValuesResult() const
    {
        return m_goldValuesResult;
    }

private:
    uint32_t   m_paramCode        = ::nvcv::test::ParamValue(std::get<0>(GetParam()));
    int        m_paramMajor       = ::nvcv::test::ParamValue(std::get<1>(GetParam()));
    int        m_paramMinor       = ::nvcv::test::ParamValue(std::get<2>(GetParam()));
    int        m_paramPatch       = ::nvcv::test::ParamValue(std::get<3>(GetParam()));
    int        m_paramTweak       = ::nvcv::test::ParamValue(std::get<4>(GetParam()));
    NVCVStatus m_goldCtorResult   = ::nvcv::test::ParamValue(std::get<5>(GetParam()));
    NVCVStatus m_goldValuesResult = ::nvcv::test::ParamValue(std::get<6>(GetParam()));
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive, VersionTests,
                              test::ValueList<uint32_t, int, int, int, int>
                              {
                                  { MAKE_VERSION(0,  0,  0,  0),  0,  0,  0,  0},
                                  { MAKE_VERSION(1,  0,  0,  0),  1,  0,  0,  0},
                                  { MAKE_VERSION(0,  1,  0,  0),  0,  1,  0,  0},
                                  { MAKE_VERSION(0,  0,  1,  0),  0,  0,  1,  0},
                                  { MAKE_VERSION(0,  0,  0,  1),  0,  0,  0,  1},
                                  { MAKE_VERSION(1,  2,  3,  4),  1,  2,  3,  4},
                                  { MAKE_VERSION(3,  2,  1,  0),  3,  2,  1,  0},
                                  {MAKE_VERSION(99,  0,  0,  0), 99,  0,  0,  0},
                                  { MAKE_VERSION(0, 99,  0,  0),  0, 99,  0,  0},
                                  { MAKE_VERSION(0,  0, 99,  0),  0,  0, 99,  0},
                                  { MAKE_VERSION(0,  0,  0, 99),  0,  0,  0, 99},
                                  {MAKE_VERSION(99, 99, 99, 99), 99, 99, 99, 99},
                                  {MAKE_VERSION(100, 99, 99, 99), 100, 99, 99, 99},
                                  {MAKE_VERSION(500, 99, 99, 99), 500, 99, 99, 99},
                                } * NVCV_SUCCESS * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_ctor_fail, VersionTests,
                              test::ValueList<uint32_t, int, int, int, int>
                              {
                                  { MAKE_VERSION(-1,   0,   0,   0),  -1,   0,   0,   0},
                                  {  MAKE_VERSION(0,  -1,   0,   0),   0,  -1,   0,   0},
                                  {  MAKE_VERSION(0,   0,  -1,   0),   0,   0,  -1,   0},
                                  {  MAKE_VERSION(0,   0,   0,  -1),   0,   0,   0,  -1},
                                  { MAKE_VERSION(-4,  -2,  -9,   4),  -4,  -2,  -9,   4},
                                  {MAKE_VERSION(105, 102, 120, 150), 105, 102, 120, 150},
                              } * NVCV_ERROR_INVALID_ARGUMENT * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_values_fail, VersionTests,
                              test::ValueList<uint32_t, int, int, int, int>
                              {
                                  {MAKE_VERSION(0, 100,   0,   0), 0, 100,   0,   0},
                                  {MAKE_VERSION(0,   0, 100,   0), 0,   0, 100,   0},
                                  {MAKE_VERSION(0,   0,   0, 100), 0,   0,   0, 100},
                              } * NVCV_SUCCESS * NVCV_ERROR_INVALID_ARGUMENT);

// clang-format on

TEST_P(VersionTests, code_to_version)
{
    std::unique_ptr<util::Version> ver;

    // Code is always valid because any overflow will just make at most the major version
    // larger than expected. We can't check for that, it's still a valid version.
    ASSERT_NO_THROW(ver = std::make_unique<util::Version>(paramCode()));

    if (goldValuesResult() == NVCV_SUCCESS)
    {
        ASSERT_NE(nullptr, ver);

        EXPECT_EQ(ver->major(), paramMajor());
        EXPECT_EQ(ver->minor(), paramMinor());
        EXPECT_EQ(ver->patch(), paramPatch());
        EXPECT_EQ(ver->tweak(), paramTweak());
    }
    else if (ver && goldCtorResult() == NVCV_SUCCESS)
    {
        EXPECT_TRUE(ver->major() != paramMajor() || ver->minor() != paramMinor() || ver->patch() != paramPatch()
                    || ver->tweak() != paramTweak());
    }
}

TEST_P(VersionTests, version_to_code)
{
    std::unique_ptr<util::Version> ver;
    switch (goldValuesResult())
    {
    case NVCV_SUCCESS:
        ASSERT_NO_THROW(ver = std::make_unique<util::Version>(paramMajor(), paramMinor(), paramPatch(), paramTweak()));
        break;

    case NVCV_ERROR_INVALID_ARGUMENT:
        ASSERT_THROW(ver = std::make_unique<util::Version>(paramMajor(), paramMinor(), paramPatch(), paramTweak()),
                     std::invalid_argument);
        break;
    default:
        assert(false);
    }

    if (ver)
    {
        EXPECT_EQ(paramCode(), ver->code());
    }
}

class VersionStringTests
    : public t::TestWithParam<std::tuple<test::Param<"major", int>,            // 0
                                         test::Param<"minor", int>,            // 1
                                         test::Param<"patch", int>,            // 2
                                         test::Param<"tweak", int>,            // 3
                                         test::Param<"result", const char *>>> // 4
{
protected:
    int paramMajor() const
    {
        return m_paramMajor;
    }

    int paramMinor() const
    {
        return m_paramMinor;
    }

    int paramPatch() const
    {
        return m_paramPatch;
    }

    int paramTweak() const
    {
        return m_paramTweak;
    }

    const char *goldResult() const
    {
        return m_goldResult;
    }

private:
    int         m_paramMajor = ::nvcv::test::ParamValue(std::get<0>(GetParam()));
    int         m_paramMinor = ::nvcv::test::ParamValue(std::get<1>(GetParam()));
    int         m_paramPatch = ::nvcv::test::ParamValue(std::get<2>(GetParam()));
    int         m_paramTweak = ::nvcv::test::ParamValue(std::get<3>(GetParam()));
    const char *m_goldResult = ::nvcv::test::ParamValue(std::get<4>(GetParam()));
};

NVCV_INSTANTIATE_TEST_SUITE_P(Positive, VersionStringTests,
                              test::ValueList<int, int, int, int, const char *>{
                                  { 0,  0,  0,  0,       "v0.0.0"},
                                  { 1,  2,  3,  4,     "v1.2.3.4"},
                                  { 1,  2,  3,  0,       "v1.2.3"},
                                  {99, 99, 99, 99, "v99.99.99.99"},
                                  {99, 99, 99,  0,    "v99.99.99"}
});

TEST_P(VersionStringTests, test)
{
    util::Version ver(paramMajor(), paramMinor(), paramPatch(), paramTweak());

    std::ostringstream ss;
    ss << ver;

    EXPECT_STREQ(goldResult(), ss.str().c_str());
}

class VersionComparisonTests
    : public t::TestWithParam<std::tuple<test::Param<"lhs", util::Version, util::Version{0, 0, 0, 0}>, // 0
                                         test::Param<"rhs", util::Version, util::Version{0, 0, 0, 0}>, // 1
                                         test::Param<"result", int>>>                                  // 2
{
protected:
    const util::Version &paramLHS() const
    {
        return m_paramLHS;
    }

    const util::Version &paramRHS() const
    {
        return m_paramRHS;
    }

    int goldResult() const
    {
        return m_goldResult;
    }

private:
    util::Version m_paramLHS   = ::nvcv::test::ParamValue(std::get<0>(GetParam()));
    util::Version m_paramRHS   = ::nvcv::test::ParamValue(std::get<1>(GetParam()));
    int           m_goldResult = ::nvcv::test::ParamValue(std::get<2>(GetParam()));
};

NVCV_INSTANTIATE_TEST_SUITE_P(Positive, VersionComparisonTests,
                              test::ValueList<util::Version, util::Version, int>{
                                  {util::Version{1, 0, 0}, util::Version{0, 1, 0}, 1},
                                  {util::Version{1, 2, 3}, util::Version{1, 2, 3}, 0},
                                  {util::Version{1, 0, 0}, util::Version{0, 0, 1}, 1},
});

TEST_P(VersionComparisonTests, lower_than)
{
    EXPECT_EQ(goldResult() < 0, paramLHS() < paramRHS());
}

TEST_P(VersionComparisonTests, lower_equal_than)
{
    EXPECT_EQ(goldResult() <= 0, paramLHS() <= paramRHS());
}

TEST_P(VersionComparisonTests, equal_than)
{
    EXPECT_EQ(goldResult() == 0, paramLHS() == paramRHS());
}

TEST_P(VersionComparisonTests, not_equal_than)
{
    EXPECT_EQ(goldResult() != 0, paramLHS() != paramRHS());
}

TEST_P(VersionComparisonTests, greater_equal_than)
{
    EXPECT_EQ(goldResult() >= 0, paramLHS() >= paramRHS());
}

TEST_P(VersionComparisonTests, greater_than)
{
    EXPECT_EQ(goldResult() > 0, paramLHS() > paramRHS());
}
