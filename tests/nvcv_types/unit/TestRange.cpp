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

#include <common/TypedTests.hpp>
#include <nvcv/util/Ranges.hpp>

#include <array>
#include <list>
#include <map>
#include <vector>

namespace ttest = nvcv::test::type;
namespace util  = nvcv::util;

namespace {

constexpr int DEFAULT_RANGE_SIZE = 5;

class RangeMember
{
public:
    explicit RangeMember(int len = DEFAULT_RANGE_SIZE)
        : m_length(len)
    {
    }

    int *begin() const
    {
        return nullptr;
    }

    int *end() const
    {
        return begin() + m_length;
    }

private:
    int m_length;
};

class NotRangeBegin
{
public:
    explicit NotRangeBegin(int = DEFAULT_RANGE_SIZE) {}

    int *begin() const
    {
        return nullptr;
    }
};

class NotRangeEnd
{
public:
    explicit NotRangeEnd(int = DEFAULT_RANGE_SIZE) {}

    int *end() const
    {
        return nullptr;
    }
};

class NotRange
{
public:
    explicit NotRange(int = DEFAULT_RANGE_SIZE) {}
};

namespace range {
class RangeGlobal
{
public:
    explicit RangeGlobal(int len = DEFAULT_RANGE_SIZE)
        : m_length(len)
    {
    }

    int *my_begin() const
    {
        return nullptr;
    }

    int *my_end() const
    {
        return my_begin() + m_length;
    }

private:
    int m_length;
};

int *begin(const RangeGlobal &r)
{
    return r.my_begin();
}

int *end(const RangeGlobal &r)
{
    return r.my_end();
}
} // namespace range

} // namespace

// clang-format off
NVCV_TYPED_TEST_SUITE(RangeTest,
                      ttest::Types<RangeMember, range::RangeGlobal, int[DEFAULT_RANGE_SIZE]>);

// clang-format on

TYPED_TEST(RangeTest, begin)
{
    TypeParam r;
    using std::begin;

    ASSERT_EQ(begin(r), util::ranges::Begin(r));
}

TYPED_TEST(RangeTest, end)
{
    TypeParam r{};
    using std::end;
    ASSERT_EQ(end(r), util::ranges::End(r));
}

TYPED_TEST(RangeTest, size)
{
    TypeParam r{};
    using std::begin;
    using std::end;
    ASSERT_EQ(std::distance(begin(r), end(r)), util::ranges::Size(r));
}

TYPED_TEST(RangeTest, data)
{
    TypeParam r{};
    using std::begin;
    ASSERT_EQ(&*begin(r), util::ranges::Data(r));
}

NVCV_TYPED_TEST_SUITE(RangePositiveTest, ttest::Types<std::vector<float>, RangeMember, range::RangeGlobal, int[10]>);

TYPED_TEST(RangePositiveTest, works)
{
    ASSERT_TRUE(util::ranges::IsRange<TypeParam>);
}

NVCV_TYPED_TEST_SUITE(RangeNegativeTest, ttest::Types<int, NotRange, NotRangeBegin, NotRangeEnd>);

TYPED_TEST(RangeNegativeTest, works)
{
    ASSERT_FALSE(util::ranges::IsRange<TypeParam>);
}

NVCV_TYPED_TEST_SUITE(
    RangeValueTest,
    ttest::Types<ttest::Types<const std::vector<float>, const float>, ttest::Types<std::vector<float>, float>,
                 ttest::Types<RangeMember, int>, ttest::Types<range::RangeGlobal, int>>);

TYPED_TEST(RangeValueTest, works)
{
    using Range = ttest::GetType<TypeParam, 0>;
    using Value = ttest::GetType<TypeParam, 1>;

    ASSERT_TRUE((std::is_same_v<util::ranges::RangeValue<Range>, Value>));
}

NVCV_TYPED_TEST_SUITE(RangeRandomAccessPositiveTest, ttest::Types<std::vector<float>, std::array<float, 5>, int[10]>);

TYPED_TEST(RangeRandomAccessPositiveTest, works)
{
    ASSERT_TRUE(util::ranges::IsRandomAccessRange<TypeParam>);
}

#if 0
NVCV_TYPED_TEST_SUITE(RangeRandomAccessNegativeTest, ttest::Types<std::list<float>, std::map<float,int>, int, NotRange, NotRangeBegin, NotRangeEnd>);

TYPED_TEST(RangeRandomAccessNegativeTest, works)
{
    ASSERT_FALSE(util::ranges::IsRandomAccessRange<TypeParam>);
}

#endif
