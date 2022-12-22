/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace t    = ::testing;
namespace test = nvcv::test;

NVCV_TEST_SUITE_P(ValueTestsTests, test::ValueList{1, 2} * test::ValueList{'c', 'd'});

TEST_P(ValueTestsTests, test)
{
    // For now we're concerned if typed tests will compile.
    // TODO: How to test if the tests were correctly generated?
    int  p1 = GetParamValue<0>();
    char p2 = GetParamValue<1>();

    EXPECT_THAT(p1, t::AnyOf(1, 2));
    EXPECT_THAT(p2, t::AnyOf('c', 'd'));
}

namespace {
struct Foo
{
    Foo(int value_)
        : value(value_)
    {
    }

    int value;

    bool operator<(const Foo &that) const
    {
        return value < that.value;
    }

    bool operator==(const Foo &that) const
    {
        return value == that.value;
    }

    friend std::ostream &operator<<(std::ostream &out, Foo foo)
    {
        return out << foo.value;
    }
};
} // namespace

class ValueTestsConversionTests : public t::TestWithParam<Foo>
{
};

NVCV_INSTANTIATE_TEST_SUITE_P(_, ValueTestsConversionTests, test::Value(Foo{5}));

TEST_P(ValueTestsConversionTests, test)
{
    EXPECT_EQ(5, GetParam().value);
}

class ValueTestsNamedParameterTests
    : public t::TestWithParam<
          std::tuple<test::Param<"withDefault", int>, test::Param<"withExplicitDefault", char, 'c'>>>
{
};

NVCV_INSTANTIATE_TEST_SUITE_P(_, ValueTestsNamedParameterTests, test::ValueList{1, 2} * test::ValueList{'a', 'b'});

TEST_P(ValueTestsNamedParameterTests, test)
{
    int  pi = std::get<0>(GetParam());
    char pc = std::get<1>(GetParam());

    EXPECT_THAT(pi, t::AnyOf(1, 2));
    EXPECT_THAT(pc, t::AnyOf('a', 'b'));
}

class ValueTestsNamedDefaultExplicitParameterTests
    : public t::TestWithParam<
          std::tuple<test::Param<"withDefault", int>, test::Param<"withExplicitDefault", char, 'c'>>>
{
};

NVCV_INSTANTIATE_TEST_SUITE_P(_, ValueTestsNamedDefaultExplicitParameterTests,
                              test::ValueList{1, 2} * test::ValueDefault());

TEST_P(ValueTestsNamedDefaultExplicitParameterTests, test)
{
    int  pi = std::get<0>(GetParam());
    char pc = std::get<1>(GetParam());

    EXPECT_THAT(pi, t::AnyOf(1, 2));
    EXPECT_THAT(pc, 'c');
}

class ValueTestsNamedDefaultImplicitParameterTests
    : public t::TestWithParam<
          std::tuple<test::Param<"withDefault", int>, test::Param<"withImplicitDefault", char, 'c'>>>
{
};

NVCV_INSTANTIATE_TEST_SUITE_P(_, ValueTestsNamedDefaultImplicitParameterTests,
                              test::ValueDefault() * test::ValueList{'a', 'b'});

TEST_P(ValueTestsNamedDefaultImplicitParameterTests, test)
{
    int  pi = std::get<0>(GetParam());
    char pc = std::get<1>(GetParam());

    EXPECT_THAT(pi, 0);
    EXPECT_THAT(pc, t::AnyOf('a', 'b'));
}

class ValueTestsNamedNoDefaultParameterTests
    : public t::TestWithParam<std::tuple<test::Param<"param", Foo>, test::Param<"char", char>>>
{
};

NVCV_INSTANTIATE_TEST_SUITE_P(_, ValueTestsNamedNoDefaultParameterTests, test::Value(123) * test::ValueList{'a', 'b'});

TEST_P(ValueTestsNamedNoDefaultParameterTests, test)
{
    Foo  pf = std::get<0>(GetParam());
    char pc = std::get<1>(GetParam());

    EXPECT_THAT(pf.value, 123);
    EXPECT_THAT(pc, t::AnyOf('a', 'b'));
}

class ValueTestsInferParameterTypesTests
    : public t::TestWithParam<std::tuple<test::Param<"param", Foo>, test::Param<"char", char>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, ValueTestsInferParameterTypesTests,
{
    { Foo{1}, 'r' },
    { Foo{2}, 'o' },
    { Foo{3}, 'd' },
});

TEST_P(ValueTestsInferParameterTypesTests, test)
{
    Foo  pf = std::get<0>(GetParam());
    char pc = std::get<1>(GetParam());

    test::ValueList<Foo, char> params =
    {
        { Foo{1}, 'r' },
        { Foo{2}, 'o' },
        { Foo{3}, 'd' },
    };

    EXPECT_THAT((test::ValueList<Foo,char>{{pf, pc}}), t::IsSubsetOf(params));
}
