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

#include <common/TypeList.hpp>

namespace test {
using namespace nvcv::test::type;
using nvcv::test::ValueList;
} // namespace test

namespace util = nvcv::util;

TEST(TypeListTests, get_type)
{
    ASSERT_TRUE((std::is_same_v<test::GetType<int, 0>, int>));
    ASSERT_TRUE((std::is_same_v<test::GetType<test::Types<int, float, double>, 0>, int>));
    ASSERT_TRUE((std::is_same_v<test::GetType<test::Types<int, float, double>, 1>, float>));
    ASSERT_TRUE((std::is_same_v<test::GetType<test::Types<int, float, double>, 2>, double>));
}

TEST(TypeListTests, create_values)
{
    ASSERT_TRUE((std::is_same_v<test::Values<>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Values<1>, test::Types<test::Value<1>>>));
    ASSERT_TRUE((std::is_same_v<test::Values<1, 2, 3>, test::Types<test::Value<1>, test::Value<2>, test::Value<3>>>));
    ASSERT_TRUE((std::is_same_v<test::Values<1, 2L, 3ul, 'b'>,
                                test::Types<test::Value<1>, test::Value<2L>, test::Value<3ul>, test::Value<'b'>>>));
}

TEST(TypeListTests, get_value)
{
    ASSERT_TRUE((std::is_same_v<decltype(test::GetValue<test::Values<1>, 0>), const int>));
    ASSERT_EQ(1, (test::GetValue<test::Values<1>, 0>));

    ASSERT_TRUE((std::is_same_v<decltype(test::GetValue<test::Values<1, 2, 3>, 1>), const int>));
    ASSERT_EQ(2, (test::GetValue<test::Values<1, 2, 3>, 1>));

    ASSERT_TRUE((std::is_same_v<decltype(test::GetValue<test::Values<1, 'c', 3ul>, 0>), const int>));
    ASSERT_EQ(1, (test::GetValue<test::Values<1, 'c', 3ul>, 0>));

    ASSERT_TRUE((std::is_same_v<decltype(test::GetValue<test::Values<1, 'c', 3ul>, 1>), const char>));
    ASSERT_EQ('c', (test::GetValue<test::Values<1, 'c', 3ul>, 1>));

    // Using 5ul instead of 3ul to avoid gcc-7.0 bug below:
    // Be aware that gcc-7.x has a bug where V's type will be
    // wrong if the value was already instantiated with another type.
    // gcc-8.0 fixes it. clang-6.0.0 doesn't have this bug.
    // Ex: decltype(test::GetValue<test::Values<1,3ul,1>>) == 'const int' instead of 'const unsigned long'
    // Ref: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79092
    ASSERT_TRUE((std::is_same_v<decltype(test::GetValue<test::Values<1, 'c', 5ul>, 2>), const unsigned long>));
    ASSERT_EQ(5ul, (test::GetValue<test::Values<1, 'c', 5ul>, 2>));
}

TEST(TypeListTests, subset)
{
    ASSERT_TRUE((std::is_same_v<test::Subset<test::Types<>, 0, 0>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Subset<test::Types<int>, 0, 1>, test::Types<int>>));
    ASSERT_TRUE((std::is_same_v<test::Subset<test::Types<int, float>, 1, 1>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Subset<test::Types<int, float>, 2, 2>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Subset<test::Types<int, float>, 1, 2>, test::Types<float>>));
    ASSERT_TRUE((std::is_same_v<test::Subset<test::Types<int, float, char, double>, 1, 3>, test::Types<float, char>>));
    ASSERT_TRUE((std::is_same_v<test::Subset<test::Types<int, float, char>, 0, 1>, test::Types<int>>));
    ASSERT_TRUE((std::is_same_v<test::Subset<int, 0, 1>, test::Types<int>>));
    ASSERT_TRUE((std::is_same_v<test::Subset<int, 0, 0>, test::Types<>>));
}

TEST(TypeListTests, concat)
{
    ASSERT_TRUE((std::is_same_v<test::Concat<>, test::Types<>>));

    ASSERT_TRUE((std::is_same_v<test::Concat<test::Types<long, void *, char *>>, test::Types<long, void *, char *>>));

    ASSERT_TRUE((std::is_same_v<test::Concat<test::Types<long, void *, char *>, test::Types<float, char, double>>,
                                test::Types<long, void *, char *, float, char, double>>));

    ASSERT_TRUE((std::is_same_v<test::Concat<test::Types<long, void *, char *>, test::Types<float, char, double>,
                                             test::Types<int *, long *, unsigned>>,
                                test::Types<long, void *, char *, float, char, double, int *, long *, unsigned>>));
}

TEST(TypeListTests, flatten)
{
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<>>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<int>>, test::Types<int>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<int, double>>, test::Types<int, double>>));

    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int>, float>>, test::Types<int, float>>));
    ASSERT_TRUE(
        (std::is_same_v<test::Flatten<test::Types<test::Types<int, double>, float>>, test::Types<int, double, float>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int, double, char>, float>>,
                                test::Types<int, double, char, float>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int, double, char, int>, float>>,
                                test::Types<int, double, char, int, float>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int, double, char, int, float>, float>>,
                                test::Types<int, double, char, int, float, float>>));

    ASSERT_TRUE(
        (std::is_same_v<test::Flatten<test::Types<test::Types<int>, double, float>>, test::Types<int, double, float>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int>, double, char, float>>,
                                test::Types<int, double, char, float>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int>, double, char, int, float>>,
                                test::Types<int, double, char, int, float>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int>, double, char, int, float, float>>,
                                test::Types<int, double, char, int, float, float>>));

    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int, test::Types<double>>, float>>,
                                test::Types<int, double, float>>));
    ASSERT_TRUE((std::is_same_v<test::Flatten<test::Types<test::Types<int, test::Types<double, char>>, float>>,
                                test::Types<int, double, char, float>>));
}

TEST(TypeListTests, combine)
{
    ASSERT_TRUE((std::is_same_v<test::Combine<>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<>, test::Types<>>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<>, test::Types<int>>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<>, test::Types<int, double>>, test::Types<>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<>, test::Types<int, double>, test::Types<>>, test::Types<>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<>, test::Types<>, test::Types<>>, test::Types<>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int, double>, test::Types<>, test::Types<>>, test::Types<>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int>, test::Types<>>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int, double>, test::Types<>>, test::Types<>>));

    ASSERT_TRUE(
        (std::is_same_v<test::Combine<test::Types<int>, test::Types<int>>, test::Types<test::Types<int, int>>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int, double>, test::Types<int>>,
                                test::Types<test::Types<int, int>, test::Types<double, int>>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int>, test::Types<double, char>>,
                                test::Types<test::Types<int, double>, test::Types<int, char>>>));
    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int, double>, test::Types<short, char>>,
                                test::Types<test::Types<int, short>, test::Types<int, char>, test::Types<double, short>,
                                            test::Types<double, char>>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int, double>, test::Types<int>, test::Types<float>>,
                                test::Types<test::Types<int, int, float>, test::Types<double, int, float>>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int, double>, test::Types<int>, test::Types<float, char>>,
                                test::Types<test::Types<int, int, float>, test::Types<int, int, char>,
                                            test::Types<double, int, float>, test::Types<double, int, char>>>));

    ASSERT_TRUE(
        (std::is_same_v<
            test::Combine<test::Types<int, double>, test::Types<int, short>, test::Types<float, char>>,
            test::Types<test::Types<int, int, float>, test::Types<int, int, char>, test::Types<int, short, float>,
                        test::Types<int, short, char>, test::Types<double, int, float>, test::Types<double, int, char>,
                        test::Types<double, short, float>, test::Types<double, short, char>>>));

    ASSERT_TRUE((std::is_same_v<test::Combine<test::Types<int, float>, int>,
                                test::Types<test::Types<int, int>, test::Types<float, int>>>));
    ASSERT_TRUE((std::is_same_v<test::Combine<int, test::Types<int, float>>,
                                test::Types<test::Types<int, int>, test::Types<int, float>>>));
}

TEST(TypeListTests, all_same)
{
    ASSERT_TRUE((test::AllSame::Call<test::Types<int, int>>::value));
    ASSERT_FALSE((test::AllSame::Call<test::Types<bool, int>>::value));

    ASSERT_TRUE((test::AllSame::Call<int, int>::value));
    ASSERT_FALSE((test::AllSame::Call<int, bool>::value));

    ASSERT_TRUE((test::AllSame::Call<int, int, int>::value));
    ASSERT_FALSE((test::AllSame::Call<int, float, int>::value));
    ASSERT_FALSE((test::AllSame::Call<int, int, float>::value));
}

TEST(TypeListTests, exists)
{
    ASSERT_TRUE((test::Exists<int, test::Types<int, char, float>>));
    ASSERT_FALSE((test::Exists<int, test::Types<double, char, float>>));
    ASSERT_FALSE((test::Exists<int, test::Types<>>));
    ASSERT_TRUE((test::Exists<int, test::Types<double, char, float, int>>));
    ASSERT_FALSE((test::Exists<int, test::Types<double>>));
    ASSERT_TRUE((test::Exists<int, test::Types<int>>));
}

TEST(TypeListTests, contained_in)
{
    ASSERT_TRUE((test::ContainedIn<test::Types<test::Types<int, char>>>::Call<test::Types<int, char>>::value));
    ASSERT_FALSE((test::ContainedIn<test::Types<test::Types<int, char>>>::Call<test::Types<int, float>>::value));
    ASSERT_FALSE((test::ContainedIn<test::Types<>>::Call<test::Types<int, float>>::value));
    ASSERT_TRUE((test::ContainedIn<test::Types<test::Types<int, float>, test::Types<char, char>>>::Call<
                 test::Types<int, float>>::value));
    ASSERT_FALSE((test::ContainedIn<test::Types<test::Types<int, float>, test::Types<char, char>>>::Call<
                  test::Types<int, double>>::value));
    ASSERT_TRUE((test::ContainedIn<test::Types<test::Types<int, float>, test::Types<>>>::Call<test::Types<>>::value));
    ASSERT_FALSE(
        (test::ContainedIn<test::Types<test::Types<int, float>, test::Types<int>>>::Call<test::Types<>>::value));
}

TEST(TypeListTests, remove_if)
{
    ASSERT_TRUE((std::is_same_v<test::RemoveIf<test::AllSame, test::Types<>>, test::Types<>>));

    ASSERT_TRUE(
        (std::is_same_v<test::RemoveIf<test::AllSame, test::Types<test::Types<int, int, int>>>, test::Types<>>));

    ASSERT_TRUE((std::is_same_v<test::RemoveIf<test::AllSame, test::Types<test::Types<int, float, int>>>,
                                test::Types<test::Types<int, float, int>>>));

    ASSERT_TRUE((std::is_same_v<
                 test::RemoveIf<test::AllSame, test::Types<test::Types<int, float, char>, test::Types<int, int, int>,
                                                           test::Types<int, int, char>>>,
                 test::Types<test::Types<int, float, char>, test::Types<int, int, char>>>));

    ASSERT_TRUE(
        (std::is_same_v<
            test::RemoveIf<test::AllSame, test::Types<test::Types<int, float, char>, test::Types<int, float, char>,
                                                      test::Types<int, int, char>>>,
            test::Types<test::Types<int, float, char>, test::Types<int, float, char>, test::Types<int, int, char>>>));

    ASSERT_TRUE((std::is_same_v<
                 test::RemoveIf<test::ContainedIn<test::Types<test::Types<int, char>, test::Types<float, int>>>,
                                test::Types<test::Types<char, char>, test::Types<float, int>, test::Types<int, int>>>,
                 test::Types<test::Types<char, char>, test::Types<int, int>>>));
}

TEST(TypeListTests, transform)
{
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Rep<2>, test::Types<int, float>>,
                                test::Types<test::Types<int, int>, test::Types<float, float>>>));
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Rep<1>, test::Types<int, float>>,
                                test::Types<test::Types<int>, test::Types<float>>>));
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Rep<0>, test::Types<int, float>>,
                                test::Types<test::Types<>, test::Types<>>>));
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Rep<2>, test::Types<int>>, test::Types<test::Types<int, int>>>));
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Rep<1>, test::Types<int>>, test::Types<test::Types<int>>>));
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Rep<0>, test::Types<>>, test::Types<>>));
}

// Apply

template<class T>
struct XForm
{
    using type = T;
};

template<>
struct XForm<int>
{
    using type = unsigned;
};

template<>
struct XForm<unsigned>
{
    using type = int;
};

TEST(TypeListTests, apply)
{
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Apply<XForm>, test::Types<int, float, unsigned>>,
                                test::Types<unsigned, float, int>>));
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Apply<XForm>, test::Types<>>, test::Types<>>));
}

TEST(TypeListTests, deref)
{
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Deref<XForm>, test::Types<test::Types<float>>>,
                                test::Types<test::Types<float>>>));
    ASSERT_TRUE((std::is_same_v<test::Transform<test::Deref<XForm>, test::Types<test::Types<int>>>,
                                test::Types<test::Types<unsigned>>>));

    ASSERT_TRUE((std::is_same_v<test::Transform<test::Deref<XForm>, test::Types<test::Types<int, float, unsigned>,
                                                                                test::Types<int, unsigned, unsigned>>>,
                                test::Types<test::Types<unsigned, float, int>, test::Types<unsigned, int, int>>>));
}

TEST(TypeListTests, append)
{
    ASSERT_TRUE((std::is_same_v<test::Append<test::Types<>>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Append<test::Types<>, int>, test::Types<int>>));
    ASSERT_TRUE((std::is_same_v<test::Append<test::Types<int>>, test::Types<int>>));
    ASSERT_TRUE((std::is_same_v<test::Append<test::Types<int>, float>, test::Types<int, float>>));
    ASSERT_TRUE((std::is_same_v<test::Append<test::Types<int>, float, char>, test::Types<int, float, char>>));
}

TEST(TypeListTests, unique)
{
    ASSERT_TRUE((std::is_same_v<test::Unique<test::Types<>>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Unique<test::Types<int, char, float>>, test::Types<int, char, float>>));
    ASSERT_TRUE((std::is_same_v<test::Unique<test::Types<int, int, float>>, test::Types<int, float>>));
    ASSERT_TRUE((std::is_same_v<test::Unique<test::Types<int, char, char, float>>, test::Types<int, char, float>>));
    ASSERT_TRUE((std::is_same_v<test::Unique<test::Types<int, char, float, float>>, test::Types<int, char, float>>));
}

TEST(TypeListTests, contains)
{
    struct Foo
    {
        int x, y;

        bool operator==(Foo that) const
        {
            return x == that.x && y == that.y;
        }
    };

    ASSERT_TRUE(test::Contains(test::Values<Foo{1, 3}>(), Foo{1, 3}));
    ASSERT_TRUE(test::Contains(test::Values<Foo{5, 2}, Foo{3, 5}, Foo{1, 3}>(), Foo{1, 3}));
    ASSERT_FALSE(test::Contains(test::Values<Foo{5, 2}, Foo{3, 5}, Foo{1, 3}>(), Foo{1, 7}));
    ASSERT_FALSE(test::Contains(test::Values<Foo{1, 4}>(), Foo{1, 3}));
    ASSERT_FALSE(test::Contains(test::Values<>(), Foo{1, 3}));

    ASSERT_TRUE(test::Contains(test::Values<1>(), 1));
    ASSERT_FALSE(test::Contains(test::Values<1>(), 2));
    ASSERT_TRUE(test::Contains(test::Values<1, 3, 4, 2>(), 3));
    ASSERT_FALSE(test::Contains(test::Values<1, 3, 4, 2>(), 7));
    ASSERT_FALSE(test::Contains(test::Values<>(), 0));

    ASSERT_TRUE(test::Contains(test::Values<1>(), 1u));
}

TEST(TypeListTests, remove)
{
    ASSERT_TRUE((std::is_same_v<test::Remove<test::Types<int, float, char>, 1>, test::Types<int, char>>));
    ASSERT_TRUE((std::is_same_v<test::Remove<test::Types<int, float, char>, 0, 2>, test::Types<float>>));
    ASSERT_TRUE((std::is_same_v<test::Remove<test::Types<int, char>>, test::Types<int, char>>));
    ASSERT_TRUE((std::is_same_v<test::Remove<test::Types<int>, 0>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Remove<test::Types<int, char>, 0, 1>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::Remove<test::Types<>>, test::Types<>>));
}

TEST(TypeListTests, get_size)
{
    ASSERT_EQ(0, (test::GetSize<test::Types<>>));
    ASSERT_EQ(1, (test::GetSize<test::Types<int>>));
    ASSERT_EQ(2, (test::GetSize<test::Types<int, int>>));
    ASSERT_EQ(2, (test::GetSize<test::Types<int, void>>));
}

TEST(TypeListTests, zip)
{
    ASSERT_TRUE((std::is_same_v<test::Zip<test::Types<int, float, char>, test::Values<1, 2, 3>>,
                                test::Types<test::Types<int, test::Value<1>>, test::Types<float, test::Value<2>>,
                                            test::Types<char, test::Value<3>>>>));
    ASSERT_TRUE(
        (std::is_same_v<test::Zip<test::Types<int>, test::Values<1>>, test::Types<test::Types<int, test::Value<1>>>>));
    ASSERT_TRUE((std::is_same_v<test::Zip<test::Types<int>>, test::Types<test::Types<int>>>));
    ASSERT_TRUE((std::is_same_v<test::Zip<>, test::Types<>>));
}

TEST(TypeListTests, set_difference)
{
    ASSERT_TRUE((std::is_same_v<test::SetDifference<test::Types<int, float, char>, test::Types<float>>,
                                test::Types<int, char>>));
    ASSERT_TRUE((std::is_same_v<test::SetDifference<test::Types<int, float, char>, test::Types<unsigned>>,
                                test::Types<int, float, char>>));
    ASSERT_TRUE((std::is_same_v<test::SetDifference<test::Types<int, float, char>, test::Types<int, unsigned>>,
                                test::Types<float, char>>));
    ASSERT_TRUE((std::is_same_v<test::SetDifference<test::Types<int, float, char>, test::Types<int, char>>,
                                test::Types<float>>));
    ASSERT_TRUE(
        (std::is_same_v<test::SetDifference<test::Types<int, float, char>, test::Types<int, char, double, char *>>,
                        test::Types<float>>));
    ASSERT_TRUE((std::is_same_v<test::SetDifference<test::Types<int>, test::Types<int>>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::SetDifference<test::Types<int>, test::Types<float>>, test::Types<int>>));
    ASSERT_TRUE((std::is_same_v<test::SetDifference<test::Types<>, test::Types<float>>, test::Types<>>));
    ASSERT_TRUE((std::is_same_v<test::SetDifference<test::Types<>, test::Types<>>, test::Types<>>));
}

TEST(TypeListTests, empty_to_value_list)
{
    EXPECT_EQ(test::ValueList<>(), test::ToValueList<test::Types<>>());
}

TEST(TypeListTests, heterogeneous_but_implicitly_convertible_types_to_value_list)
{
    enum A
    {
        a1,
        a2
    };

    enum B
    {
        b1,
        b2
    };

    using V = test::Values<4ul, 2, a1, b2, 'c'>;

    test::ValueList<unsigned long> gold
        = {(unsigned long)4ul, (unsigned long)2, (unsigned long)a1, (unsigned long)b2, (unsigned long)'c'};

    EXPECT_EQ(gold, test::ToValueList<V>());
}
