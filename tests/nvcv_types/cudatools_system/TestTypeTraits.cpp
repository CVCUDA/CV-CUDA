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

#include <common/TypedTests.hpp>    // for NVCV_TYPED_TEST_SUITE_F, etc.
#include <common/ValueTests.hpp>    // for StringLiteral
#include <nvcv/cuda/TypeTraits.hpp> // the object of this test

#include <limits> // for std::numeric_limits, etc.

namespace t    = ::testing;
namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

template<int N>
using TStr = typename test::StringLiteral<N>;

// --------------------------- Testing TypeTraits ------------------------------

template<typename T>
class TypeTraitsBaseTest : public t::Test
{
public:
    using Type = T;
};

template<typename T>
class TypeTraitsSupportedBaseTest : public TypeTraitsBaseTest<T>
{
};

using TypeTraitsSupportedBaseTypes = t::Types<unsigned char, signed char, unsigned short, short, unsigned int, int,
                                              unsigned long, long, unsigned long long, long long, float, double>;

TYPED_TEST_SUITE(TypeTraitsSupportedBaseTest, TypeTraitsSupportedBaseTypes);

TYPED_TEST(TypeTraitsSupportedBaseTest, correct_type_traits)
{
    using TypeTraitsT = cuda::TypeTraits<typename TestFixture::Type>;

    EXPECT_TRUE((std::is_same_v<typename TestFixture::Type, typename TypeTraitsT::base_type>));

    EXPECT_FALSE(cuda::IsCompound<typename TestFixture::Type>);

    EXPECT_EQ(TypeTraitsT::components, cuda::NumComponents<typename TestFixture::Type>);
    EXPECT_EQ(TypeTraitsT::elements, cuda::NumElements<typename TestFixture::Type>);

    EXPECT_EQ(TypeTraitsT::components, 0);

    EXPECT_EQ(TypeTraitsT::elements, 1);

    EXPECT_EQ(TypeTraitsT::min, std::numeric_limits<typename TestFixture::Type>::min());
    EXPECT_EQ(TypeTraitsT::max, std::numeric_limits<typename TestFixture::Type>::max());
}

// ------------------------- Testing HasTypeTraits -----------------------------

template<typename T>
class HasTypeTraitsUnsupportedTest : public TypeTraitsBaseTest<T>
{
};

typedef struct _float5
{
    float a, b, c, d, e;
} float5;

using UnsupportedBaseTypes = t::Types<void, long double, float5>;

TYPED_TEST_SUITE(HasTypeTraitsUnsupportedTest, UnsupportedBaseTypes);

TYPED_TEST(HasTypeTraitsUnsupportedTest, is_false)
{
    EXPECT_FALSE(cuda::HasTypeTraits<typename TestFixture::Type>);
}

TEST(HasTypeTraitsWithTwoUnsupportedTypesTest, is_false)
{
    EXPECT_FALSE((cuda::HasTypeTraits<long double, float5>));
}

template<typename T>
class HasTypeTraitsSupportedTest : public TypeTraitsBaseTest<T>
{
};

using SupportedBaseTypes = t::Types<int, float3>;

TYPED_TEST_SUITE(HasTypeTraitsSupportedTest, SupportedBaseTypes);

TYPED_TEST(HasTypeTraitsSupportedTest, is_true)
{
    EXPECT_TRUE(cuda::HasTypeTraits<typename TestFixture::Type>);
}

TEST(HasTypeTraitsWithTwoSupportedTypesTest, is_true)
{
    EXPECT_TRUE((cuda::HasTypeTraits<unsigned int, double4>));
}

// -------------------------- Testing IsCompound -------------------------------

TEST(IsCompoundTest, is_false)
{
    EXPECT_FALSE((cuda::IsCompound<unsigned char>));
    EXPECT_FALSE((cuda::IsCompound<char>));
    EXPECT_FALSE((cuda::IsCompound<unsigned short>));
    EXPECT_FALSE((cuda::IsCompound<int>));
}

TEST(IsCompoundTest, is_true)
{
    EXPECT_TRUE((cuda::IsCompound<uchar1>));
    EXPECT_TRUE((cuda::IsCompound<short2>));
    EXPECT_TRUE((cuda::IsCompound<uint3>));
    EXPECT_TRUE((cuda::IsCompound<float4>));
}

// --------------------- Testing HasEnoughComponents ---------------------------

TEST(HasEnoughComponentsTest, is_false)
{
    EXPECT_FALSE((cuda::HasEnoughComponents<unsigned char, 1>));
    EXPECT_FALSE((cuda::HasEnoughComponents<uchar1, 2>));
    EXPECT_FALSE((cuda::HasEnoughComponents<short2, 3>));
    EXPECT_FALSE((cuda::HasEnoughComponents<uint3, 4>));
    EXPECT_FALSE((cuda::HasEnoughComponents<float4, 5>));
}

TEST(HasEnoughComponentsTest, is_true)
{
    EXPECT_TRUE((cuda::HasEnoughComponents<unsigned char, 0>));
    EXPECT_TRUE((cuda::HasEnoughComponents<uchar1, 1>));
    EXPECT_TRUE((cuda::HasEnoughComponents<short2, 2>));
    EXPECT_TRUE((cuda::HasEnoughComponents<uint3, 3>));
    EXPECT_TRUE((cuda::HasEnoughComponents<float4, 4>));
}

// ---------------------------- Testing BaseType -------------------------------

template<typename T>
class TypeTraitsSupportedVectorTest : public TypeTraitsBaseTest<T>
{
};

using TypeTraitsSupportedVectorTypes = t::Types<dim3, uchar1, char2, ushort3, short4, uint1, int2, ulong3, long4,
                                                ulonglong1, longlong2, float3, double4>;

TYPED_TEST_SUITE(TypeTraitsSupportedVectorTest, TypeTraitsSupportedVectorTypes);

TYPED_TEST(TypeTraitsSupportedVectorTest, correct_type_traits)
{
    using TypeTraitsT = cuda::TypeTraits<typename TestFixture::Type>;
    using BaseType    = cuda::BaseType<typename TestFixture::Type>;

    EXPECT_TRUE((std::is_same_v<BaseType, typename TypeTraitsT::base_type>));

    EXPECT_TRUE(cuda::IsCompound<typename TestFixture::Type>);

    EXPECT_EQ(TypeTraitsT::components, cuda::NumComponents<typename TestFixture::Type>);
    EXPECT_EQ(TypeTraitsT::elements, cuda::NumElements<typename TestFixture::Type>);

    EXPECT_TRUE(TypeTraitsT::components >= 1);
    EXPECT_TRUE(TypeTraitsT::components <= 4);
    EXPECT_TRUE(TypeTraitsT::components == sizeof(typename TestFixture::Type) / sizeof(BaseType));

    EXPECT_TRUE(TypeTraitsT::elements >= 1);
    EXPECT_TRUE(TypeTraitsT::elements <= 4);
    EXPECT_TRUE(TypeTraitsT::elements == sizeof(typename TestFixture::Type) / sizeof(BaseType));

    EXPECT_EQ(TypeTraitsT::min, std::numeric_limits<typename TypeTraitsT::base_type>::min());
    EXPECT_EQ(TypeTraitsT::max, std::numeric_limits<typename TypeTraitsT::base_type>::max());
}

// ------------------- Testing TypeTraits with Type Qualifiers -----------------

class TypeTraitsTypeQualifierTest : public t::Test
{
};

TEST_F(TypeTraitsTypeQualifierTest, correct_type_traits_as_if_no_qualifier)
{
    using Type = char;

    EXPECT_TRUE((std::is_same_v<cuda::BaseType<const Type>, cuda::BaseType<Type>>));
    EXPECT_TRUE((std::is_same_v<cuda::BaseType<volatile Type>, cuda::BaseType<Type>>));
    EXPECT_TRUE((std::is_same_v<cuda::BaseType<const volatile Type>, cuda::BaseType<Type>>));

    EXPECT_TRUE(cuda::NumComponents<const Type> == cuda::NumComponents<Type>);
    EXPECT_TRUE(cuda::NumComponents<volatile Type> == cuda::NumComponents<Type>);
    EXPECT_TRUE(cuda::NumComponents<const volatile Type> == cuda::NumComponents<Type>);

    EXPECT_TRUE(cuda::NumElements<const Type> == cuda::NumElements<Type>);
    EXPECT_TRUE(cuda::NumElements<volatile Type> == cuda::NumElements<Type>);
    EXPECT_TRUE(cuda::NumElements<const volatile Type> == cuda::NumElements<Type>);
}

// ---------------------------- Testing MakeType -------------------------------

template<class T>
class TypeTraitsMakeTypeVectorTest : public t::Test
{
public:
    using BaseType                     = test::type::GetType<T, 0>;
    static constexpr int NumComponents = test::type::GetValue<T, 1>;
    static constexpr int NumElements   = NumComponents == 0 ? 1 : NumComponents;
};

using MakeTypeSupportedVectorTypes = test::type::Combine<TypeTraitsSupportedBaseTypes, test::Values<0, 4>>;

NVCV_TYPED_TEST_SUITE_F(TypeTraitsMakeTypeVectorTest, MakeTypeSupportedVectorTypes);

TYPED_TEST(TypeTraitsMakeTypeVectorTest, correct_type_traits)
{
    using CompoundType = cuda::MakeType<typename TestFixture::BaseType, TestFixture::NumComponents>;

    EXPECT_TRUE((std::is_same_v<typename TestFixture::BaseType, typename cuda::BaseType<CompoundType>>));

    EXPECT_TRUE(cuda::NumComponents<CompoundType> == TestFixture::NumComponents);

    EXPECT_TRUE(cuda::NumElements<CompoundType> == TestFixture::NumElements);

    EXPECT_EQ(TestFixture::NumComponents == 0, (std::is_same_v<typename TestFixture::BaseType, CompoundType>));
}

// -------------------- Testing MakeType with Type Qualifiers ------------------

template<typename T>
class TypeTraitsMakeTypeWithQualifierTest : public TypeTraitsMakeTypeVectorTest<T>
{
};

using MakeTypeSomeTypesWithQualifiers
    = test::type::Concat<test::type::Combine<test::Types<const int>, test::Values<0, 1>>,
                         test::type::Combine<test::Types<volatile int>, test::Values<0, 1>>,
                         test::type::Combine<test::Types<const volatile int>, test::Values<0, 1>>>;

NVCV_TYPED_TEST_SUITE_F(TypeTraitsMakeTypeWithQualifierTest, MakeTypeSomeTypesWithQualifiers);

TYPED_TEST(TypeTraitsMakeTypeWithQualifierTest, correct_type_qualifiers)
{
    using CompoundType = cuda::MakeType<typename TestFixture::BaseType, TestFixture::NumComponents>;

    EXPECT_EQ(std::is_const_v<typename TestFixture::BaseType>, std::is_const_v<CompoundType>);
    EXPECT_EQ(std::is_volatile_v<typename TestFixture::BaseType>, std::is_volatile_v<CompoundType>);
}

// ------------------------ Testing ConvertBaseTypeTo --------------------------

template<typename T>
class TypeTraitsConvertBaseTypeToTest : public TypeTraitsBaseTest<T>
{
};

NVCV_TYPED_TEST_SUITE_F(TypeTraitsConvertBaseTypeToTest, t::Types<char, short1, char2, uint3, double4>);

TYPED_TEST(TypeTraitsConvertBaseTypeToTest, correct_type_traits)
{
    using FloatType = cuda::ConvertBaseTypeTo<float, typename TestFixture::Type>;

    EXPECT_TRUE((std::is_same_v<cuda::BaseType<FloatType>, float>));

    EXPECT_TRUE(cuda::NumComponents<typename TestFixture::Type> == cuda::NumComponents<FloatType>);

    EXPECT_TRUE(cuda::NumElements<typename TestFixture::Type> == cuda::NumElements<FloatType>);

    EXPECT_FALSE((std::is_same_v<FloatType, typename TestFixture::Type>));
}

// -------------- Testing ConvertBaseTypeTo with Type Qualifiers ---------------

template<typename T>
class TypeTraitsConvertBaseTypeToWithQualifiersTest : public TypeTraitsBaseTest<T>
{
};

NVCV_TYPED_TEST_SUITE_F(TypeTraitsConvertBaseTypeToWithQualifiersTest,
                        t::Types<const short3, volatile short3, const volatile short3>);

TYPED_TEST(TypeTraitsConvertBaseTypeToWithQualifiersTest, correct_type_qualifiers)
{
    using FloatType = cuda::ConvertBaseTypeTo<float, typename TestFixture::Type>;

    EXPECT_EQ(std::is_const_v<typename TestFixture::Type>, std::is_const_v<FloatType>);
    EXPECT_EQ(std::is_volatile_v<typename TestFixture::Type>, std::is_volatile_v<FloatType>);
}

// --------------------------- Testing GetElement ------------------------------

template<typename T>
class TypeTraitsGetElementTest : public TypeTraitsBaseTest<T>
{
public:
    using DataType                   = typename TypeTraitsBaseTest<T>::Type;
    static constexpr int NumElements = cuda::NumElements<DataType>;

    DataType pix;

    TypeTraitsGetElementTest()
    {
        if constexpr (NumElements == 4)
        {
            pix = {1, 2, 3, 4};
        }
        else if constexpr (NumElements == 3)
        {
            pix = {1, 2, 3};
        }
        else if constexpr (NumElements == 2)
        {
            pix = {1, 2};
        }
        else if constexpr (NumElements == 1)
        {
            pix = {1};
        }
    }
};

using SomeSupportedTypes = t::Types<char, ushort1, uchar2, int3, float4, dim3>;

TYPED_TEST_SUITE(TypeTraitsGetElementTest, SomeSupportedTypes);

TYPED_TEST(TypeTraitsGetElementTest, correct_elements)
{
    for (int e = 0; e < this->NumElements; ++e)
    {
        EXPECT_EQ(nvcv::cuda::GetElement(this->pix, e), e + 1);
    }
}

// ----------------------------- Testing SetAll --------------------------------

template<typename T>
class TypeTraitsSetAllTest : public TypeTraitsBaseTest<T>
{
};

NVCV_TYPED_TEST_SUITE_F(TypeTraitsSetAllTest, t::Types<double, short1, uchar2, int3, float4>);

TYPED_TEST(TypeTraitsSetAllTest, correct_output_with_type)
{
    using T = typename TestFixture::Type;

    cuda::BaseType<T> gold = 3;

    auto test = cuda::SetAll<T>(gold);

    using TestType = decltype(test);

    EXPECT_TRUE((std::is_same_v<TestType, T>));

    for (int c = 0; c < cuda::NumElements<TestType>; ++c)
    {
        EXPECT_EQ(cuda::GetElement(test, c), gold);
    }
}

TYPED_TEST(TypeTraitsSetAllTest, correct_output_with_num_components)
{
    using T         = typename TestFixture::Type;
    constexpr int N = cuda::NumComponents<T>;

    cuda::BaseType<T> gold = 3;

    auto test = cuda::SetAll<N>(gold);

    using TestType = decltype(test);

    EXPECT_TRUE((std::is_same_v<TestType, T>));

    for (int c = 0; c < cuda::NumElements<TestType>; ++c)
    {
        EXPECT_EQ(cuda::GetElement(test, c), gold);
    }
}

// ------------------- Testing GetTypeName and operator << ---------------------

template<class T>
class TypeTraitsVectorTypePrintTest : public t::Test
{
public:
    using Type = test::type::GetType<T, 0>;

    static constexpr TStr GoldTypeName    = test::type::GetValue<T, 1>;
    static constexpr TStr GoldValueOutput = test::type::GetValue<T, 2>;

    Type val;

    TypeTraitsVectorTypePrintTest()
    {
        for (int e = 0; e < cuda::NumElements<Type>; ++e)
        {
            cuda::GetElement(val, e) = e + 1;
        }
    }
};

NVCV_TYPED_TEST_SUITE_F(
    TypeTraitsVectorTypePrintTest,
    test::type::Zip<test::Types<float, double1, int2, short3, ulong4>,
                    test::Values<TStr("float"), TStr("double1"), TStr("int2"), TStr("short3"), TStr("ulong4")>,
                    test::Values<TStr("1"), TStr("double1(1)"), TStr("int2(1, 2)"), TStr("short3(1, 2, 3)"),
                                 TStr("ulong4(1, 2, 3, 4)")>>);

TYPED_TEST(TypeTraitsVectorTypePrintTest, correct_output_stream)
{
    EXPECT_STREQ(nvcv::cuda::GetTypeName<typename TestFixture::Type>(), this->GoldTypeName.value);

    std::ostringstream oss;

    EXPECT_NO_THROW(oss << this->val);

    EXPECT_STREQ(oss.str().c_str(), this->GoldValueOutput.value);
}
