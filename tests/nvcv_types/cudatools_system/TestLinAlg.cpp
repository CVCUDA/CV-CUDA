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

#include <common/TypedTests.hpp>     // for NVCV_TYPED_TEST_SUITE, etc.
#include <common/ValueTests.hpp>     // for StringLiteral
#include <nvcv/cuda/math/LinAlg.hpp> // the object of this test

#include <cmath>       // for std::pow, etc.
#include <numeric>     // for std::iota, etc.
#include <sstream>     // for std::stringstream, etc.
#include <type_traits> // for std::remove_reference_t, etc.

namespace test  = nvcv::test;
namespace math  = nvcv::cuda::math;
namespace ttype = nvcv::test::type;

template<int N>
using TStr = typename test::StringLiteral<N>;

using schar = signed char;
using uchar = unsigned char;

#define SCALAR(T, V) ttype::Value<T{V}>

#define VEC(T, N, ...) ttype::Value<math::Vector<T, N>{{__VA_ARGS__}}>

#define MAT(T, M, N, ...) ttype::Value<math::Matrix<T, M, N>{{__VA_ARGS__}}>

// --------------------------- Testing LinAlgVector ----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgVectorTest, ttype::Zip<test::Types<uchar, int, float>, test::Values<1, 5, 3>>);

// clang-format on

TYPED_TEST(LinAlgVectorTest, correct_type_size_content)
{
    using VectorType = ttype::GetType<TypeParam, 0>;
    constexpr int N  = ttype::GetValue<TypeParam, 1>;
    using Vector     = math::Vector<VectorType, N>;

    Vector vec{{1}};

    EXPECT_TRUE((std::is_same_v<typename Vector::Type, VectorType>));

    EXPECT_EQ(vec.size(), N);

    EXPECT_EQ(vec[0], 1);
}

TYPED_TEST(LinAlgVectorTest, correct_with_constexpr)
{
    using VectorType = ttype::GetType<TypeParam, 0>;
    constexpr int N  = ttype::GetValue<TypeParam, 1>;
    using Vector     = math::Vector<VectorType, N>;

    constexpr Vector vec{{1}};

    EXPECT_TRUE((std::is_same_v<typename Vector::Type, VectorType>));

    EXPECT_EQ(vec.size(), N);

    EXPECT_EQ(vec[0], 1);
}

TYPED_TEST(LinAlgVectorTest, can_change_content)
{
    using VectorType = ttype::GetType<TypeParam, 0>;
    constexpr int N  = ttype::GetValue<TypeParam, 1>;

    math::Vector<VectorType, N> vec;

    for (int i = 0; i < vec.size(); ++i)
    {
        vec[i] = i;

        EXPECT_EQ(vec[i], i);
    }
}

TYPED_TEST(LinAlgVectorTest, pointer_works)
{
    using VectorType = ttype::GetType<TypeParam, 0>;
    constexpr int N  = ttype::GetValue<TypeParam, 1>;

    math::Vector<VectorType, N> vec;

    VectorType *begin = vec;

    EXPECT_EQ(begin, vec.begin());

    VectorType *end = begin + N;

    EXPECT_EQ(end, vec.end());

    std::iota(begin, end, 0);

    std::vector<VectorType> test(begin, end);

    EXPECT_EQ(test, vec.to_vector());

    std::vector<VectorType> gold(N);

    std::iota(gold.begin(), gold.end(), 0);

    EXPECT_EQ(test, gold);
}

TYPED_TEST(LinAlgVectorTest, to_vector_works)
{
    using VectorType = ttype::GetType<TypeParam, 0>;
    constexpr int N  = ttype::GetValue<TypeParam, 1>;

    math::Vector<VectorType, N> vec;

    for (int i = 0; i < vec.size(); ++i)
    {
        vec[i] = i;
    }

    std::vector<VectorType> test = vec.to_vector();
    std::vector<VectorType> gold(N);

    std::iota(gold.begin(), gold.end(), 0);

    EXPECT_EQ(test, gold);
}

TYPED_TEST(LinAlgVectorTest, subv_works)
{
    using VectorType = ttype::GetType<TypeParam, 0>;
    constexpr int N  = ttype::GetValue<TypeParam, 1>;

    math::Vector<VectorType, N> vec{{3}};

    auto test = vec.template subv<1>(0);

    EXPECT_TRUE((std::is_same_v<typename decltype(test)::Type, VectorType>));

    EXPECT_EQ(test.size(), 1);

    EXPECT_EQ(test[0], 3);
}

// --------------------------- Testing LinAlgMatrix ----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgMatrixTest, ttype::Zip<
                      test::Types<uchar, int, float>,
                      test::Values<1, 2, 4>,
                      test::Values<1, 3, 3>
>);

// clang-format on

TYPED_TEST(LinAlgMatrixTest, correct_type_size_content)
{
    using MatrixType = ttype::GetType<TypeParam, 0>;
    constexpr int M  = ttype::GetValue<TypeParam, 1>;
    constexpr int N  = ttype::GetValue<TypeParam, 2>;
    using Matrix     = math::Matrix<MatrixType, M, N>;

    Matrix mat{{1}};

    EXPECT_TRUE((std::is_same_v<typename Matrix::Type, MatrixType>));

    EXPECT_EQ(mat.rows(), M);
    EXPECT_EQ(mat.cols(), N);

    EXPECT_EQ(mat[0][0], 1);
}

TYPED_TEST(LinAlgMatrixTest, correct_with_one_template_argument)
{
    using MatrixType = ttype::GetType<TypeParam, 0>;
    constexpr int M  = ttype::GetValue<TypeParam, 1>;
    using Matrix     = math::Matrix<MatrixType, M>;

    Matrix mat{{1}};

    EXPECT_TRUE((std::is_same_v<typename Matrix::Type, MatrixType>));

    EXPECT_EQ(mat.rows(), M);
    EXPECT_EQ(mat.cols(), M);

    int2 c{0, 0};

    EXPECT_EQ(mat[c], 1);
}

TYPED_TEST(LinAlgMatrixTest, correct_with_constexpr)
{
    using MatrixType = ttype::GetType<TypeParam, 0>;
    constexpr int M  = ttype::GetValue<TypeParam, 1>;
    constexpr int N  = ttype::GetValue<TypeParam, 2>;
    using Matrix     = math::Matrix<MatrixType, M, N>;

    constexpr Matrix mat{{1}};

    EXPECT_TRUE((std::is_same_v<typename Matrix::Type, MatrixType>));

    EXPECT_EQ(mat.rows(), M);
    EXPECT_EQ(mat.cols(), N);

    EXPECT_EQ(mat[0][0], 1);
}

TYPED_TEST(LinAlgMatrixTest, correct_with_one_template_argument_and_constexpr)
{
    using MatrixType = ttype::GetType<TypeParam, 0>;
    constexpr int M  = ttype::GetValue<TypeParam, 1>;
    using Matrix     = math::Matrix<MatrixType, M>;

    constexpr Matrix mat{{1}};

    EXPECT_TRUE((std::is_same_v<typename Matrix::Type, MatrixType>));

    EXPECT_EQ(mat.rows(), M);
    EXPECT_EQ(mat.cols(), M);

    int2 c{0, 0};

    EXPECT_EQ(mat[c], 1);
}

TYPED_TEST(LinAlgMatrixTest, can_change_content)
{
    using MatrixType = ttype::GetType<TypeParam, 0>;
    constexpr int M  = ttype::GetValue<TypeParam, 1>;
    constexpr int N  = ttype::GetValue<TypeParam, 2>;

    math::Matrix<MatrixType, M, N> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            mat[i][j] = i * mat.rows() + j;

            int2 c{j, i};

            EXPECT_EQ(mat[c], i * mat.rows() + j);
        }
    }
}

TYPED_TEST(LinAlgMatrixTest, col_works)
{
    using MatrixType = ttype::GetType<TypeParam, 0>;
    constexpr int M  = ttype::GetValue<TypeParam, 1>;
    constexpr int N  = ttype::GetValue<TypeParam, 2>;

    math::Matrix<MatrixType, M, N> mat;

    math::Vector<MatrixType, M> gold;

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            mat[i][j] = i * mat.rows() + j;
        }

        gold[i] = i * mat.rows();
    }

    math::Vector<MatrixType, M> test = mat.col(0);

    EXPECT_EQ(test, gold);
}

TYPED_TEST(LinAlgMatrixTest, set_col_works)
{
    using MatrixType = ttype::GetType<TypeParam, 0>;
    constexpr int M  = ttype::GetValue<TypeParam, 1>;
    constexpr int N  = ttype::GetValue<TypeParam, 2>;

    math::Matrix<MatrixType, M, N> mat{{1}};

    math::Vector<MatrixType, M> vec;

    std::iota(vec.begin(), vec.end(), 0);

    mat.set_col(0, vec);

    for (int i = 0; i < M; ++i)
    {
        EXPECT_EQ(mat[i][0], i);
    }
}

TYPED_TEST(LinAlgMatrixTest, set_col_with_pointer_works)
{
    using MatrixType = ttype::GetType<TypeParam, 0>;
    constexpr int M  = ttype::GetValue<TypeParam, 1>;
    constexpr int N  = ttype::GetValue<TypeParam, 2>;

    math::Matrix<MatrixType, M, N> mat{{1}};

    math::Vector<MatrixType, M> vec;

    std::iota(vec.begin(), vec.end(), 0);

    mat.set_col(0, static_cast<MatrixType *>(vec));

    for (int i = 0; i < M; ++i)
    {
        EXPECT_EQ(mat[i][0], i);
    }
}

// ------------------------ Testing LinAlg sub-matrix --------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgSubMatrixTest, ttype::Zip<
/* Matrix type */     test::Types<uchar, int, float>,
/* Matrix rows */     test::Values<4, 3, 5>,
/* Matrix cols */     test::Values<5, 3, 4>,
/* subm skip row */   test::Values<1, 0, 3>,
/* subm skip col */   test::Values<2, 1, 2>
>);

// clang-format on

TYPED_TEST(LinAlgSubMatrixTest, subm_works)
{
    using MatrixType     = ttype::GetType<TypeParam, 0>;
    constexpr int M      = ttype::GetValue<TypeParam, 1>;
    constexpr int N      = ttype::GetValue<TypeParam, 2>;
    constexpr int skip_i = ttype::GetValue<TypeParam, 3>;
    constexpr int skip_j = ttype::GetValue<TypeParam, 4>;

    math::Matrix<MatrixType, M, N> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        std::iota(mat[i].begin(), mat[i].end(), 0);
    }

    auto test = mat.subm(skip_i, skip_j);

    EXPECT_TRUE((std::is_same_v<typename decltype(test)::Type, MatrixType>));

    EXPECT_EQ(test.rows(), M - 1);
    EXPECT_EQ(test.cols(), N - 1);

    math::Matrix<MatrixType, M - 1, N - 1> gold;

    int ri = 0;
    for (int i = 0; i < mat.rows(); ++i)
    {
        if (i == skip_i)
        {
            continue;
        }
        int rj = 0;
        for (int j = 0; j < mat.cols(); ++j)
        {
            if (j == skip_j)
            {
                continue;
            }
            gold[ri][rj] = mat[i][j];
            ++rj;
        }
        ++ri;
    }

    EXPECT_EQ(test, gold);
}

// ----------------------- Testing LinAlg operator == --------------------------

TEST(LinAlgVectorEqualScalarTest, correct_result)
{
    math::Vector<int, 1> vec1{{1}};
    math::Vector<int, 3> vec2{
        {2, 2, 2}
    };

    EXPECT_TRUE(vec1 == 1);
    EXPECT_TRUE(vec2 == 2);
    EXPECT_FALSE(vec1 == 2);
    EXPECT_FALSE(vec2 == 1);
}

TEST(LinAlgVectorEqualVectorTest, correct_result)
{
    math::Vector<int, 3> iotaVec1{
        {1, 2, 3}
    };
    math::Vector<int, 3> iotaVec2{
        {1, 2, 3}
    };
    math::Vector<int, 3> flipVec1{
        {3, 2, 1}
    };

    EXPECT_TRUE(iotaVec1 == iotaVec2);
    EXPECT_FALSE(iotaVec1 == flipVec1);
    EXPECT_FALSE(iotaVec2 == flipVec1);
}

TEST(LinAlgMatrixEqualScalarTest, correct_result)
{
    math::Matrix<int, 1, 1> mat1{{{1}}};
    math::Matrix<int, 3, 3> mat2{
        {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}
    };

    EXPECT_TRUE(mat1 == 1);
    EXPECT_TRUE(mat2 == 2);
    EXPECT_FALSE(mat1 == 2);
    EXPECT_FALSE(mat2 == 1);
}

TEST(LinAlgMatrixEqualMatrixTest, correct_result)
{
    math::Matrix<int, 3, 3> iotaMat1{
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
    };
    math::Matrix<int, 3, 3> iotaMat2{
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
    };
    math::Matrix<int, 3, 3> flipMat1{
        {{7, 8, 9}, {4, 5, 6}, {1, 2, 3}}
    };

    EXPECT_TRUE(iotaMat1 == iotaMat2);
    EXPECT_FALSE(flipMat1 == iotaMat1);
    EXPECT_FALSE(flipMat1 == iotaMat2);
}

// ----------------------- Testing LinAlg operator < ---------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgOperatorLessTest, ttype::Types<
    ttype::Types<VEC(short, 2, 3, 2), VEC(short, 2, -3, -2)>,
    ttype::Types<MAT(schar, 2, 3, {1, 2, 3}, {2, 3, 4}), MAT(schar, 2, 3, {-1, -2, -3}, {-2, -3, -4})>
>);

// clang-format on

TYPED_TEST(LinAlgOperatorLessTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;

    auto test1 = input1 < input2;
    auto test2 = input2 < input1;

    constexpr auto test3 = ttype::GetValue<TypeParam, 0> < ttype::GetValue<TypeParam, 1>;
    constexpr auto test4 = ttype::GetValue<TypeParam, 1> < ttype::GetValue<TypeParam, 0>;

    EXPECT_FALSE(test1);
    EXPECT_TRUE(test2);
    EXPECT_FALSE(test3);
    EXPECT_TRUE(test4);
}

// ----------------------- Testing LinAlg operator << --------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgOutputStreamTest, ttype::Types<
    ttype::Types<VEC(int, 3, 1, 2, 3), ttype::Value<TStr("[1,2,3]")>>,
    ttype::Types<MAT(int, 2, 2, {1, 2}, {3, 4}), ttype::Value<TStr("[1,2;3,4]")>>
>);

// clang-format on

TYPED_TEST(LinAlgOutputStreamTest, correct_output)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    std::ostringstream oss;

    EXPECT_NO_THROW(oss << test);

    EXPECT_STREQ(oss.str().c_str(), gold.value);
}

// -------------------- Testing LinAlg unary operator - ------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgUnaryMinusTest, ttype::Types<
    ttype::Types<VEC(short, 2, 3, 2), VEC(short, 2, -3, -2)>,
    ttype::Types<VEC(int, 3, -33, 44, -22), VEC(int, 3, 33, -44, 22)>,
    ttype::Types<VEC(float, 4, 1.25f, 2.5f, 3.f, -4.f), VEC(float, 4, -1.25f, -2.5f, -3.f, 4.f)>,
    ttype::Types<MAT(schar, 2, 3, {1, 2, 3}, {2, 3, 4}), MAT(schar, 2, 3, {-1, -2, -3}, {-2, -3, -4})>,
    ttype::Types<MAT(long, 1, 2, {12345, -56789}), MAT(long, 1, 2, {-12345, 56789})>,
    ttype::Types<MAT(double, 3, 1, {1.0}, {2.0}, {3.0}), MAT(double, 3, 1, {-1.0}, {-2.0}, {-3.0})>
>);

// clang-format on

TYPED_TEST(LinAlgUnaryMinusTest, correct_output)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test1 = -input;

    constexpr auto test2 = -ttype::GetValue<TypeParam, 0>;

    EXPECT_EQ(test1, gold);
    EXPECT_EQ(test2, gold);
}

// ---------- Testing LinAlg operators (with assignment) +, -, *, / ------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgAddTest, ttype::Types<
    ttype::Types<VEC(uchar, 2, 3, 2), VEC(uchar, 2, 4, 5), VEC(uchar, 2, 7, 7)>,
    ttype::Types<VEC(int, 3, -33, 44, -22), VEC(int, 3, -44, -33, 11), VEC(int, 3, -77, 11, -11)>,
    ttype::Types<VEC(float, 4, 1.25f, 2.5f, 3.f, -4.f), VEC(float, 4, 2.f, 3.75f, -5.f, 6.f), VEC(float, 4, 3.25f, 6.25f, -2.f, 2.f)>,
    ttype::Types<MAT(schar, 2, 2, {1, -2}, {2, -3}), MAT(schar, 2, 2, {-1, 2}, {2, -3}), MAT(schar, 2, 2, {0, 0}, {4, -6})>,
    ttype::Types<MAT(long, 1, 2, {12345, -56789}), MAT(long, 1, 2, {-10, 20}), MAT(long, 1, 2, {12335, -56769})>,
    ttype::Types<MAT(double, 2, 1, {1.25}, {2.5}), MAT(double, 2, 1, {0.75}, {-2.25}), MAT(double, 2, 1, {2.0}, {0.25})>
>);

// clang-format on

TYPED_TEST(LinAlgAddTest, correct_output)
{
    auto input0 = ttype::GetValue<TypeParam, 0>;
    auto input1 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 + input1;
    auto test2 = input1 + input0;
    auto test3 = input0;
    auto test4 = input1;

    constexpr auto test5 = ttype::GetValue<TypeParam, 0> + ttype::GetValue<TypeParam, 1>;

    test3 += input1;
    test4 += input0;

    EXPECT_EQ(test1, gold);
    EXPECT_EQ(test2, gold);
    EXPECT_EQ(test3, gold);
    EXPECT_EQ(test4, gold);
    EXPECT_EQ(test5, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgVectorScalarAddTest, ttype::Types<
    ttype::Types<VEC(uchar, 2, 3, 2), SCALAR(uchar, 2), VEC(uchar, 2, 5, 4)>,
    ttype::Types<VEC(int, 3, -33, 44, -22), SCALAR(int, -11), VEC(int, 3, -44, 33, -33)>,
    ttype::Types<VEC(float, 4, 1.25f, 2.5f, 3.f, -4.f), SCALAR(float, 3.f), VEC(float, 4, 4.25f, 5.5f, 6.f, -1.f)>
>);

// clang-format on

TYPED_TEST(LinAlgVectorScalarAddTest, correct_output)
{
    auto input0  = ttype::GetValue<TypeParam, 0>;
    auto scalar1 = ttype::GetValue<TypeParam, 1>;
    auto gold    = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 + scalar1;
    auto test2 = scalar1 + input0;
    auto test3 = input0;

    constexpr auto test4 = ttype::GetValue<TypeParam, 0> + ttype::GetValue<TypeParam, 1>;

    test3 += scalar1;

    EXPECT_EQ(test1, gold);
    EXPECT_EQ(test2, gold);
    EXPECT_EQ(test3, gold);
    EXPECT_EQ(test4, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgSubTest, ttype::Types<
    ttype::Types<VEC(uchar, 2, 5, 6), VEC(uchar, 2, 4, 5), VEC(uchar, 2, 1, 1)>,
    ttype::Types<VEC(int, 3, -33, 44, -22), VEC(int, 3, -44, -33, 11), VEC(int, 3, 11, 77, -33)>,
    ttype::Types<VEC(float, 4, 1.25f, 2.5f, 3.f, -4.f), VEC(float, 4, 1.f, 1.75f, -5.f, 6.f), VEC(float, 4, 0.25f, 0.75f, 8.f, -10.f)>,
    ttype::Types<MAT(schar, 2, 2, {1, -2}, {2, -3}), MAT(schar, 2, 2, {-1, 2}, {2, -3}), MAT(schar, 2, 2, {2, -4}, {0, 0})>,
    ttype::Types<MAT(long, 1, 2, {12345, -56789}), MAT(long, 1, 2, {10, -20}), MAT(long, 1, 2, {12335, -56769})>,
    ttype::Types<MAT(double, 2, 1, {1.25}, {2.5}), MAT(double, 2, 1, {0.75}, {2.25}), MAT(double, 2, 1, {0.5}, {0.25})>
>);

// clang-format on

TYPED_TEST(LinAlgSubTest, correct_output)
{
    auto input0 = ttype::GetValue<TypeParam, 0>;
    auto input1 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 - input1;
    auto test2 = input1 - input0;
    auto test3 = input0;
    auto test4 = input1;

    constexpr auto test5 = ttype::GetValue<TypeParam, 0> - ttype::GetValue<TypeParam, 1>;

    test3 -= input1;
    test4 -= input0;

    EXPECT_EQ(test1, gold);
    EXPECT_NE(test2, gold);
    EXPECT_EQ(test3, gold);
    EXPECT_NE(test4, gold);
    EXPECT_EQ(test5, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgVectorScalarSubTest, ttype::Types<
    ttype::Types<VEC(uchar, 2, 3, 2), SCALAR(uchar, 2), VEC(uchar, 2, 1, 0)>,
    ttype::Types<VEC(int, 3, -33, 44, -22), SCALAR(int, -11), VEC(int, 3, -22, 55, -11)>,
    ttype::Types<VEC(float, 4, 1.25f, 2.5f, 3.f, -4.f), SCALAR(float, 1.f), VEC(float, 4, 0.25f, 1.5f, 2.f, -5.f)>
>);

// clang-format on

TYPED_TEST(LinAlgVectorScalarSubTest, correct_output)
{
    auto input0  = ttype::GetValue<TypeParam, 0>;
    auto scalar1 = ttype::GetValue<TypeParam, 1>;
    auto gold    = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 - scalar1;
    auto test2 = scalar1 - input0;
    auto test3 = input0;

    constexpr auto test4 = ttype::GetValue<TypeParam, 0> - ttype::GetValue<TypeParam, 1>;

    test3 -= scalar1;

    EXPECT_EQ(test1, gold);
    EXPECT_NE(test2, gold);
    EXPECT_EQ(test3, gold);
    EXPECT_EQ(test4, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgVectorMulTest, ttype::Types<
    ttype::Types<VEC(uchar, 2, 3, 2), VEC(uchar, 2, 4, 5), VEC(uchar, 2, 12, 10)>,
    ttype::Types<VEC(int, 3, -3, 4, -2), VEC(int, 3, -4, -3, 1), VEC(int, 3, 12, -12, -2)>,
    ttype::Types<VEC(float, 4, 1.f, 2.5f, 3.f, -4.f), VEC(float, 4, 3.f, 2.f, -5.f, 6.f), VEC(float, 4, 3.f, 5.f, -15.f, -24.f)>
>);

// clang-format on

TYPED_TEST(LinAlgVectorMulTest, correct_output)
{
    auto input0 = ttype::GetValue<TypeParam, 0>;
    auto input1 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 * input1;
    auto test2 = input1 * input0;
    auto test3 = input0;
    auto test4 = input1;

    constexpr auto test5 = ttype::GetValue<TypeParam, 0> * ttype::GetValue<TypeParam, 1>;

    test3 *= input1;
    test4 *= input0;

    EXPECT_EQ(test1, gold);
    EXPECT_EQ(test2, gold);
    EXPECT_EQ(test3, gold);
    EXPECT_EQ(test4, gold);
    EXPECT_EQ(test5, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgVectorScalarMulTest, ttype::Types<
    ttype::Types<VEC(uchar, 2, 3, 2), SCALAR(uchar, 2), VEC(uchar, 2, 6, 4)>,
    ttype::Types<VEC(int, 3, -3, 4, -2), SCALAR(int, -3), VEC(int, 3, 9, -12, 6)>,
    ttype::Types<VEC(float, 4, 1.f, 2.5f, 3.f, -4.f), SCALAR(float, 3.f), VEC(float, 4, 3.f, 7.5f, 9.f, -12.f)>
>);

// clang-format on

TYPED_TEST(LinAlgVectorScalarMulTest, correct_output)
{
    auto input0  = ttype::GetValue<TypeParam, 0>;
    auto scalar1 = ttype::GetValue<TypeParam, 1>;
    auto gold    = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 * scalar1;
    auto test2 = scalar1 * input0;
    auto test3 = input0;

    constexpr auto test4 = ttype::GetValue<TypeParam, 0> * ttype::GetValue<TypeParam, 1>;

    test3 *= scalar1;

    EXPECT_EQ(test1, gold);
    EXPECT_EQ(test2, gold);
    EXPECT_EQ(test3, gold);
    EXPECT_EQ(test4, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgMatrixMulTest, ttype::Types<
    ttype::Types<MAT(schar, 2, 3, {1, 2, 3}, {4, 5, 6}), VEC(schar, 3, 1, 2, 3), VEC(schar, 2, 14, 32)>,
    ttype::Types<MAT(int, 2, 2, {1, 2}, {3, 4}), MAT(int, 2, 2, {2, 3}, {4, 5}), MAT(int, 2, 2, {10, 13}, {22, 29})>,
    ttype::Types<MAT(float, 2, 3, {2.f, 1.f, 2.f}, {1.f, 3.f, 1.f}), MAT(float, 3, 2, {1.f, 3.f}, {2.f, 2.f}, {3.f, 1.f}),
                 MAT(float, 2, 2, {10.f, 10.f}, {10.f, 10.f})>
>);

// clang-format on

TYPED_TEST(LinAlgMatrixMulTest, correct_output)
{
    auto input0 = ttype::GetValue<TypeParam, 0>;
    auto input1 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 * input1;

    constexpr auto test2 = ttype::GetValue<TypeParam, 0> * ttype::GetValue<TypeParam, 1>;

    EXPECT_EQ(test1, gold);
    EXPECT_EQ(test2, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgVectorDivTest, ttype::Types<
    ttype::Types<VEC(uchar, 2, 8, 5), VEC(uchar, 2, 4, 5), VEC(uchar, 2, 2, 1)>,
    ttype::Types<VEC(int, 3, -33, 44, -22), VEC(int, 3, -3, -2, 1), VEC(int, 3, 11, -22, -22)>,
    ttype::Types<VEC(float, 4, 1.f, 2.5f, 3.f, -4.f), VEC(float, 4, 1.f, -2.5f, 2.f, 4.f), VEC(float, 4, 1.f, -1.f, 1.5f, -1.f)>
>);

// clang-format on

TYPED_TEST(LinAlgVectorDivTest, correct_output)
{
    auto input0 = ttype::GetValue<TypeParam, 0>;
    auto input1 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 / input1;
    auto test2 = input1 / input0;
    auto test3 = input0;
    auto test4 = input1;

    constexpr auto test5 = ttype::GetValue<TypeParam, 0> / ttype::GetValue<TypeParam, 1>;

    test3 /= input1;
    test4 /= input0;

    EXPECT_EQ(test1, gold);
    EXPECT_NE(test2, gold);
    EXPECT_EQ(test3, gold);
    EXPECT_NE(test4, gold);
    EXPECT_EQ(test5, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgVectorScalarDivTest, ttype::Types<
    ttype::Types<VEC(uchar, 2, 4, 2), SCALAR(uchar, 2), VEC(uchar, 2, 2, 1)>,
    ttype::Types<VEC(int, 3, -6, 4, -2), SCALAR(int, -2), VEC(int, 3, 3, -2, 1)>,
    ttype::Types<VEC(float, 4, 1.f, 2.5f, 3.f, -4.f), SCALAR(float, 2.f), VEC(float, 4, 0.5f, 1.25f, 1.5f, -2.f)>
>);

// clang-format on

TYPED_TEST(LinAlgVectorScalarDivTest, correct_output)
{
    auto input0  = ttype::GetValue<TypeParam, 0>;
    auto scalar1 = ttype::GetValue<TypeParam, 1>;
    auto gold    = ttype::GetValue<TypeParam, 2>;

    auto test1 = input0 / scalar1;
    auto test2 = scalar1 / input0;
    auto test3 = input0;

    constexpr auto test4 = ttype::GetValue<TypeParam, 0> / ttype::GetValue<TypeParam, 1>;

    test3 /= scalar1;

    EXPECT_EQ(test1, gold);
    EXPECT_NE(test2, gold);
    EXPECT_EQ(test3, gold);
    EXPECT_EQ(test4, gold);
}

// -------------- Testing LinAlg special vectors and matrices ------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgSpecialVectorMatrixTest, ttype::Zip<
                      test::Types<int, float>,
                      test::Values<4, 9>,
                      test::Values<8, 13>
>);

// clang-format on

TYPED_TEST(LinAlgSpecialVectorMatrixTest, correct_content_of_zeros)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;
    constexpr int N = ttype::GetValue<TypeParam, 2>;

    auto vec0 = math::zeros<Type, M>();
    auto mat0 = math::zeros<Type, M, N>();

    constexpr auto cvec0 = math::zeros<Type, M>();
    constexpr auto cmat0 = math::zeros<Type, M>();

    EXPECT_EQ(vec0, Type{0});
    EXPECT_EQ(mat0, Type{0});
    EXPECT_EQ(cvec0, Type{0});
    EXPECT_EQ(cmat0, Type{0});
}

TYPED_TEST(LinAlgSpecialVectorMatrixTest, correct_content_of_ones)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;
    constexpr int N = ttype::GetValue<TypeParam, 2>;

    auto vec1 = math::ones<Type, M>();
    auto mat1 = math::ones<Type, M, N>();

    constexpr auto cvec1 = math::ones<Type, M>();
    constexpr auto cmat1 = math::ones<Type, M>();

    EXPECT_EQ(vec1, Type{1});
    EXPECT_EQ(mat1, Type{1});
    EXPECT_EQ(cvec1, Type{1});
    EXPECT_EQ(cmat1, Type{1});
}

TYPED_TEST(LinAlgSpecialVectorMatrixTest, correct_content_of_identity)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;
    constexpr int N = ttype::GetValue<TypeParam, 2>;

    auto eye1 = math::identity<Type, M, M>();
    auto eye2 = math::identity<Type, M, N>();

    for (int i = 0; i < eye1.rows(); ++i)
    {
        for (int j = 0; j < eye1.cols(); ++j)
        {
            int diagonal = (i == j) ? 1 : 0;

            EXPECT_EQ(eye1[i][j], diagonal);
        }
        for (int j = 0; j < eye2.cols(); ++j)
        {
            int diagonal = (i == j) ? 1 : 0;

            EXPECT_EQ(eye2[i][j], diagonal);
        }
    }
}

TYPED_TEST(LinAlgSpecialVectorMatrixTest, correct_content_of_vander)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Vector<Type, M> vec;

    std::iota(vec.begin(), vec.end(), 0);

    auto matVander = math::vander(vec);

    for (int i = 0; i < matVander.rows(); ++i)
    {
        for (int j = 0; j < matVander.cols(); ++j)
        {
            EXPECT_EQ(matVander[i][j], std::pow(vec[j], i));
        }
    }
}

TYPED_TEST(LinAlgSpecialVectorMatrixTest, correct_content_of_compan)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Vector<Type, M> vec;

    std::iota(vec.begin(), vec.end(), 0);

    auto matCompan = math::compan(vec);

    for (int i = 0; i < matCompan.rows(); ++i)
    {
        for (int j = 0; j < matCompan.cols(); ++j)
        {
            Type value = ((j == matCompan.cols() - 1) ? -vec[i] : ((j == i - 1) ? 1 : 0));

            EXPECT_EQ(matCompan[i][j], value);
        }
    }
}

TYPED_TEST(LinAlgSpecialVectorMatrixTest, correct_content_of_diag)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Vector<Type, M> vec;

    std::iota(vec.begin(), vec.end(), 0);

    auto matDiag = math::diag(vec);

    for (int i = 0; i < matDiag.rows(); ++i)
    {
        for (int j = 0; j < matDiag.cols(); ++j)
        {
            Type value = (i == j) ? vec[i] : 0;

            EXPECT_EQ(matDiag[i][j], value);
        }
    }
}

// ------------ Testing LinAlg dot and reverse vector operations ---------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgDotAndReverseVectorTest, ttype::Zip<
                      test::Types<int, float>,
                      test::Values<4, 9>
>);

// clang-format on

TYPED_TEST(LinAlgDotAndReverseVectorTest, correct_content_of_dot)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Vector<Type, M> vec1, vec2;

    std::iota(vec1.begin(), vec1.end(), 0);
    std::iota(vec2.begin(), vec2.end(), 0);

    auto test = math::dot(vec1, vec2);

    auto gold = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0);

    EXPECT_EQ(test, gold);
}

TYPED_TEST(LinAlgDotAndReverseVectorTest, correct_content_of_reverse)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Vector<Type, M> vec;

    std::iota(vec.begin(), vec.end(), 0);

    auto test = math::reverse(vec);

    auto gold = vec;

    std::reverse(gold.begin(), gold.end());

    EXPECT_EQ(test, gold);
}

// -------------------- Testing LinAlg transp* operations ----------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgTranspTest, ttype::Zip<
                      test::Types<int, float>,
                      test::Values<4, 9>,
                      test::Values<8, 13>
>);

// clang-format on

TYPED_TEST(LinAlgTranspTest, correct_content_of_transp)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;
    constexpr int N = ttype::GetValue<TypeParam, 1>;

    math::Matrix<Type, M, N> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        std::iota(mat[i].begin(), mat[i].end(), 0);
    }

    auto test = math::transp(mat);

    EXPECT_TRUE((std::is_same_v<typename decltype(test)::Type, Type>));

    EXPECT_EQ(test.rows(), N);
    EXPECT_EQ(test.cols(), M);

    math::Matrix<Type, N, M> gold;

    for (int i = 0; i < gold.rows(); ++i)
    {
        for (int j = 0; j < gold.cols(); ++j)
        {
            gold[i][j] = mat[j][i];
        }
    }

    EXPECT_EQ(test, gold);
}

TYPED_TEST(LinAlgTranspTest, correct_content_of_transp_inplace)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Matrix<Type, M> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        std::iota(mat[i].begin(), mat[i].end(), 0);
    }

    auto test1 = mat;
    auto test2 = math::transp_inplace(test1);

    EXPECT_EQ(test1, test2);

    EXPECT_TRUE((std::is_same_v<typename decltype(test1)::Type, Type>));

    EXPECT_EQ(test1.rows(), M);
    EXPECT_EQ(test1.cols(), M);

    math::Matrix<Type, M, M> gold;

    for (int i = 0; i < gold.rows(); ++i)
    {
        for (int j = 0; j < gold.cols(); ++j)
        {
            gold[i][j] = mat[j][i];
        }
    }

    EXPECT_EQ(test1, gold);
}

TYPED_TEST(LinAlgTranspTest, correct_content_of_transp_vector)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Vector<Type, M> vec;

    std::iota(vec.begin(), vec.end(), 0);

    auto test = math::transp(vec);

    EXPECT_TRUE((std::is_same_v<typename decltype(test)::Type, Type>));

    EXPECT_EQ(test.rows(), M);
    EXPECT_EQ(test.cols(), 1);

    math::Matrix<Type, M, 1> gold;

    for (int i = 0; i < gold.rows(); ++i)
    {
        for (int j = 0; j < gold.cols(); ++j)
        {
            gold[i][j] = vec[i];
        }
    }

    EXPECT_EQ(test, gold);
}

// ------------------ Testing LinAlg flip* matrix operations -------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgFlipMatrixTest, ttype::Zip<
                      test::Types<int, float>,
                      test::Values<4, 9>,
                      test::Values<8, 13>
>);

// clang-format on

TYPED_TEST(LinAlgFlipMatrixTest, correct_content_of_flip)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;
    constexpr int N = ttype::GetValue<TypeParam, 1>;

    math::Matrix<Type, M, N> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        std::iota(mat[i].begin(), mat[i].end(), 0);
    }

    auto test = math::flip(mat);

    EXPECT_TRUE((std::is_same_v<typename decltype(test)::Type, Type>));

    EXPECT_EQ(test.rows(), M);
    EXPECT_EQ(test.cols(), N);

    math::Matrix<Type, N, M> gold;

    for (int i = 0; i < gold.rows(); ++i)
    {
        for (int j = 0; j < gold.cols(); ++j)
        {
            gold[i][j] = mat[gold.rows() - 1 - i][gold.cols() - 1 - j];
        }
    }

    EXPECT_EQ(test, gold);
}

TYPED_TEST(LinAlgFlipMatrixTest, correct_content_of_flip_rows)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;
    constexpr int N = ttype::GetValue<TypeParam, 1>;

    math::Matrix<Type, M, N> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        std::iota(mat[i].begin(), mat[i].end(), 0);
    }

    auto test = math::flip_rows(mat);

    EXPECT_TRUE((std::is_same_v<typename decltype(test)::Type, Type>));

    EXPECT_EQ(test.rows(), M);
    EXPECT_EQ(test.cols(), N);

    math::Matrix<Type, N, M> gold;

    for (int i = 0; i < gold.rows(); ++i)
    {
        for (int j = 0; j < gold.cols(); ++j)
        {
            gold[i][j] = mat[gold.rows() - 1 - i][j];
        }
    }

    EXPECT_EQ(test, gold);
}

TYPED_TEST(LinAlgFlipMatrixTest, correct_content_of_flip_cols)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;
    constexpr int N = ttype::GetValue<TypeParam, 1>;

    math::Matrix<Type, M, N> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        std::iota(mat[i].begin(), mat[i].end(), 0);
    }

    auto test = math::flip_cols(mat);

    EXPECT_TRUE((std::is_same_v<typename decltype(test)::Type, Type>));

    EXPECT_EQ(test.rows(), M);
    EXPECT_EQ(test.cols(), N);

    math::Matrix<Type, N, M> gold;

    for (int i = 0; i < gold.rows(); ++i)
    {
        for (int j = 0; j < gold.cols(); ++j)
        {
            gold[i][j] = mat[i][mat.cols() - 1 - j];
        }
    }

    EXPECT_EQ(test, gold);
}

// ------------------- Testing LinAlg det matrix operations --------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgDetMatrixTest, ttype::Zip<
                      test::Types<uchar, int, float>,
                      test::Values<2, 3, 4>
>);

// clang-format on

template<typename T, int M>
struct GoldDet
{
    T operator()(const math::Matrix<T, M, M> &m)
    {
        GoldDet<T, M - 1> goldDet;
        T                 d = 0;
        for (int i = 0; i < M; ++i)
        {
            d += ((i % 2 == 0 ? 1 : -1) * m[0][i] * goldDet(m.subm(0, i)));
        }
        return d;
    }
};

template<typename T>
struct GoldDet<T, 0>
{
    T operator()(const math::Matrix<T, 0, 0> &m)
    {
        return T{1};
    }
};

TYPED_TEST(LinAlgDetMatrixTest, correct_content_of_det)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Matrix<Type, M, M> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        std::iota(mat[i].begin(), mat[i].end(), 0);
    }

    auto test = math::det(mat);

    EXPECT_TRUE((std::is_same_v<decltype(test), Type>));

    GoldDet<Type, M> goldDet;

    Type gold = goldDet(mat);

    EXPECT_EQ(test, gold);
}

// --------------- Testing LinAlg inv_inplace matrix operations ----------------

// clang-format off
NVCV_TYPED_TEST_SUITE(LinAlgInvMatrixTest, ttype::Zip<
                      test::Types<float, double, float>,
                      test::Values<1, 2, 3>
>);

// clang-format on

template<typename T, int M>
struct GoldInv
{
    void operator()(math::Matrix<T, M, M> &m) {}
};

template<typename T>
struct GoldInv<T, 1>
{
    void operator()(math::Matrix<T, 1, 1> &m)
    {
        m[0][0] = T{1} / m[0][0];
    }
};

template<typename T>
struct GoldInv<T, 2>
{
    void operator()(math::Matrix<T, 2, 2> &m)
    {
        GoldDet<T, 2> goldDet;
        T             d = goldDet(m);
        std::swap(m[0][0], m[1][1]);
        m[0][0] /= d;
        m[1][1] /= d;

        m[0][1] = -m[0][1] / d;
        m[1][0] = -m[1][0] / d;
    }
};

template<typename T>
struct GoldInv<T, 3>
{
    void operator()(math::Matrix<T, 3, 3> &m)
    {
        GoldDet<T, 3> goldDet;
        T             d = goldDet(m);

        math::Matrix<T, 3, 3> A;
        A[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
        A[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / d;
        A[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
        A[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / d;
        A[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
        A[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / d;
        A[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
        A[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / d;
        A[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;

        m = A;
    }
};

TYPED_TEST(LinAlgInvMatrixTest, correct_content_of_inv_inplace)
{
    using Type      = ttype::GetType<TypeParam, 0>;
    constexpr int M = ttype::GetValue<TypeParam, 1>;

    math::Matrix<Type, M, M> mat;

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            mat[i][j] = (i * mat.cols() + j) / static_cast<Type>(M * M);
        }
    }

    auto test = mat;

    math::inv_inplace(test);

    GoldInv<Type, M> goldInv;

    auto gold = mat;

    goldInv(gold);

    EXPECT_EQ(test, gold);
}
