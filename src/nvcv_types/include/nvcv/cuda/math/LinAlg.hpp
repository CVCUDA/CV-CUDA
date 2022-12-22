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

/**
 * @file LinAlg.hpp
 *
 * @brief Defines linear algebra classes and functions.
 */

#ifndef NVCV_CUDA_MATH_LINALG_HPP
#define NVCV_CUDA_MATH_LINALG_HPP

#include <cuda_runtime.h> // for __host__, etc.

#include <algorithm> // for std::swap, etc.
#include <cassert>   // for assert, etc.
#include <cmath>     // for std::pow, etc.
#include <cstdlib>   // for std::size_t, etc.
#include <ostream>   // for std::ostream, etc.
#include <vector>    // for std::vector, etc.

namespace nvcv::cuda::math {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_LINALG Linear algebra
 * @{
 */

/**
 * @brief Vector class to represent small vectors.
 *
 * @tparam T Vector value type.
 * @tparam N Number of elements.
 */
template<class T, int N>
class Vector
{
public:
    // @brief Type of values in this vector.
    using Type = T;

    /**
     * @brief Get size (number of elements) of this vector
     *
     * @return Vector size
     */
    constexpr __host__ __device__ int size() const
    {
        return N;
    }

    /**
     * @brief Subscript operator for read-only access.
     *
     * @param[in] i Position to access
     *
     * @return Value (constant reference) at given position
     */
    constexpr const __host__ __device__ T &operator[](int i) const
    {
        assert(i >= 0 && i < size());
        return m_data[i];
    }

    /**
     * @brief Subscript operator for read-and-write access.
     *
     * @param[in] i Position to access
     *
     * @return Value (reference) at given position
     */
    constexpr __host__ __device__ T &operator[](int i)
    {
        assert(i >= 0 && i < size());
        return m_data[i];
    }

    /**
     * @brief Pointer-access operator (constant)
     *
     * @return Pointer to the first element of this vector
     */
    constexpr __host__ __device__ operator const T *() const
    {
        return &m_data[0];
    }

    /**
     * @brief Pointer-access operator
     *
     * @return Pointer to the first element of this vector
     */
    constexpr __host__ __device__ operator T *()
    {
        return &m_data[0];
    }

    /**
     * @brief Begin (constant) pointer access
     *
     * @return Pointer to the first element of this vector
     */
    constexpr const __host__ __device__ T *cbegin() const
    {
        return &m_data[0];
    }

    /**
     * @brief Begin pointer access
     *
     * @return Pointer to the first element of this vector
     */
    constexpr __host__ __device__ T *begin()
    {
        return &m_data[0];
    }

    /**
     * @brief End (constant) pointer access
     *
     * @return Pointer to the one-past-last element of this vector
     */
    constexpr const __host__ __device__ T *cend() const
    {
        return &m_data[0] + size();
    }

    /**
     * @brief End pointer access
     *
     * @return Pointer to the one-past-last element of this vector
     */
    constexpr __host__ __device__ T *end()
    {
        return &m_data[0] + size();
    }

    /**
     * @brief Convert a vector of this class to an stl vector
     *
     * @return STL std::vector
     */
    std::vector<T> to_vector() const
    {
        return std::vector<T>(cbegin(), cend());
    }

    /**
     * @brief Get a sub-vector of this vector
     *
     * @param[in] beg Position to start getting values from this vector
     *
     * @return Vector of value from given index to the end
     *
     * @tparam R Size of the sub-vector to be returned
     */
    template<int R>
    constexpr __host__ __device__ Vector<T, R> subv(int beg) const
    {
        Vector<T, R> v;
        for (int i = beg; i < beg + R; ++i)
        {
            v[i - beg] = m_data[i];
        }
        return v;
    }

    // @brief On-purpose public data to allow POD-class direct initialization
    T m_data[N] = {};
};

/**
 * @brief Matrix class to represent small matrices.
 *
 * @tparam T Matrix value type.
 * @tparam M Number of rows.
 * @tparam N Number of columns. Default is M.
 */
template<class T, int M, int N = M>
class Matrix
{
public:
    // @brief Type of values in this matrix.
    using Type = T;

    /**
     * @brief Get number of rows of this matrix
     *
     * @return Number of rows
     */
    constexpr __host__ __device__ int rows() const
    {
        return M;
    }

    /**
     * @brief Get number of columns of this matrix
     *
     * @return Number of columns
     */
    constexpr __host__ __device__ int cols() const
    {
        return N;
    }

    /**
     * @brief Subscript operator for read-only access.
     *
     * @param[in] i Row of the matrix to access
     *
     * @return Vector (constant reference) of the corresponding row
     */
    constexpr const __host__ __device__ Vector<T, N> &operator[](int i) const
    {
        assert(i >= 0 && i < rows());
        return m_data[i];
    }

    /**
     * @brief Subscript operator for read-and-write access.
     *
     * @param[in] i Row of the matrix to access
     *
     * @return Vector (reference) of the corresponding row
     */
    constexpr __host__ __device__ Vector<T, N> &operator[](int i)
    {
        assert(i >= 0 && i < rows());
        return m_data[i];
    }

    /**
     * @brief Subscript operator for read-only access of matrix elements.
     *
     * @param[in] c Coordinates (y row and x column) of the matrix element to access
     *
     * @return Element (constant reference) of the corresponding row and column
     */
    constexpr const __host__ __device__ T &operator[](int2 c) const
    {
        assert(c.y >= 0 && c.y < rows());
        assert(c.x >= 0 && c.x < cols());
        return m_data[c.y][c.x];
    }

    /**
     * @brief Subscript operator for read-and-write access of matrix elements.
     *
     * @param[in] c Coordinates (y row and x column) of the matrix element to access
     *
     * @return Element (reference) of the corresponding row and column
     */
    constexpr __host__ __device__ T &operator[](int2 c)
    {
        assert(c.y >= 0 && c.y < rows());
        assert(c.x >= 0 && c.x < cols());
        return m_data[c.y][c.x];
    }

    /**
     * @brief Get column j of this matrix
     *
     * @param[in] j Index of column to get
     *
     * @return Column j (copied) as a vector
     */
    constexpr __host__ __device__ Vector<T, M> col(int j) const
    {
        Vector<T, M> c;
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            c[i] = m_data[i][j];
        }
        return c;
    }

    /**
     * @brief Set column j of this matrix
     *
     * @param[in] j Index of column to set
     * @param[in] c Vector to place in matrix column
     */
    constexpr __host__ __device__ void set_col(int j, const Vector<T, M> &c)
    {
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            m_data[i][j] = c[i];
        }
    }

    // @overload void set_col(int j, const T *c)
    constexpr __host__ __device__ void set_col(int j, const T *c)
    {
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            m_data[i][j] = c[i];
        }
    }

    /**
     * @brief Get a sub-matrix of this matrix
     *
     * @param[in] skip_i Row to skip when getting values from this matrix
     * @param[in] skip_j Column to skip when getting values from this matrix
     *
     * @return Matrix with one less row and one less column
     */
    constexpr __host__ __device__ Matrix<T, M - 1, N - 1> subm(int skip_i, int skip_j) const
    {
        Matrix<T, M - 1, N - 1> ret;
        int                     ri = 0;
        for (int i = 0; i < rows(); ++i)
        {
            if (i == skip_i)
            {
                continue;
            }
            int rj = 0;
            for (int j = 0; j < cols(); ++j)
            {
                if (j == skip_j)
                {
                    continue;
                }
                ret[ri][rj] = (*this)[i][j];
                ++rj;
            }
            ++ri;
        }
        return ret;
    }

    // @brief On-purpose public data to allow POD-class direct initialization
    Vector<T, N> m_data[M];
};

namespace detail {

template<class T>
constexpr __host__ __device__ void swap(T &a, T &b)
{
#ifdef __CUDA_ARCH__
    T c = a;
    a   = b;
    b   = c;
#else
    std::swap(a, b);
#endif
}

} // namespace detail

// Vector-based operations -----------------------------------------------------

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator+=(Vector<T, N> &lhs, const Vector<T, N> &rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] += rhs[j];
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator+(const Vector<T, N> &a, const Vector<T, N> &b)
{
    Vector<T, N> r(a);
    return r += b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator+=(Vector<T, N> &lhs, T rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] += rhs;
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator+(const Vector<T, N> &a, T b)
{
    Vector<T, N> r(a);
    return r += b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator+(T a, const Vector<T, N> &b)
{
    Vector<T, N> r(b);
    return r += a;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator-=(Vector<T, N> &lhs, const Vector<T, N> &rhs)
{
    return lhs += -rhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator-(const Vector<T, N> &a, const Vector<T, N> &b)
{
    Vector<T, N> r(a);
    return r -= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator-=(Vector<T, N> &lhs, T rhs)
{
    return lhs += static_cast<T>(-rhs);
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator-(const Vector<T, N> &a, T b)
{
    Vector<T, N> r(a);
    return r -= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator-(T a, const Vector<T, N> &b)
{
    Vector<T, N> r(-b);
    return r += a;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator*=(Vector<T, N> &lhs, const T &rhs)
{
#pragma unroll
    for (int j = 0; j < lhs.size(); ++j)
    {
        lhs[j] *= rhs;
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator*(const Vector<T, N> &a, const T &b)
{
    Vector<T, N> r(a);
    return r *= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator*(const T &a, const Vector<T, N> &b)
{
    return b * a;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator*=(Vector<T, N> &lhs, const Vector<T, N> &rhs)
{
#pragma unroll
    for (int j = 0; j < lhs.size(); ++j)
    {
        lhs[j] *= rhs[j];
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator*(const Vector<T, N> &a, const Vector<T, N> &b)
{
    Vector<T, N> r(a);
    return r *= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator/=(Vector<T, N> &lhs, const T &rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] /= rhs;
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator/(const Vector<T, N> &a, const T &b)
{
    Vector<T, N> r(a);
    return r /= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator/(T a, const Vector<T, N> &b)
{
    Vector<T, N> r(b);
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        r[j] = a / r[j];
    }
    return r;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator/=(Vector<T, N> &lhs, const Vector<T, N> &rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] /= rhs[j];
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator/(const Vector<T, N> &a, const Vector<T, N> &b)
{
    Vector<T, N> r(a);
    return r /= b;
}

template<class T, int N>
std::ostream &operator<<(std::ostream &out, const Vector<T, N> &v)
{
    out << '[';
    for (int i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if (i < v.size() - 1)
        {
            out << ',';
        }
    }
    return out << ']';
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator-(const Vector<T, N> &v)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i)
    {
        r[i] = -v[i];
    }
    return r;
}

template<class T, int N>
constexpr __host__ __device__ bool operator==(const Vector<T, N> &a, const Vector<T, N> &b)
{
    for (int i = 0; i < N; ++i)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

template<class T, int N>
constexpr __host__ __device__ bool operator==(const T &a, const Vector<T, N> &b)
{
    for (int i = 0; i < N; ++i)
    {
        if (a != b[i])
        {
            return false;
        }
    }
    return true;
}

template<class T, int N>
constexpr __host__ __device__ bool operator==(const Vector<T, N> &a, const T &b)
{
    for (int i = 0; i < N; ++i)
    {
        if (a[i] != b)
        {
            return false;
        }
    }
    return true;
}

template<class T, int N>
constexpr __host__ __device__ bool operator<(const Vector<T, N> &a, const Vector<T, N> &b)
{
    for (int i = 0; i < N; ++i)
    {
        if (a[i] != b[i])
        {
            return a[i] < b[i];
        }
    }
    return false;
}

// Matrix-based operations -----------------------------------------------------

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator*(const Matrix<T, M, N> &m, T val)
{
    Matrix<T, M, N> r(m);
    return r *= val;
}

template<class T, int M, int N>
std::ostream &operator<<(std::ostream &out, const Matrix<T, M, N> &m)
{
    out << '[';
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = 0; j < m.cols(); ++j)
        {
            out << m[i][j];
            if (j < m.cols() - 1)
            {
                out << ',';
            }
        }
        if (i < m.rows() - 1)
        {
            out << ";";
        }
    }
    return out << ']';
}

template<class T, int M, int N, int P>
constexpr __host__ __device__ Matrix<T, M, P> operator*(const Matrix<T, M, N> &a, const Matrix<T, N, P> &b)
{
    Matrix<T, M, P> r;
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < P; ++j)
        {
            r[i][j] = a[i][0] * b[0][j];
#pragma unroll
            for (int k = 1; k < N; ++k)
            {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> &operator*=(Matrix<T, M, N> &lhs, T rhs)
{
#pragma unroll
    for (int i = 0; i < lhs.rows(); ++i)
    {
#pragma unroll
        for (int j = 0; j < lhs.cols(); ++j)
        {
            lhs[i][j] *= rhs;
        }
    }
    return lhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator*(T val, const Matrix<T, M, N> &m)
{
    return m * val;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> &operator+=(Matrix<T, M, N> &lhs, const Matrix<T, M, N> &rhs)
{
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            lhs[i][j] += rhs[i][j];
        }
    }
    return lhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator+(const Matrix<T, M, N> &lhs, const Matrix<T, M, N> &rhs)
{
    Matrix<T, M, N> r(lhs);
    return r += rhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> &operator-=(Matrix<T, M, N> &lhs, const Matrix<T, M, N> &rhs)
{
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            lhs[i][j] -= rhs[i][j];
        }
    }
    return lhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator-(const Matrix<T, M, N> &a, const Matrix<T, M, N> &b)
{
    Matrix<T, M, N> r(a);
    return r -= b;
}

template<class T, int M, int N>
constexpr __host__ __device__ Vector<T, N> operator*(const Vector<T, M> &v, const Matrix<T, M, N> &m)
{
    Vector<T, N> r;

#pragma unroll
    for (int j = 0; j < m.cols(); ++j)
    {
        r[j] = v[0] * m[0][j];
#pragma unroll
        for (int i = 1; i < m.rows(); ++i)
        {
            r[j] += v[i] * m[i][j];
        }
    }

    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ Vector<T, M> operator*(const Matrix<T, M, N> &m, const Vector<T, N> &v)
{
    Vector<T, M> r;

#pragma unroll
    for (int i = 0; i < m.rows(); ++i)
    {
        r[i] = m[i][0] * v[0];
#pragma unroll
        for (int j = 1; j < m.cols(); ++j)
        {
            r[i] += m[i][j] * v[j];
        }
    }

    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator*(const Matrix<T, M, 1> &m, const Vector<T, N> &v)
{
    Matrix<T, M, N> r;

#pragma unroll
    for (int i = 0; i < r.rows(); ++i)
    {
#pragma unroll
        for (int j = 0; j < r.cols(); ++j)
        {
            r[i][j] = m[i][0] * v[j];
        }
    }

    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ Vector<T, N> &operator*=(Vector<T, M> &v, const Matrix<T, M, M> &m)
{
    return v = v * m;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> &operator*=(Matrix<T, M, N> &lhs, const Matrix<T, N, N> &rhs)
{
    return lhs = lhs * rhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator-(const Matrix<T, M, N> &m)
{
    Matrix<T, M, N> r;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            r[i][j] = -m[i][j];
        }
    }
    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ bool operator==(const Matrix<T, M, N> &a, const Matrix<T, M, N> &b)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }
    }
    return true;
}

template<class T, int M, int N>
constexpr __host__ __device__ bool operator==(const T &a, const Matrix<T, M, N> &b)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (a != b[i])
            {
                return false;
            }
        }
    }
    return true;
}

template<class T, int M, int N>
constexpr __host__ __device__ bool operator==(const Matrix<T, M, N> &a, const T &b)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (a[i] != b)
            {
                return false;
            }
        }
    }
    return true;
}

template<class T, int M, int N>
constexpr __host__ __device__ bool operator<(const Matrix<T, M, N> &a, const Matrix<T, M, N> &b)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (a[i] != b[i])
            {
                return a[i] < b[i];
            }
        }
    }

    // if we reach here, it means that a==b
    return false;
}

// Special matrices ------------------------------------------------------------

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> zeros()
{
    Vector<T, N> v;
    if constexpr (N > 0)
    {
#if __CUDA_ARCH__
#    pragma unroll
        for (int j = 0; j < v.size(); ++j)
        {
            v[j] = T{0};
        }
#else
        std::fill(&v[0], &v[N - 1] + 1, T{0});
#endif
    }
    return v; // I'm hoping that RVO will kick in
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> zeros()
{
    Matrix<T, M, N> mat;
    if constexpr (M > 0 && N > 0)
    {
#if __CUDA_ARCH__
#    pragma unroll
        for (int i = 0; i < mat.rows(); ++i)
        {
#    pragma unroll
            for (int j = 0; j < mat.cols(); ++j)
            {
                mat[i][j] = T{0};
            }
        }
#else
        std::fill(&mat[0][0], &mat[M - 1][N - 1] + 1, T{0});
#endif
    }
    return mat; // I'm hoping that RVO will kick in
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> ones()
{
    Vector<T, N> v;
    if constexpr (N > 0)
    {
#if __CUDA_ARCH__
#    pragma unroll
        for (int j = 0; j < v.size(); ++j)
        {
            v[j] = T{1};
        }
#else
        std::fill(&v[0], &v[N - 1] + 1, T{1});
#endif
    }
    return v; // I'm hoping that RVO will kick in
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> ones()
{
    Matrix<T, M, N> mat;
    if constexpr (M > 0 && N > 0)
    {
#if __CUDA_ARCH__
#    pragma unroll
        for (int i = 0; i < mat.rows(); ++i)
        {
#    pragma unroll
            for (int j = 0; j < mat.cols(); ++j)
            {
                mat[i][j] = T{1};
            }
        }
#else
        std::fill(&mat[0][0], &mat[M - 1][N - 1] + 1, T{1});
#endif
    }
    return mat; // I'm hoping that RVO will kick in
}

template<class T, int M, int N>
__host__ __device__ Matrix<T, M, N> identity()
{
    Matrix<T, M, N> mat;

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            mat[i][j] = i == j ? 1 : 0;
        }
    }

    return mat;
}

template<class T, int M>
__host__ __device__ Matrix<T, M, M> vander(const Vector<T, M> &v)
{
    using std::pow;

    Matrix<T, M, M> m;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            m[i][j] = pow(v[j], i);
        }
    }

    return m;
}

template<class T, int R>
__host__ __device__ Matrix<T, R, R> compan(const Vector<T, R> &a)
{
    Matrix<T, R, R> m;
    for (int i = 0; i < R; ++i)
    {
        for (int j = 0; j < R; ++j)
        {
            if (j == R - 1)
            {
                m[i][j] = -a[i];
            }
            else
            {
                m[i][j] = j == i - 1 ? 1 : 0;
            }
        }
    }
    return m;
}

template<class T, int M>
__host__ __device__ Matrix<T, M, M> diag(const Vector<T, M> &v)
{
    Matrix<T, M, M> m;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            m[i][j] = i == j ? v[i] : 0;
        }
    }
    return m;
}

// Basic operations ------------------------------------------------------------

template<class T, int N>
constexpr __host__ __device__ T dot(const Vector<T, N> &a, const Vector<T, N> &b)
{
    T d = a[0] + b[0];
#pragma unroll
    for (int j = 1; j < a.size(); ++j)
    {
        d += a[j] * b[j];
    }
    return d;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> reverse(const Vector<T, N> &a)
{
    Vector<T, N> r;
#pragma unroll
    for (int j = 0; j < r.size(); ++j)
    {
        r[j] = a[a.size() - 1 - j];
    }
    return r;
}

// Transposition ---------------------------------------------------------------

template<class T, int M>
__host__ __device__ Matrix<T, M, M> &transp_inplace(Matrix<T, M, M> &m)
{
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = i + 1; j < m.cols(); ++j)
        {
            detail::swap(m[i][j], m[j][i]);
        }
    }
    return m;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, N, M> transp(const Matrix<T, M, N> &m)
{
    Matrix<T, N, M> tm;
#pragma unroll
    for (int i = 0; i < m.rows(); ++i)
    {
#pragma unroll
        for (int j = 0; j < m.cols(); ++j)
        {
            tm[j][i] = m[i][j];
        }
    }
    return tm;
}

template<class T, int N>
constexpr __host__ __device__ Matrix<T, N, 1> transp(const Vector<T, N> &v)
{
    Matrix<T, N, 1> tv;
    tv.set_col(0, v);
    return tv;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> flip_rows(const Matrix<T, M, N> &m)
{
    Matrix<T, M, N> f;
#pragma unroll
    for (int i = 0; i < m.rows(); ++i)
    {
#pragma unroll
        for (int j = 0; j < m.cols(); ++j)
        {
            f[i][j] = m[M - 1 - i][j];
        }
    }
    return f;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> flip_cols(const Matrix<T, M, N> &m)
{
    Matrix<T, M, N> f;
#pragma unroll
    for (int i = 0; i < m.rows(); ++i)
    {
#pragma unroll
        for (int j = 0; j < m.cols(); ++j)
        {
            f[i][j] = m[i][N - 1 - j];
        }
    }
    return f;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> flip(const Matrix<T, M, N> &m)
{
    Matrix<T, M, N> f;
#pragma unroll
    for (int i = 0; i < m.rows(); ++i)
    {
#pragma unroll
        for (int j = 0; j < m.cols(); ++j)
        {
            f[i][j] = m[M - 1 - i][N - 1 - j];
        }
    }
    return f;
}

// Determinant -----------------------------------------------------------------

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 0, 0> &m)
{
    return T{1};
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 1, 1> &m)
{
    return m[0][0];
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 2, 2> &m)
{
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 3, 3> &m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
         + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

template<class T, int M>
constexpr __host__ __device__ T det(const Matrix<T, M, M> &m)
{
    T d = T{0};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
        d += ((i % 2 == 0 ? 1 : -1) * m[0][i] * det(m.subm(0, i)));
    }
    return d;
}

// Matrix Inverse --------------------------------------------------------------

template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 1, 1> &m)
{
    m[0][0] = T{1} / m[0][0];
}

template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 2, 2> &m)
{
    T d = det(m);

    detail::swap(m[0][0], m[1][1]);
    m[0][0] /= d;
    m[1][1] /= d;

    m[0][1] = -m[0][1] / d;
    m[1][0] = -m[1][0] / d;
}

template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 3, 3> &m)
{
    T d = det(m);

    Matrix<T, 3, 3> A;
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

/**@}*/

} // namespace nvcv::cuda::math

#endif // NVCV_CUDA_MATH_LINALG_HPP
