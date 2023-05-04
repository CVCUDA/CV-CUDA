/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file FullTensorWrap.hpp
 *
 * @brief Defines N-D tensor wrapper with full information on strides and shapes.
 */

#ifndef NVCV_CUDA_FULL_TENSOR_WRAP_HPP
#define NVCV_CUDA_FULL_TENSOR_WRAP_HPP

#include "TypeTraits.hpp" // for HasTypeTraits, etc.

#include <nvcv/TensorData.hpp> // for TensorDataStridedCuda, etc.

#include <type_traits>

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_FULLTENSORWRAP FullTensorWrap classes
 * @{
 */

/**
 * FullTensorWrap class is a non-owning wrap of a N-D tensor used for easy access of its elements in CUDA device.
 *
 * FullTensorWrap is a wrapper of a multi-dimensional tensor that holds all information related to it, i.e. \p N
 * strides and \p N shapes, where \p N is its number of dimensions.
 *
 * Template arguments:
 * - T type of the values inside the tensor
 * - N dimensions
 *
 * @tparam T Type (it can be const) of each element (or value) inside the tensor wrapper.
 * @tparam N dimensions.
 */
template<typename T, int N>
class FullTensorWrap;

template<typename T, int N>
class FullTensorWrap<const T, N>
{
    // It is a requirement of this class that its type has type traits.
    static_assert(HasTypeTraits<T>, "FullTensorWrap<T> can only be used if T has type traits");

public:
    // The type provided as template parameter is the value type, i.e. the type of each element inside this wrapper.
    using ValueType = const T;

    // The number of dimensions is provided as a template parameter.
    static constexpr int kNumDimensions = N;
    // The number of variable strides is fixed as the number of dimensions.
    static constexpr int kVariableStrides = N;
    // The number of constant strides is fixed as 0, meaning there is no compile-time stride.
    static constexpr int kConstantStrides = 0;

    FullTensorWrap() = default;

    /**
     * Constructs a constant FullTensorWrap by wrapping a const \p data pointer argument.
     *
     * @param[in] data Pointer to the data that will be wrapped.
     * @param[in] strides Array of strides in bytes from first to last dimension.
     * @param[in] shapes Array of shapes (number of elements) from first to last dimension.
     *
     * @tparam DataType Type of (inferred from) the data pointer argument.
     */
    template<typename DataType>
    explicit __host__ __device__ FullTensorWrap(const DataType *data, const int (&strides)[N], const int (&shapes)[N])
        : m_data(reinterpret_cast<const std::byte *>(data))
    {
#pragma unroll
        for (int i = 0; i < kNumDimensions; ++i)
        {
            m_strides[i] = strides[i];
            m_shapes[i]  = shapes[i];
        }
    }

    /**
     * Constructs a constant FullTensorWrap by wrapping a \p tensor argument.
     *
     * @param[in] tensor Tensor reference to the tensor that will be wrapped.
     */
    __host__ FullTensorWrap(const TensorDataStridedCuda &tensor)
    {
        m_data = reinterpret_cast<const std::byte *>(tensor.basePtr());

#pragma unroll
        for (int i = 0; i < kNumDimensions; ++i)
        {
            assert(tensor.stride(i) <= TypeTraits<int>::max);
            assert(tensor.shape(i) <= TypeTraits<int>::max);

            m_strides[i] = tensor.stride(i);
            m_shapes[i]  = tensor.shape(i);
        }
    }

    /**
     * Get strides in bytes for read-only access.
     *
     * @return The const array (as a pointer) containing the strides in bytes.
     */
    __host__ __device__ const int *strides() const
    {
        return m_strides;
    }

    /**
     * Get shapes for read-only access.
     *
     * @return The const array (as a pointer) containing the shapes.
     */
    __host__ __device__ const int *shapes() const
    {
        return m_shapes;
    }

    /**
     * Subscript operator for read-only access.
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed.
     *
     * @return Accessed const reference.
     */
    template<typename DimType>
    inline const __host__ __device__ T &operator[](DimType c) const
    {
        if constexpr (NumElements<DimType> == 1)
        {
            if constexpr (NumComponents<DimType> == 0)
            {
                return *doGetPtr(c);
            }
            else
            {
                return *doGetPtr(c.x);
            }
        }
        else if constexpr (NumElements<DimType> == 2)
        {
            return *doGetPtr(c.y, c.x);
        }
        else if constexpr (NumElements<DimType> == 3)
        {
            return *doGetPtr(c.z, c.y, c.x);
        }
        else if constexpr (NumElements<DimType> == 4)
        {
            return *doGetPtr(c.w, c.z, c.y, c.x);
        }
    }

    /**
     * Get a read-only proxy (as pointer) at the Dth dimension.
     *
     * @param[in] c0..D Each coordinate from first to last dimension.
     *
     * @return The const pointer to the beginning of the Dth dimension.
     */
    template<typename... Args>
    inline const __host__ __device__ T *ptr(Args... c) const
    {
        return doGetPtr(c...);
    }

protected:
    template<typename... Args>
    inline const __host__ __device__ T *doGetPtr(Args... c) const
    {
        static_assert(std::conjunction_v<std::is_same<int, Args>...>);
        static_assert(sizeof...(Args) <= kNumDimensions);

        int coords[] = {std::forward<int>(c)...};

        // Computing offset first potentially postpones or avoids 64-bit math during addressing
        int offset = 0;
#pragma unroll
        for (int i = 0; i < static_cast<int>(sizeof...(Args)); ++i)
        {
            offset += coords[i] * m_strides[i];
        }

        return reinterpret_cast<const T *>(m_data + offset);
    }

private:
    const std::byte *m_data                    = nullptr;
    int              m_strides[kNumDimensions] = {};
    int              m_shapes[kNumDimensions]  = {};
};

/**
 * FullTensor wrapper class specialized for non-constant value type.
 *
 * @tparam T Type (non-const) of each element inside the tensor wrapper.
 * @tparam N Number of dimensions.
 */
template<typename T, int N>
class FullTensorWrap : public FullTensorWrap<const T, N>
{
    using Base = FullTensorWrap<const T, N>;

public:
    using ValueType = T;

    using Base::kConstantStrides;
    using Base::kNumDimensions;
    using Base::kVariableStrides;

    FullTensorWrap() = default;

    /**
     * Constructs a FullTensorWrap by wrapping a \p data pointer argument.
     *
     * @param[in] data Pointer to the data that will be wrapped
     * @param[in] strides Array of strides in bytes from first to last dimension.
     * @param[in] shapes Array of shapes (number of elements) from first to last dimension.
     *
     * @tparam DataType Type of (inferred from) the data pointer argument.
     */
    template<typename DataType>
    explicit __host__ __device__ FullTensorWrap(DataType *data, const int (&strides)[N], const int (&shapes)[N])
        : Base(data, strides, shapes)
    {
    }

    /**
     * Constructs a FullTensorWrap by wrapping a \p tensor argument.
     *
     * @param[in] tensor Tensor reference to the tensor that will be wrapped.
     */
    __host__ FullTensorWrap(const TensorDataStridedCuda &tensor)
        : Base(tensor)
    {
    }

    // Get strides in bytes for read-only access.
    using Base::strides;

    // Get shapes for read-only access.
    using Base::shapes;

    /**
     * Subscript operator for read-and-write access.
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed.
     *
     * @return Accessed reference.
     */
    template<typename DimType>
    inline __host__ __device__ T &operator[](DimType c) const
    {
        if constexpr (NumElements<DimType> == 1)
        {
            if constexpr (NumComponents<DimType> == 0)
            {
                return *doGetPtr(c);
            }
            else
            {
                return *doGetPtr(c.x);
            }
        }
        else if constexpr (NumElements<DimType> == 2)
        {
            return *doGetPtr(c.y, c.x);
        }
        else if constexpr (NumElements<DimType> == 3)
        {
            return *doGetPtr(c.z, c.y, c.x);
        }
        else if constexpr (NumElements<DimType> == 4)
        {
            return *doGetPtr(c.w, c.z, c.y, c.x);
        }
    }

    /**
     * Get a read-and-write proxy (as pointer) at the Dth dimension.
     *
     * @param[in] c0..D Each coordinate from first to last dimension
     *
     * @return The pointer to the beginning of the Dth dimension
     */
    template<typename... Args>
    inline __host__ __device__ T *ptr(Args... c) const
    {
        return doGetPtr(c...);
    }

protected:
    template<typename... Args>
    inline __host__ __device__ T *doGetPtr(Args... c) const
    {
        return const_cast<T *>(Base::doGetPtr(c...));
    }
};

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_FULL_TENSOR_WRAP_HPP
