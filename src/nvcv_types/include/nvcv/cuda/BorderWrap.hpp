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
 * @file BorderWrap.hpp
 *
 * @brief Defines border wrapper over tensors for border handling.
 */

#ifndef NVCV_CUDA_BORDER_WRAP_HPP
#define NVCV_CUDA_BORDER_WRAP_HPP

#include "TensorWrap.hpp" // for TensorWrap, etc.
#include "TypeTraits.hpp" // for NumElements, etc.

#include <cvcuda/Types.h>       // for NVCVBorderType, etc.
#include <nvcv/ITensorData.hpp> // for ITensorDataStridedCuda, etc.

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_BORDER Border functions
 * @{
 */

/**
 * Function to check if given coordinate is outside range defined by given size
 *
 * @tparam Active Flag to turn this function active
 * @tparam T Type of the values given to this function
 *
 * @param[in] c Coordinate to check if it is outside the range [0, s)
 * @param[in] s Size that defines the inside range [0, s)
 *
 * @return True if given coordinate is outside given size
 */
template<bool Active = true, typename T>
constexpr inline bool __host__ __device__ IsOutside(T c, T s)
{
    if constexpr (Active)
    {
        return (c < 0) || (c >= s);
    }
    return false;
}

/**
 * Function to get a border-aware index considering the range defined by given size
 *
 * @note This function does not work for NVCV_BORDER_CONSTANT
 *
 * @tparam B It is a \ref NVCVBorderType indicating the border to be used
 * @tparam Active Flag to turn this function active
 * @tparam T Type of the values given to this function
 *
 * @param[in] c Coordinate (input index) to put back inside valid range [0, s)
 * @param[in] s Size that defines the valid range [0, s)
 */
template<NVCVBorderType B, bool Active = true, typename T>
constexpr inline T __host__ __device__ GetIndexWithBorder(T c, T s)
{
    static_assert(B != NVCV_BORDER_CONSTANT, "GetIndexWithBorder cannot be used with NVCV_BORDER_CONSTANT");

    if constexpr (Active)
    {
        if constexpr (B == NVCV_BORDER_REPLICATE)
        {
            c = (c < 0) ? 0 : (c >= s ? s - 1 : c);
        }
        else if constexpr (B == NVCV_BORDER_WRAP)
        {
            c = c % s;
            if (c < 0)
            {
                c += s;
            }
        }
        else if constexpr (B == NVCV_BORDER_REFLECT)
        {
            T s2 = s * 2;
            c    = c % s2;
            if (c < 0)
            {
                c += s2;
            }
            c = s - 1 - (abs(2 * c + 1 - s2) >> 1);
        }
        else if constexpr (B == NVCV_BORDER_REFLECT101)
        {
            c = c % (2 * s - 2);
            if (c < 0)
            {
                c += 2 * s - 2;
            }
            c = s - 1 - abs(s - 1 - c);
        }
    }

    return c;
}

/**@}*/

namespace detail {

template<class TensorWrapper, NVCVBorderType B, bool... ActiveDimensions>
class BorderWrapImpl
{
public:
    using TensorWrap = TensorWrapper;
    using ValueType  = typename TensorWrap::ValueType;

    static constexpr int            kNumDimensions = TensorWrap::kNumDimensions;
    static constexpr NVCVBorderType kBorderType    = B;

    static_assert(kNumDimensions == sizeof...(ActiveDimensions));

    static constexpr bool kActiveDimensions[]  = {ActiveDimensions...};
    static constexpr int  kNumActiveDimensions = ((ActiveDimensions ? 1 : 0) + ...);

    struct ActiveMap
    {
        int from[kNumDimensions];

        constexpr ActiveMap()
            : from()
        {
            int j = 0;
            for (int i = 0; i < kNumDimensions; ++i)
            {
                from[i] = kActiveDimensions[i] ? j++ : -1;
            }
        }
    };

    static constexpr ActiveMap kMap{};

    BorderWrapImpl() = default;

    template<typename... Args>
    explicit __host__ __device__ BorderWrapImpl(TensorWrap tensorWrap, Args... tensorShape)
        : m_tensorWrap(tensorWrap)
        , m_tensorShape{std::forward<int>(tensorShape)...}
    {
        static_assert(std::conjunction_v<std::is_same<int, Args>...>);
        static_assert(sizeof...(Args) == kNumActiveDimensions);
    }

    explicit __host__ BorderWrapImpl(const ITensorDataStridedCuda &tensor)
        : m_tensorWrap(tensor)
    {
        int j = 0;
#pragma unroll
        for (int i = 0; i < kNumDimensions; ++i)
        {
            if (kActiveDimensions[i])
            {
                m_tensorShape[j++] = tensor.shape(i);
            }
        }
    }

    inline const __host__ __device__ TensorWrap &tensorWrap() const
    {
        return m_tensorWrap;
    }

    inline __host__ __device__ const int *tensorShape() const
    {
        return m_tensorShape;
    }

    inline __host__ __device__ ValueType borderValue() const
    {
        return ValueType{};
    }

protected:
    const TensorWrap m_tensorWrap                        = {};
    int              m_tensorShape[kNumActiveDimensions] = {0};
};

} // namespace detail

/**
 * @defgroup NVCV_CPP_CUDATOOLS_BORDERWRAP BorderWrap classes
 * @{
 */

/**
 * Border wrapper class used to wrap a \ref TensorWrap adding border handling to it.
 *
 * This class wraps a \ref TensorWrap to add border handling functionality.  It provides the methods \ref ptr and
 * \ref operator[] to do the same semantic access, pointer or reference respectively, in the wrapped TensorWrap but
 * border aware.  It also provides a compile-time set of boolean flags to inform active border-aware dimensions.
 * Active dimensions participate in border handling, storing the corresponding dimension shape.  Inactive
 * dimensions are not checked, the dimension shape is not stored, and thus core dump (or segmentation fault) might
 * happen if accessing outside boundaries of inactive dimensions.
 *
 * @sa NVCV_CPP_CUDATOOLS_BORDERWRAPS
 *
 * @code
 * using DataType = ...;
 * using TensorWrap2D = TensorWrap<-1, -1, DataType>;
 * using BorderWrap2D = BorderWrap<Tensor, NVCV_BORDER_REFLECT, 1, 1>;
 * TensorWrap2D tensorWrap(...);
 * int2 tensorShape = ...;
 * BorderWrap2D borderAwareTensor(tensorWrap, tensorShape.x, tensorShape.y);
 * // Now use borderAwareTensor instead of tensorWrap to access elements inside or outside the tensor,
 * // outside elements use reflect border, that is the outside index is reflected back inside the tensor
 * @endcode
 *
 * @tparam TensorWrapper It is a \ref TensorWrap class with any dimension and type
 * @tparam B It is a \ref NVCVBorderType indicating the border to be used
 * @tparam ActiveDimensions Flags to inform active (true) or inactive (false) dimensions
 */
template<class TensorWrapper, NVCVBorderType B, bool... ActiveDimensions>
class BorderWrap : public detail::BorderWrapImpl<TensorWrapper, B, ActiveDimensions...>
{
    using Base = detail::BorderWrapImpl<TensorWrapper, B, ActiveDimensions...>;

public:
    using typename Base::TensorWrap;
    using typename Base::ValueType;

    using Base::kActiveDimensions;
    using Base::kBorderType;
    using Base::kMap;
    using Base::kNumActiveDimensions;
    using Base::kNumDimensions;

    BorderWrap() = default;

    /**
     * Constructs a BorderWrap by wrapping a \p tensorWrap
     *
     * @param[in] tensorWrap A \ref TensorWrap object to be wrapped
     * @param[in] borderValue The border value is ignored in non-constant border types
     * @param[in] tensorShape0..D Each shape from first to last dimension of the \ref TensorWrap
     */
    template<typename... Args>
    explicit __host__ __device__ BorderWrap(TensorWrap tensorWrap, ValueType borderValue, Args... tensorShape)
        : Base(tensorWrap, tensorShape...)
    {
    }

    /**
     * Constructs a BorderWrap by wrapping a \p tensor
     *
     * @param[in] tensor A \ref ITensorDataStridedCuda object to be wrapped
     * @param[in] borderValue The border value is ignored in non-constant border types
     */
    explicit __host__ BorderWrap(const ITensorDataStridedCuda &tensor, ValueType borderValue = {})
        : Base(tensor)
    {
    }

    // Get the tensor wrapped by this border wrap
    using Base::tensorWrap;

    // Get the shape of the tensor wrapped by this border wrap
    using Base::tensorShape;

    // Get border value of this border wrap, none is stored so an empty value is returned
    using Base::borderValue;

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type)
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed
     *
     * @return Accessed (const) reference
     */
    template<typename DimType>
    inline __host__ __device__ ValueType &operator[](DimType c) const
    {
        constexpr int N = NumElements<DimType>;
        static_assert(kNumDimensions >= N);

        constexpr auto Is = std::make_index_sequence<N>{};

        if constexpr (N == 1)
        {
            return *doGetPtr(Is, c.x);
        }
        else if constexpr (N == 2)
        {
            return *doGetPtr(Is, c.y, c.x);
        }
        else if constexpr (N == 3)
        {
            return *doGetPtr(Is, c.z, c.y, c.x);
        }
        else if constexpr (N == 4)
        {
            return *doGetPtr(Is, c.w, c.z, c.y, c.x);
        }
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the Dth dimension
     *
     * @param[in] c0..D Each coordinate from first to last dimension
     *
     * @return The (const) pointer to the beginning at the Dth dimension
     */
    template<typename... Args>
    inline __host__ __device__ ValueType *ptr(Args... c) const
    {
        return doGetPtr(std::index_sequence_for<Args...>{}, std::forward<Args>(c)...);
    }

private:
    template<typename... Args, std::size_t... Is>
    inline __host__ __device__ ValueType *doGetPtr(std::index_sequence<Is...>, Args... c) const
    {
        return Base::m_tensorWrap.ptr(
            GetIndexWithBorder<kBorderType, kActiveDimensions[Is]>(c, Base::m_tensorShape[kMap.from[Is]])...);
    }
};

/**
 * Border wrapper class specialized for \ref NVCV_BORDER_CONSTANT
 *
 * @tparam TensorWrapper It is a \ref TensorWrap class with any dimension and type
 * @tparam ActiveDimensions Flags to inform active (true) or inactive (false) dimensions
 */
template<class TensorWrapper, bool... ActiveDimensions>
class BorderWrap<TensorWrapper, NVCV_BORDER_CONSTANT, ActiveDimensions...>
    : public detail::BorderWrapImpl<TensorWrapper, NVCV_BORDER_CONSTANT, ActiveDimensions...>
{
    using Base = detail::BorderWrapImpl<TensorWrapper, NVCV_BORDER_CONSTANT, ActiveDimensions...>;

public:
    using typename Base::TensorWrap;
    using typename Base::ValueType;

    using Base::kActiveDimensions;
    using Base::kBorderType;
    using Base::kMap;
    using Base::kNumActiveDimensions;
    using Base::kNumDimensions;

    BorderWrap() = default;

    /**
     * Constructs a BorderWrap by wrapping a \p tensorWrap
     *
     * @param[in] tensorWrap A \ref TensorWrap object to be wrapped
     * @param[in] borderValue The border value to be used when accessing outside the tensor
     * @param[in] tensorShape0..D Each shape from first to last dimension of the \ref TensorWrap
     */
    template<typename... Args>
    explicit __host__ __device__ BorderWrap(TensorWrap tensorWrap, ValueType borderValue, Args... tensorShape)
        : Base(tensorWrap, tensorShape...)
        , m_borderValue(borderValue)
    {
    }

    /**
     * Constructs a BorderWrap by wrapping a \p tensor
     *
     * @param[in] tensor A \ref ITensorDataStridedCuda object to be wrapped
     * @param[in] borderValue The border value to be used when accessing outside the tensor
     */
    explicit __host__ BorderWrap(const ITensorDataStridedCuda &tensor, ValueType borderValue = {})
        : Base(tensor)
        , m_borderValue(borderValue)
    {
    }

    // Get the tensor wrapped by this border wrap
    using Base::tensorWrap;

    // Get the shape of the tensor wrapped by this border wrap
    using Base::tensorShape;

    /**
     * Get the border value of this border wrap
     *
     * @return The border value
     */
    inline __host__ __device__ ValueType borderValue() const
    {
        return m_borderValue;
    }

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type)
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed
     *
     * @return Accessed (const) reference
     */
    template<typename DimType>
    inline __host__ __device__ ValueType &operator[](DimType c) const
    {
        constexpr int N = NumElements<DimType>;
        static_assert(kNumDimensions >= N);

        constexpr auto Is = std::make_index_sequence<N>{};

        ValueType *p = nullptr;

        if constexpr (N == 1)
        {
            p = doGetPtr(Is, c.x);
        }
        else if constexpr (N == 2)
        {
            p = doGetPtr(Is, c.y, c.x);
        }
        else if constexpr (N == 3)
        {
            p = doGetPtr(Is, c.z, c.y, c.x);
        }
        else if constexpr (N == 4)
        {
            p = doGetPtr(Is, c.w, c.z, c.y, c.x);
        }

        if (p == nullptr)
        {
            return m_borderValue;
        }

        return *p;
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the Dth dimension
     *
     * @note This method may return a nullptr pointer when accessing outside the wrapped \ref TensorWrap since this
     * border wrap is for constant border and there is no pointer representation for the constant border value.
     *
     * @param[in] c0..D Each coordinate from first to last dimension
     *
     * @return The (const) pointer to the beginning at the Dth dimension
     */
    template<typename... Args>
    inline __host__ __device__ ValueType *ptr(Args... c) const
    {
        return doGetPtr(std::index_sequence_for<Args...>{}, std::forward<Args>(c)...);
    }

private:
    template<typename... Args, std::size_t... Is>
    inline __host__ __device__ ValueType *doGetPtr(std::index_sequence<Is...>, Args... c) const
    {
        if ((IsOutside<kActiveDimensions[Is]>(c, Base::m_tensorShape[kMap.from[Is]]) || ...))
        {
            return nullptr;
        }
        return Base::m_tensorWrap.ptr(c...);
    }

    const ValueType m_borderValue = SetAll<ValueType>(0);
};

/**@}*/

/**
 *  Specializes \ref BorderWrap template classes for different tensor layouts.
 *
 *  The specializations consider the H (height) and W (width) dimensions as active dimensions and others as
 *  inactive dimensions.  Active dimensions participate in border handling.  Inactive dimensions are not checked,
 *  thus segmentation fault might happen if accessing outside its boundaries.
 *
 *  The supported layouts include the following dimension labels:
 *  - N: number of samples in a batch
 *  - H, W: height and width
 *  - C: number of channels
 *
 *  Template arguments:
 *  - T data type of each element in \ref TensorWrap
 *  - B border extension to be used in active dimensions, one of \ref NVCVBorderType
 *
 *  @sa NVCV_CPP_CUDATOOLS_BORDERWRAP
 *
 *  @defgroup NVCV_CPP_CUDATOOLS_BORDERWRAPS BorderWrap shortcuts
 *  @{
 */

template<typename T, NVCVBorderType B>
using BorderWrapHW = BorderWrap<Tensor2DWrap<T>, B, true, true>;

template<typename T, NVCVBorderType B>
using BorderWrapNHW = BorderWrap<Tensor3DWrap<T>, B, false, true, true>;

template<typename T, NVCVBorderType B>
using BorderWrapNHWC = BorderWrap<Tensor4DWrap<T>, B, false, true, true, false>;

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_BORDER_WRAP_HPP
