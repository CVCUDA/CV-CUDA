/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "FullTensorWrap.hpp" // for FullTensorWrap, etc.
#include "TensorWrap.hpp"     // for TensorWrap, etc.
#include "TypeTraits.hpp"     // for NumElements, etc.

#include <nvcv/BorderType.h>         // for NVCVBorderType, etc.
#include <nvcv/TensorData.hpp>       // for TensorDataStridedCuda, etc.
#include <nvcv/TensorDataAccess.hpp> // for TensorDataAccessStridedImagePlanar, etc.
#include <util/Assert.h>             // for NVCV_ASSERT, etc.

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_BORDER Border functions
 * @{
 */

/**
 * Function to check if given coordinate is outside range defined by given size.
 *
 * @tparam Active Flag to turn this function active.
 * @tparam T Type of the values given to this function.
 *
 * @param[in] c Coordinate to check if it is outside the range [0, s).
 * @param[in] s Size that defines the inside range [0, s).
 *
 * @return True if given coordinate is outside given size.
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
 * Function to get a border-aware index considering the range defined by given size.
 *
 * @note This function does not work for NVCV_BORDER_CONSTANT.
 *
 * @tparam B It is a \ref NVCVBorderType indicating the border to be used.
 * @tparam Active Flag to turn this function active.
 * @tparam T Type of the values given to this function.
 *
 * @param[in] c Coordinate (input index) to put back inside valid range [0, s).
 * @param[in] s Size that defines the valid range [0, s).
 */
template<NVCVBorderType B, bool Active = true, typename T>
constexpr inline T __host__ __device__ GetIndexWithBorder(T c, T s)
{
    static_assert(B != NVCV_BORDER_CONSTANT, "GetIndexWithBorder cannot be used with NVCV_BORDER_CONSTANT");

    if constexpr (Active)
    {
        assert(s > 0);

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
            if (s == 1)
            {
                c = 0;
            }
            else
            {
                c = c % (2 * s - 2);
                if (c < 0)
                {
                    c += 2 * s - 2;
                }
                c = s - 1 - abs(s - 1 - c);
            }
        }

        assert(c >= 0 && c < s);
    }

    return c;
}

/**@}*/

namespace detail {

template<class TW, NVCVBorderType B, bool... ActiveDimensions>
class BorderWrapImpl
{
public:
    using TensorWrapper = TW;
    using ValueType     = typename TensorWrapper::ValueType;

    static constexpr int            kNumDimensions = TensorWrapper::kNumDimensions;
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
                if (kActiveDimensions[i])
                {
                    from[i] = j++;
                }
            }
        }
    };

    static constexpr ActiveMap kMap{};

    BorderWrapImpl() = default;

    template<typename... Args>
    explicit __host__ __device__ BorderWrapImpl(TensorWrapper tensorWrap, Args... tensorShape)
        : m_tensorWrap(tensorWrap)
        , m_tensorShape{std::forward<int>(tensorShape)...}
    {
        if constexpr (sizeof...(Args) == 0)
        {
            static_assert(std::is_base_of_v<TensorWrapper, FullTensorWrap<ValueType, kNumDimensions>>);
            int j = 0;
#pragma unroll
            for (int i = 0; i < kNumDimensions; ++i)
            {
                if (kActiveDimensions[i])
                {
                    m_tensorShape[j++] = tensorWrap.shapes()[i];
                }
            }
        }
        else
        {
            static_assert(std::conjunction_v<std::is_same<int, Args>...>);
            static_assert(sizeof...(Args) == kNumActiveDimensions);
        }
    }

    explicit __host__ BorderWrapImpl(const TensorDataStridedCuda &tensor)
        : m_tensorWrap(tensor)
    {
        NVCV_ASSERT(tensor.rank() >= kNumDimensions);

        int j = 0;
#pragma unroll
        for (int i = 0; i < kNumDimensions; ++i)
        {
            if (kActiveDimensions[i])
            {
                NVCV_ASSERT(tensor.shape(i) <= TypeTraits<int>::max);

                m_tensorShape[j++] = tensor.shape(i);
            }
        }
    }

    inline const __host__ __device__ TensorWrapper &tensorWrap() const
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
    const TensorWrapper m_tensorWrap                        = {};
    int                 m_tensorShape[kNumActiveDimensions] = {0};
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
 * using BorderWrap2D = BorderWrap<TensorWrap2D, NVCV_BORDER_REFLECT, true, true>;
 * TensorWrap2D tensorWrap(...);
 * int2 tensorShape = ...;
 * BorderWrap2D borderAwareTensor(tensorWrap, tensorShape.x, tensorShape.y);
 * // Now use borderAwareTensor instead of tensorWrap to access elements inside or outside the tensor,
 * // outside elements use reflect border, that is the outside index is reflected back inside the tensor
 * @endcode
 *
 * @tparam TW It is a \ref TensorWrap class with any dimension and type.
 * @tparam B It is a \ref NVCVBorderType indicating the border to be used.
 * @tparam ActiveDimensions Flags to inform active (true) or inactive (false) dimensions.
 */
template<class TW, NVCVBorderType B, bool... ActiveDimensions>
class BorderWrap : public detail::BorderWrapImpl<TW, B, ActiveDimensions...>
{
    using Base = detail::BorderWrapImpl<TW, B, ActiveDimensions...>;

public:
    using typename Base::TensorWrapper;
    using typename Base::ValueType;

    using Base::kActiveDimensions;
    using Base::kBorderType;
    using Base::kMap;
    using Base::kNumActiveDimensions;
    using Base::kNumDimensions;

    BorderWrap() = default;

    /**
     * Constructs a BorderWrap by wrapping a \p tensorWrap.
     *
     * @param[in] tensorWrap A \ref TensorWrap or \ref FullTensorWrap object to be wrapped.
     * @param[in] borderValue The border value is ignored in non-constant border types.
     * @param[in] tensorShape0..D Each shape from first to last dimension of the \ref TensorWrap.
     *                            This may be empty in case of wrapping a \ref FullTensorWrap.
     */
    template<typename... Args>
    explicit __host__ __device__ BorderWrap(TensorWrapper tensorWrap, ValueType borderValue, Args... tensorShape)
        : Base(tensorWrap, tensorShape...)
    {
    }

    /**
     * Constructs a BorderWrap by wrapping a \p tensor.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value is ignored in non-constant border types.
     */
    explicit __host__ BorderWrap(const TensorDataStridedCuda &tensor, ValueType borderValue = {})
        : Base(tensor)
    {
    }

    // Get the tensor wrapped by this border wrap.
    using Base::tensorWrap;

    // Get the shape of the tensor wrapped by this border wrap.
    using Base::tensorShape;

    // Get border value of this border wrap, none is stored so an empty value is returned.
    using Base::borderValue;

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type).
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed.
     *
     * @return Accessed reference.
     */
    template<typename DimType, class = Require<std::is_same_v<int, BaseType<DimType>>>>
    inline __host__ __device__ ValueType &operator[](DimType c) const
    {
        constexpr int N = NumElements<DimType>;
        static_assert(kNumDimensions >= N);

        constexpr auto Is = std::make_index_sequence<N>{};

        if constexpr (N == 1)
        {
            if constexpr (NumComponents<DimType> == 0)
            {
                return *doGetPtr(Is, c);
            }
            else
            {
                return *doGetPtr(Is, c.x);
            }
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
     * Get a read-only or read-and-write proxy (as pointer) at the Dth dimension.
     *
     * @param[in] c0..D Each coordinate from first to last dimension.
     *
     * @return The (const) pointer to the beginning at the Dth dimension.
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
 * Border wrapper class specialized for \ref NVCV_BORDER_CONSTANT.
 *
 * @tparam TW It is a \ref TensorWrap class with any dimension and type.
 * @tparam ActiveDimensions Flags to inform active (true) or inactive (false) dimensions.
 */
template<class TW, bool... ActiveDimensions>
class BorderWrap<TW, NVCV_BORDER_CONSTANT, ActiveDimensions...>
    : public detail::BorderWrapImpl<TW, NVCV_BORDER_CONSTANT, ActiveDimensions...>
{
    using Base = detail::BorderWrapImpl<TW, NVCV_BORDER_CONSTANT, ActiveDimensions...>;

public:
    using typename Base::TensorWrapper;
    using typename Base::ValueType;

    using Base::kActiveDimensions;
    using Base::kBorderType;
    using Base::kMap;
    using Base::kNumActiveDimensions;
    using Base::kNumDimensions;

    BorderWrap() = default;

    /**
     * Constructs a BorderWrap by wrapping a \p tensorWrap.
     *
     * @param[in] tensorWrap A \ref TensorWrap or \ref FullTensorWrap object to be wrapped.
     * @param[in] borderValue The border value to be used when accessing outside the tensor.
     * @param[in] tensorShape0..D Each shape from first to last dimension of the \ref TensorWrap.
     *                            This may be empty in case of wrapping a \ref FullTensorWrap.
     */
    template<typename... Args>
    explicit __host__ __device__ BorderWrap(TensorWrapper tensorWrap, ValueType borderValue, Args... tensorShape)
        : Base(tensorWrap, tensorShape...)
        , m_borderValue(borderValue)
    {
    }

    /**
     * Constructs a BorderWrap by wrapping a \p tensor.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value to be used when accessing outside the tensor.
     */
    explicit __host__ BorderWrap(const TensorDataStridedCuda &tensor, ValueType borderValue = {})
        : Base(tensor)
        , m_borderValue(borderValue)
    {
    }

    // Get the tensor wrapped by this border wrap.
    using Base::tensorWrap;

    // Get the shape of the tensor wrapped by this border wrap.
    using Base::tensorShape;

    /**
     * Get the border value of this border wrap.
     *
     * @return The border value.
     */
    inline __host__ __device__ ValueType borderValue() const
    {
        return m_borderValue;
    }

    /**
     * Subscript operator for read-only access.
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed.
     *
     * @return Accessed const reference.
     */
    template<typename DimType, class = Require<std::is_same_v<int, BaseType<DimType>>>>
    inline const __host__ __device__ ValueType &operator[](DimType c) const
    {
        constexpr int N = NumElements<DimType>;
        static_assert(kNumDimensions >= N);

        constexpr auto Is = std::make_index_sequence<N>{};

        const ValueType *p = nullptr;

        if constexpr (N == 1)
        {
            if constexpr (NumComponents<DimType> == 0)
            {
                p = doGetPtr(Is, c);
            }
            else
            {
                p = doGetPtr(Is, c.x);
            }
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
     * Get a read-only or read-and-write proxy (as pointer) at the Dth dimension.
     *
     * @note This method may return a nullptr pointer when accessing outside the wrapped \ref TensorWrap since this
     * border wrap is for constant border and there is no pointer representation for the constant border value.
     *
     * @param[in] c0..D Each coordinate from first to last dimension.
     *
     * @return The (const) pointer to the beginning at the Dth dimension.
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
 * Factory function to create an NHW border wrap given a tensor data.
 *
 * The output \ref BorderWrap wraps an NHW 3D tensor allowing to access data per batch (N), per row (H) and per
 * column (W) of the input tensor border aware in rows (or height H) and columns (or width W).  The input tensor
 * data must have either NHWC or HWC layout, where the channel C is inside the given template type \p T,
 * e.g. T=uchar4 for RGBA8.  The active dimensions are H (second) and W (third).
 *
 * @sa NVCV_CPP_CUDATOOLS_BORDERWRAP
 *
 * @tparam T Type of the values to be accessed in the border wrap.
 * @tparam B Border extension to be used when accessing H and W, one of \ref NVCVBorderType
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 * @param[in] borderValue Border value to be used when accessing outside elements in constant border type
 *
 * @return Border wrap useful to access tensor data border aware in H and W in CUDA kernels.
 */
template<typename T, NVCVBorderType B, class = Require<HasTypeTraits<T>>>
__host__ auto CreateBorderWrapNHW(const TensorDataStridedCuda &tensor, T borderValue = {})
{
    auto tensorAccess = TensorDataAccessStridedImagePlanar::Create(tensor);
    NVCV_ASSERT(tensorAccess);
    NVCV_ASSERT(tensorAccess->numRows() <= TypeTraits<int>::max);
    NVCV_ASSERT(tensorAccess->numCols() <= TypeTraits<int>::max);

    auto tensorWrap = CreateTensorWrapNHW<T>(tensor);

    return BorderWrap<decltype(tensorWrap), B, false, true, true>(
        tensorWrap, borderValue, static_cast<int>(tensorAccess->numRows()), static_cast<int>(tensorAccess->numCols()));
}

/**
 * Factory function to create an NHWC border wrap given a tensor data.
 *
 * The output \ref BorderWrap wraps an NHWC 4D tensor allowing to access data per batch (N), per row (H), per
 * column (W) and per channel (C) of the input tensor border aware in rows (or height H) and columns (or width W).
 * The input tensor data must have either NHWC or HWC layout, where the channel C is of type \p T, e.g. T=uchar for
 * each channel of either RGB8 or RGBA8.  The active dimensions are H (second) and W (third).
 *
 * @sa NVCV_CPP_CUDATOOLS_BORDERWRAP
 *
 * @tparam T Type of the values to be accessed in the border wrap.
 * @tparam B Border extension to be used when accessing H and W, one of \ref NVCVBorderType
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 * @param[in] borderValue Border value to be used when accessing outside elements in constant border type
 *
 * @return Border wrap useful to access tensor data border aware in H and W in CUDA kernels.
 */
template<typename T, NVCVBorderType B, class = Require<HasTypeTraits<T>>>
__host__ auto CreateBorderWrapNHWC(const TensorDataStridedCuda &tensor, T borderValue = {})
{
    auto tensorAccess = TensorDataAccessStridedImagePlanar::Create(tensor);
    NVCV_ASSERT(tensorAccess);
    NVCV_ASSERT(tensorAccess->numRows() <= TypeTraits<int>::max);
    NVCV_ASSERT(tensorAccess->numCols() <= TypeTraits<int>::max);

    auto tensorWrap = CreateTensorWrapNHWC<T>(tensor);

    return BorderWrap<decltype(tensorWrap), B, false, true, true, false>(
        tensorWrap, borderValue, static_cast<int>(tensorAccess->numRows()), static_cast<int>(tensorAccess->numCols()));
}

} // namespace nvcv::cuda

#endif // NVCV_CUDA_BORDER_WRAP_HPP
