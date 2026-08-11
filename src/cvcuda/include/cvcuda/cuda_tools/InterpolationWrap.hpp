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

/**
 * @file InterpolationWrap.hpp
 *
 * @brief Defines interpolation wrapper over tensors for interpolation handling.
 */

#ifndef NVCV_CUDA_INTERPOLATION_WRAP_HPP
#define NVCV_CUDA_INTERPOLATION_WRAP_HPP

#include "BorderWrap.hpp"   // for TensorWrap, etc.
#include "MathWrappers.hpp" // for round, etc.
#include "SaturateCast.hpp" // for SaturateCast, etc.

#include <cvcuda/Types.h>      // for NVCVInterpolationType, etc.
#include <nvcv/TensorData.hpp> // for TensorDataStridedCuda, etc.

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_INTERPOLATION Interpolation functions
 * @{
 */

/**
 * Function to get an integer index from a float coordinate for interpolation purpose.
 *
 * @note When \p IndexType is an integer type, the input coordinate is first clamped to
 * `[TypeTraits<IndexType>::min / 4, TypeTraits<IndexType>::max / 4]` to guarantee that
 * common downstream arithmetic performed by callers stays within \p IndexType's range:
 * `x + 1`, `ix + cx` (CUBIC, `cx` in `[-1, 2]`), `xmin - 1`, loop `++cx`, as well as
 * \ref GetIndexWithBorder's `2*c + 1 - s2` (REFLECT) and `c % (2*s - 2)` (REFLECT101).
 * Without this clamp, an ill-conditioned transform producing `|c| >= IndexType::max`
 * would let `__float2{int,ll}_rd` saturate at `IndexType::max`; the subsequent `x + 1`
 * signed-overflows, and the resulting undefined behaviour lets the compiler elide the
 * bounds check in \ref GetIndexWithBorder for REPLICATE borders, leading to a
 * `cudaErrorIllegalAddress` when the wild index is dereferenced by \ref TensorWrap::ptr
 * (CVCUDA issue #249). For any realistic pixel offset the clamp is a no-op; only
 * pathological coordinates (orders of magnitude outside any real tensor) are clipped,
 * in which case the returned index is well-defined but not mathematically exact.
 *
 * @tparam I Interpolation type, one of \ref NVCVInterpolationType.
 * @tparam Position Interpolation position, 1 for the first index and 2 for the second index.
 * @tparam IndexType Type of the returned value. Defaults to `int64_t`; callers storing
 *                   the result in a narrower integer type (e.g. `int32_t`) should pass
 *                   that type here so the overflow-prevention clamp targets the final
 *                   storage width.
 *
 * @param[in] c Coordinate in floating-point to convert to index in integer.
 *
 * @return Index in integer suitable for interpolation computation.
 */
template<NVCVInterpolationType I, int Position = 1, typename IndexType = int64_t>
constexpr IndexType __host__ __device__ GetIndexForInterpolation(float c)
{
    static_assert(
        I == NVCV_INTERP_NEAREST || I == NVCV_INTERP_LINEAR || I == NVCV_INTERP_CUBIC || I == NVCV_INTERP_AREA,
        "GetIndexForInterpolation accepts only NVCV_INTERP_{NEAREST, LINEAR, CUBIC, AREA}");
    static_assert(Position == 1 || Position == 2, "GetIndexForInterpolation accepts only position 1 or 2");

    // Overflow-prevention clamp for downstream integer arithmetic; see the @note above
    // for the rationale.
    if constexpr (std::is_integral_v<IndexType>)
    {
        constexpr auto kSafeMax = static_cast<float>(TypeTraits<IndexType>::max / 4);
        constexpr auto kSafeMin = static_cast<float>(TypeTraits<IndexType>::min / 4);
        c                       = cuda::max(cuda::min(c, kSafeMax), kSafeMin);
    }

    if constexpr (I == NVCV_INTERP_NEAREST)
    {
        return cuda::round<RoundMode::DOWN, IndexType>(c);
    }
    else if constexpr (I == NVCV_INTERP_LINEAR)
    {
        return cuda::round<RoundMode::DOWN, IndexType>(c);
    }
    else if constexpr (I == NVCV_INTERP_CUBIC)
    {
        return cuda::round<RoundMode::DOWN, IndexType>(c);
    }
    else if constexpr (I == NVCV_INTERP_AREA)
    {
        if constexpr (Position == 1)
            return cuda::round<RoundMode::UP, IndexType>(c);
        else if constexpr (Position == 2)
            return cuda::round<RoundMode::DOWN, IndexType>(c);
    }

    return static_cast<IndexType>(c);
}

inline void __host__ __device__ GetCubicCoeffs(float delta, float &w0, float &w1, float &w2, float &w3)
{
    w0 = -.5f;
    w0 = w0 * delta + 1.f;
    w0 = w0 * delta - .5f;
    w0 = w0 * delta;

    w1 = 1.5f;
    w1 = w1 * delta - 2.5f;
    w1 = w1 * delta;
    w1 = w1 * delta + 1.f;

    w2 = -1.5f;
    w2 = w2 * delta + 2.f;
    w2 = w2 * delta + .5f;
    w2 = w2 * delta;

    w3 = 1 - w0 - w1 - w2;
}

/**@}*/

namespace detail {

template<class BW, NVCVInterpolationType I>
class InterpolationWrapImpl
{
public:
    using BorderWrapper = BW;
    using TensorWrapper = typename BorderWrapper::TensorWrapper;
    using ValueType     = typename BorderWrapper::ValueType;
    using StrideType    = typename BorderWrapper::StrideType;

    static constexpr int                   kNumDimensions     = BorderWrapper::kNumDimensions;
    static constexpr NVCVInterpolationType kInterpolationType = I;

    static constexpr auto kActiveDimensions    = BorderWrapper::kActiveDimensions;
    static constexpr int  kNumActiveDimensions = BorderWrapper::kNumActiveDimensions;

    static_assert(kNumActiveDimensions == 2, "InterpolationWrap can only wrap a BorderWrap with 2 active dimensions");

    struct ActiveCoordMap
    {
        int id[kNumDimensions] = {}; // NOSONAR: constexpr device lookup table.

        constexpr ActiveCoordMap()
        {
            int dimCoord = 0;
            for (int dim = kNumDimensions - 1, idCoord = 0; dim >= 0; dim--, idCoord++)
            {
                if (kActiveDimensions[dim])
                {
                    id[dimCoord++] = idCoord;
                }
            }
            for (int dim = 0, idCoord = kNumDimensions - 1; dim < kNumDimensions; dim++, idCoord--)
            {
                if (!kActiveDimensions[dim])
                {
                    id[dimCoord++] = idCoord;
                }
            }
        }
    };

    static constexpr ActiveCoordMap kCoordMap{};

    InterpolationWrapImpl() = default;

    explicit __host__ InterpolationWrapImpl(const TensorDataStridedCuda &tensor, ValueType borderValue = {})
        : m_borderWrap(tensor, borderValue)
    {
    }

    template<typename... Args>
    explicit __host__ __device__ InterpolationWrapImpl(TensorWrapper tensorWrap, ValueType borderValue,
                                                       Args... tensorShape)
        : m_borderWrap(tensorWrap, borderValue, tensorShape...)
    {
    }

    explicit __host__ __device__ InterpolationWrapImpl(BorderWrapper borderWrap)
        : m_borderWrap(borderWrap)
    {
    }

    inline const __host__ __device__ BorderWrapper &borderWrap() const
    {
        return m_borderWrap;
    }

    inline __host__ __device__ float scaleX() const
    {
        return float{};
    }

    inline __host__ __device__ float scaleY() const
    {
        return float{};
    }

    inline __host__ __device__ bool isIntegerArea() const
    {
        return bool{};
    }

protected:
    template<typename DimType>
    inline const __host__ __device__ ValueType &doGetValue(DimType c, StrideType x, StrideType y) const
    {
        cuda::ConvertBaseTypeTo<int, DimType> ic;
        GetElement<kCoordMap.id[0]>(ic) = static_cast<int>(x);
        GetElement<kCoordMap.id[1]>(ic) = static_cast<int>(y);
        if constexpr (NumElements<DimType> >= 3)
            GetElement<kCoordMap.id[2]>(ic) = static_cast<int>(GetElement<kCoordMap.id[2]>(c));
        if constexpr (NumElements<DimType> == 4)
            GetElement<kCoordMap.id[3]>(ic) = static_cast<int>(GetElement<kCoordMap.id[3]>(c));

        return m_borderWrap[ic];
    }

    const BorderWrapper m_borderWrap = {};
};

} // namespace detail

/**
 * @defgroup NVCV_CPP_CUDATOOLS_INTERPOLATIONWRAP InterpolationWrap classes
 * @{
 */

/**
 * Interpolation wrapper class used to wrap a \ref BorderWrap adding interpolation handling to it.
 *
 * This class wraps a \ref BorderWrap to add interpolation handling functionality.  It provides the \ref operator[]
 * to do the same semantic value access in the wrapped BorderWrap but interpolation aware.
 *
 * @note Each interpolation wrap class below is specialized for one interpolation type.
 *
 * @sa NVCV_CPP_CUDATOOLS_INTERPOLATIONWRAPS
 *
 * @code
 * using DataType = ...;
 * using TensorWrap2D = TensorWrap<-1, -1, DataType>;
 * using BorderWrap2D = BorderWrap<TensorWrap2D, NVCV_BORDER_REFLECT, true, true>;
 * using InterpWrap2D = InterpolationWrap<BorderWrap2D, NVCV_INTERP_CUBIC>;
 * TensorWrap2D tensorWrap(...);
 * BorderWrap2D borderWrap(...);
 * InterpWrap2D interpolationAwareTensor(borderWrap);
 * // Now use interpolationAwareTensor instead of borderWrap or tensorWrap to access in-between elements with
 * // on-the-fly interpolation, in this example grid-unaligned pixels use bi-cubic interpolation, grid-aligned
 * // pixels that fall outside the tensor use reflect border extension and inside is the tensor value itself
 * @endcode
 *
 * @tparam BW It is a \ref BorderWrap class with any dimension and type.
 * @tparam I It is a \ref NVCVInterpolationType defining the interpolation type to be used.
 */
template<class BW, NVCVInterpolationType I>
class InterpolationWrap : public detail::InterpolationWrapImpl<BW, I>
{
};

/**
 * Interpolation wrapper class specialized for \ref NVCV_INTERP_NEAREST.
 *
 * @tparam BW It is a \ref BorderWrap class with any dimension and type.
 */
template<class BW>
class InterpolationWrap<BW, NVCV_INTERP_NEAREST> : public detail::InterpolationWrapImpl<BW, NVCV_INTERP_NEAREST>
{
    using Base = detail::InterpolationWrapImpl<BW, NVCV_INTERP_NEAREST>;

public:
    using typename Base::BorderWrapper;
    using typename Base::StrideType;
    using typename Base::TensorWrapper;
    using typename Base::ValueType;

    using Base::kCoordMap;
    using Base::kInterpolationType;
    using Base::kNumDimensions;

    InterpolationWrap() = default;

    /**
     * Constructs an InterpolationWrap by wrapping a \p tensor.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ InterpolationWrap(const TensorDataStridedCuda &tensor, ValueType borderValue = {},
                                        float scaleX = {}, float scaleY = {})
        : Base(tensor, borderValue)
    {
    }

    /**
     * Constructs an InterpolationWrap by wrapping a \p tensorWrap.
     *
     * @param[in] tensorWrap A \ref TensorWrap object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    template<typename... Args>
    explicit __host__ __device__ InterpolationWrap(TensorWrapper tensorWrap, ValueType borderValue, float scaleX,
                                                   float scaleY, Args... tensorShape)
        : Base(tensorWrap, borderValue, tensorShape...)
    {
    }

    /**
     * Constructs an InterpolationWrap by wrapping a \p borderWrap.
     *
     * @param[in] borderWrap A \ref BorderWrap object to be wrapped.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ __device__ InterpolationWrap(BorderWrapper borderWrap, float scaleX = {}, float scaleY = {})
        : Base(borderWrap)
    {
    }

    // Get the border wrap wrapped by this interpolation wrap.
    using Base::borderWrap;

    // Get the scale X of this interpolation wrap, none is stored so an empty value is returned.
    using Base::scaleX;

    // Get the scale Y of this interpolation wrap, none is stored so an empty value is returned.
    using Base::scaleY;

    // True if this interpolation wrap is integer Area, none is stored so an empty value is returned.
    using Base::isIntegerArea;

    /**
     * Subscript operator for interpolated value access.
     *
     * @param[in] c N-D float coordinate (from last to first dimension) to be accessed with interpolation.
     *
     * @return Accessed interpolated value.
     */
    template<
        typename DimType,
        class = Require<std::is_same_v<BaseType<DimType>,
                                       float> && 2 <= NumElements<DimType> && NumElements<DimType> <= kNumDimensions>>
    inline __host__ __device__ ValueType operator[](DimType c) const
    {
        const StrideType x
            = GetIndexForInterpolation<kInterpolationType, 1, StrideType>(GetElement<kCoordMap.id[0]>(c) + .5f);
        const StrideType y
            = GetIndexForInterpolation<kInterpolationType, 1, StrideType>(GetElement<kCoordMap.id[1]>(c) + .5f);

        return Base::doGetValue(c, x, y);
    }
};

/**
 * Interpolation wrapper class specialized for \ref NVCV_INTERP_LINEAR.
 *
 * @tparam BW It is a \ref BorderWrap class with any dimension and type.
 */
template<class BW>
class InterpolationWrap<BW, NVCV_INTERP_LINEAR> : public detail::InterpolationWrapImpl<BW, NVCV_INTERP_LINEAR>
{
    using Base = detail::InterpolationWrapImpl<BW, NVCV_INTERP_LINEAR>;

public:
    using typename Base::BorderWrapper;
    using typename Base::StrideType;
    using typename Base::TensorWrapper;
    using typename Base::ValueType;

    using Base::kCoordMap;
    using Base::kInterpolationType;
    using Base::kNumDimensions;

    InterpolationWrap() = default;

    /**
     * Constructs an InterpolationWrap by wrapping a \p tensor.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ InterpolationWrap(const TensorDataStridedCuda &tensor, ValueType borderValue = {},
                                        float scaleX = {}, float scaleY = {})
        : Base(tensor, borderValue)
    {
    }

    /**
     * Constructs an InterpolationWrap by wrapping a \p tensorWrap.
     *
     * @param[in] tensorWrap A \ref TensorWrap object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    template<typename... Args>
    explicit __host__ __device__ InterpolationWrap(TensorWrapper tensorWrap, ValueType borderValue, float scaleX,
                                                   float scaleY, Args... tensorShape)
        : Base(tensorWrap, borderValue, tensorShape...)
    {
    }

    /**
     * Constructs an InterpolationWrap by wrapping a \p borderWrap.
     *
     * @param[in] borderWrap A \ref BorderWrap object to be wrapped.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ __device__ InterpolationWrap(BorderWrapper borderWrap, float scaleX = {}, float scaleY = {})
        : Base(borderWrap)
    {
    }

    // Get the border wrap wrapped by this interpolation wrap.
    using Base::borderWrap;

    // Get the scale X of this interpolation wrap, none is stored so an empty value is returned.
    using Base::scaleX;

    // Get the scale Y of this interpolation wrap, none is stored so an empty value is returned.
    using Base::scaleY;

    // True if this interpolation wrap is integer Area, none is stored so an empty value is returned.
    using Base::isIntegerArea;

    /**
     * Subscript operator for interpolated value access.
     *
     * @param[in] c N-D float coordinate (from last to first dimension) to be accessed with interpolation.
     *
     * @return Accessed interpolated value.
     */
    template<
        typename DimType,
        class = Require<std::is_same_v<BaseType<DimType>,
                                       float> && 2 <= NumElements<DimType> && NumElements<DimType> <= kNumDimensions>>
    inline __host__ __device__ ValueType operator[](DimType c) const
    {
        const float      x  = GetElement<kCoordMap.id[0]>(c);
        const float      y  = GetElement<kCoordMap.id[1]>(c);
        const StrideType x1 = GetIndexForInterpolation<kInterpolationType, 1, StrideType>(x);
        const StrideType x2 = x1 + 1;
        const StrideType y1 = GetIndexForInterpolation<kInterpolationType, 1, StrideType>(y);
        const StrideType y2 = y1 + 1;

        auto out = SetAll<ConvertBaseTypeTo<float, std::remove_cv_t<ValueType>>>(0);

        out += Base::doGetValue(c, x1, y1) * (static_cast<float>(x2) - x) * (static_cast<float>(y2) - y);
        out += Base::doGetValue(c, x2, y1) * (x - static_cast<float>(x1)) * (static_cast<float>(y2) - y);
        out += Base::doGetValue(c, x1, y2) * (static_cast<float>(x2) - x) * (y - static_cast<float>(y1));
        out += Base::doGetValue(c, x2, y2) * (x - static_cast<float>(x1)) * (y - static_cast<float>(y1));

        return SaturateCast<ValueType>(out);
    }
};

/**
 * Interpolation wrapper class specialized for \ref NVCV_INTERP_CUBIC.
 *
 * @tparam BW It is a \ref BorderWrap class with any dimension and type.
 */
template<class BW>
class InterpolationWrap<BW, NVCV_INTERP_CUBIC> : public detail::InterpolationWrapImpl<BW, NVCV_INTERP_CUBIC>
{
    using Base = detail::InterpolationWrapImpl<BW, NVCV_INTERP_CUBIC>;

public:
    using typename Base::BorderWrapper;
    using typename Base::StrideType;
    using typename Base::TensorWrapper;
    using typename Base::ValueType;

    using Base::kCoordMap;
    using Base::kInterpolationType;
    using Base::kNumDimensions;

    InterpolationWrap() = default;

    /**
     * Constructs an InterpolationWrap by wrapping a \p tensor.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ InterpolationWrap(const TensorDataStridedCuda &tensor, ValueType borderValue = {},
                                        float scaleX = {}, float scaleY = {})
        : Base(tensor, borderValue)
    {
    }

    /**
     * Constructs an InterpolationWrap by wrapping a \p tensorWrap.
     *
     * @param[in] tensorWrap A \ref TensorWrap object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    template<typename... Args>
    explicit __host__ __device__ InterpolationWrap(TensorWrapper tensorWrap, ValueType borderValue, float scaleX,
                                                   float scaleY, Args... tensorShape)
        : Base(tensorWrap, borderValue, tensorShape...)
    {
    }

    /**
     * Constructs an InterpolationWrap by wrapping a \p borderWrap.
     *
     * @param[in] borderWrap A \ref BorderWrap object to be wrapped.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ __device__ InterpolationWrap(BorderWrapper borderWrap, float scaleX = {}, float scaleY = {})
        : Base(borderWrap)
    {
    }

    // Get the border wrap wrapped by this interpolation wrap.
    using Base::borderWrap;

    // Get the scale X of this interpolation wrap, none is stored so an empty value is returned.
    using Base::scaleX;

    // Get the scale Y of this interpolation wrap, none is stored so an empty value is returned.
    using Base::scaleY;

    // True if this interpolation wrap is integer Area, none is stored so an empty value is returned.
    using Base::isIntegerArea;

    /**
     * Subscript operator for interpolated value access.
     *
     * @param[in] c N-D float coordinate (from last to first dimension) to be accessed with interpolation.
     *
     * @return Accessed interpolated value.
     */
    template<
        typename DimType,
        class = Require<std::is_same_v<BaseType<DimType>,
                                       float> && 2 <= NumElements<DimType> && NumElements<DimType> <= kNumDimensions>>
    inline __host__ __device__ ValueType operator[](DimType c) const
    {
        const float      x  = GetElement<kCoordMap.id[0]>(c);
        const float      y  = GetElement<kCoordMap.id[1]>(c);
        const StrideType ix = GetIndexForInterpolation<kInterpolationType, 1, StrideType>(x);
        const StrideType iy = GetIndexForInterpolation<kInterpolationType, 1, StrideType>(y);

        float wx[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
        GetCubicCoeffs(x - static_cast<float>(ix), wx[0], wx[1], wx[2], wx[3]);
        float wy[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
        GetCubicCoeffs(y - static_cast<float>(iy), wy[0], wy[1], wy[2], wy[3]);

        using FT = ConvertBaseTypeTo<float, std::remove_cv_t<ValueType>>;
        auto sum = SetAll<FT>(0);

#pragma unroll
        for (StrideType cy = -1; cy <= 2; cy++)
        {
#pragma unroll
            for (StrideType cx = -1; cx <= 2; cx++)
            {
                sum += Base::doGetValue(c, ix + cx, iy + cy) * (wx[cx + 1] * wy[cy + 1]);
            }
        }

        return SaturateCast<ValueType>(sum);
    }
};

/**
 * Interpolation wrapper class specialized for \ref NVCV_INTERP_AREA.
 *
 * @tparam BW It is a \ref BorderWrap class with any dimension and type.
 */
template<class BW>
class InterpolationWrap<BW, NVCV_INTERP_AREA> : public detail::InterpolationWrapImpl<BW, NVCV_INTERP_AREA>
{
    using Base = detail::InterpolationWrapImpl<BW, NVCV_INTERP_AREA>;

public:
    using typename Base::BorderWrapper;
    using typename Base::StrideType;
    using typename Base::TensorWrapper;
    using typename Base::ValueType;

    using Base::kCoordMap;
    using Base::kInterpolationType;
    using Base::kNumDimensions;

    InterpolationWrap() = default;

    /**
     * Constructs an InterpolationWrap by wrapping a \p tensor.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value for Area interpolation.
     * @param[in] scaleY The scale Y value for Area interpolation.
     */
    explicit __host__ InterpolationWrap(const TensorDataStridedCuda &tensor, ValueType borderValue = {},
                                        float scaleX = {}, float scaleY = {})
        : Base(tensor, borderValue)
        , m_scaleX(scaleX)
        , m_scaleY(scaleY)
        , m_isIntegerArea(isIntegerArea(scaleX, scaleY))
    {
    }

    /**
     * Constructs an InterpolationWrap by wrapping a \p tensorWrap.
     *
     * @param[in] tensorWrap A \ref TensorWrap object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value for Area interpolation.
     * @param[in] scaleY The scale Y value for Area interpolation.
     */
    template<typename... Args>
    explicit __host__ __device__ InterpolationWrap(TensorWrapper tensorWrap, ValueType borderValue, float scaleX,
                                                   float scaleY, Args... tensorShape)
        : Base(tensorWrap, borderValue, tensorShape...)
        , m_scaleX(scaleX)
        , m_scaleY(scaleY)
        , m_isIntegerArea(isIntegerArea(scaleX, scaleY))
    {
    }

    /**
     * Constructs an InterpolationWrap by wrapping a \p borderWrap.
     *
     * @param[in] borderWrap A \ref BorderWrap object to be wrapped.
     * @param[in] scaleX The scale X value for Area interpolation.
     * @param[in] scaleY The scale Y value for Area interpolation.
     */
    explicit __host__ __device__ InterpolationWrap(BorderWrapper borderWrap, float scaleX = {}, float scaleY = {})
        : Base(borderWrap)
        , m_scaleX(scaleX)
        , m_scaleY(scaleY)
        , m_isIntegerArea(isIntegerArea(scaleX, scaleY))
    {
    }

    // Get the border wrap wrapped by this interpolation wrap.
    using Base::borderWrap;

    // Get the Area scale X of this interpolation wrap.
    inline __host__ __device__ float scaleX() const
    {
        return m_scaleX;
    }

    // Get the Area scale Y of this interpolation wrap.
    inline __host__ __device__ float scaleY() const
    {
        return m_scaleY;
    }

    // True if this interpolation wrap is integer Area.
    inline __host__ __device__ bool isIntegerArea() const
    {
        return m_isIntegerArea;
    }

    /**
     * Subscript operator for interpolated value access.
     *
     * @param[in] c N-D float coordinate (from last to first dimension) to be accessed with interpolation.
     *
     * @return Accessed interpolated value.
     */
    template<
        typename DimType,
        class = Require<std::is_same_v<BaseType<DimType>,
                                       float> && 2 <= NumElements<DimType> && NumElements<DimType> <= kNumDimensions>>
    inline __host__ __device__ ValueType operator[](DimType c) const
    {
        return interpolateWithBounds(c, computeAreaBounds(c));
    }

    // Area box geometry for an output coordinate. It depends only on the active spatial coordinate
    // (x, y), not on any inactive channel/sample/batch dimension, so a caller resizing several planes
    // at the same (x, y) can compute it once and reuse it across planes -- avoiding the per-plane
    // recomputation that one operator[] call per plane would incur.
    struct AreaBounds
    {
        float      fsx1;
        float      fsy1;
        float      fsx2;
        float      fsy2;
        StrideType xmin;
        StrideType xmax;
        StrideType ymin;
        StrideType ymax;
    };

    // Compute the reusable area box geometry for coordinate \p c (only its spatial elements matter).
    template<
        typename DimType,
        class = Require<std::is_same_v<BaseType<DimType>,
                                       float> && 2 <= NumElements<DimType> && NumElements<DimType> <= kNumDimensions>>
    inline __host__ __device__ AreaBounds computeAreaBounds(DimType c) const
    {
        return MakeAreaBounds(c);
    }

    // Interpolate at \p c using bounds from computeAreaBounds() for the SAME (x, y); \p c selects the
    // plane/sample. operator[](c) is exactly interpolateWithBounds(c, computeAreaBounds(c)).
    template<
        typename DimType,
        class = Require<std::is_same_v<BaseType<DimType>,
                                       float> && 2 <= NumElements<DimType> && NumElements<DimType> <= kNumDimensions>>
    inline __host__ __device__ ValueType interpolateWithBounds(DimType c, const AreaBounds &bounds) const
    {
        auto out = SetAll<ConvertBaseTypeTo<float, std::remove_cv_t<ValueType>>>(0);

        if (m_isIntegerArea)
        {
            AddIntegerArea(c, bounds, out);
        }
        else
        {
            AddFractionalArea(c, bounds, out);
        }

        return SaturateCast<ValueType>(out);
    }

private:
    template<typename DimType>
    inline __host__ __device__ AreaBounds MakeAreaBounds(DimType c) const
    {
        const float fsx1 = GetElement<kCoordMap.id[0]>(c) * m_scaleX;
        const float fsy1 = GetElement<kCoordMap.id[1]>(c) * m_scaleY;
        const float fsx2 = fsx1 + m_scaleX;
        const float fsy2 = fsy1 + m_scaleY;

        return {fsx1,
                fsy1,
                fsx2,
                fsy2,
                GetIndexForInterpolation<kInterpolationType, 1, StrideType>(fsx1),
                GetIndexForInterpolation<kInterpolationType, 2, StrideType>(fsx2),
                GetIndexForInterpolation<kInterpolationType, 1, StrideType>(fsy1),
                GetIndexForInterpolation<kInterpolationType, 2, StrideType>(fsy2)};
    }

    inline static __host__ __device__ float ToFloat(StrideType value)
    {
        return static_cast<float>(value);
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddIntegerArea(DimType c, const AreaBounds &bounds, AccumType &out) const
    {
        const float scale = 1.f / (m_scaleX * m_scaleY);

        for (StrideType cy = bounds.ymin; cy < bounds.ymax; ++cy)
        {
            for (StrideType cx = bounds.xmin; cx < bounds.xmax; ++cx)
            {
                out += Base::doGetValue(c, cx, cy) * scale;
            }
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddFractionalArea(DimType c, const AreaBounds &bounds, AccumType &out) const
    {
        const float scale = FractionalAreaScale(bounds);

        AddInteriorRows(c, bounds, scale, out);

        if (ToFloat(bounds.ymin) > bounds.fsy1)
        {
            AddTopEdgeRow(c, bounds, scale, out);
        }

        if (ToFloat(bounds.ymax) < bounds.fsy2)
        {
            AddBottomEdgeRow(c, bounds, scale, out);
        }
    }

    inline __host__ __device__ float FractionalAreaScale(const AreaBounds &bounds) const
    {
        // There are 2 active dimensions (0, 1) and the coordinates are inverted (y, x)
        // so y corresponds to dimension 0 and x corresponds to dimension 1.
        const float w = ToFloat(Base::m_borderWrap.tensorShape()[1]);
        const float h = ToFloat(Base::m_borderWrap.tensorShape()[0]);

        return 1.f / (min(m_scaleX, w - bounds.fsx1) * min(m_scaleY, h - bounds.fsy1));
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddInteriorRows(DimType c, const AreaBounds &bounds, float scale,
                                                    AccumType &out) const
    {
        for (StrideType cy = bounds.ymin; cy < bounds.ymax; ++cy)
        {
            for (StrideType cx = bounds.xmin; cx < bounds.xmax; ++cx)
            {
                out += Base::doGetValue(c, cx, cy) * scale;
            }

            AddHorizontalEdgeCells(c, bounds, cy, scale, out);
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddHorizontalEdgeCells(DimType c, const AreaBounds &bounds, StrideType cy,
                                                           float scale, AccumType &out) const
    {
        if (ToFloat(bounds.xmin) > bounds.fsx1)
        {
            out += Base::doGetValue(c, (bounds.xmin - 1), cy) * ((ToFloat(bounds.xmin) - bounds.fsx1) * scale);
        }

        if (ToFloat(bounds.xmax) < bounds.fsx2)
        {
            out += Base::doGetValue(c, bounds.xmax, cy) * ((bounds.fsx2 - ToFloat(bounds.xmax)) * scale);
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddTopEdgeRow(DimType c, const AreaBounds &bounds, float scale,
                                                  AccumType &out) const
    {
        for (StrideType cx = bounds.xmin; cx < bounds.xmax; ++cx)
        {
            out += Base::doGetValue(c, cx, (bounds.ymin - 1)) * ((ToFloat(bounds.ymin) - bounds.fsy1) * scale);
        }

        if (ToFloat(bounds.xmin) > bounds.fsx1)
        {
            out += Base::doGetValue(c, (bounds.xmin - 1), (bounds.ymin - 1))
                 * ((ToFloat(bounds.ymin) - bounds.fsy1) * (ToFloat(bounds.xmin) - bounds.fsx1) * scale);
        }

        if (ToFloat(bounds.xmax) < bounds.fsx2)
        {
            out += Base::doGetValue(c, bounds.xmax, (bounds.ymin - 1))
                 * ((ToFloat(bounds.ymin) - bounds.fsy1) * (bounds.fsx2 - ToFloat(bounds.xmax)) * scale);
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddBottomEdgeRow(DimType c, const AreaBounds &bounds, float scale,
                                                     AccumType &out) const
    {
        for (StrideType cx = bounds.xmin; cx < bounds.xmax; ++cx)
        {
            out += Base::doGetValue(c, cx, bounds.ymax) * ((bounds.fsy2 - ToFloat(bounds.ymax)) * scale);
        }

        if (ToFloat(bounds.xmax) < bounds.fsx2)
        {
            out += Base::doGetValue(c, bounds.xmax, bounds.ymax)
                 * ((bounds.fsy2 - ToFloat(bounds.ymax)) * (bounds.fsx2 - ToFloat(bounds.xmax)) * scale);
        }

        if (ToFloat(bounds.xmin) > bounds.fsx1)
        {
            out += Base::doGetValue(c, (bounds.xmin - 1), bounds.ymax)
                 * ((bounds.fsy2 - ToFloat(bounds.ymax)) * (ToFloat(bounds.xmin) - bounds.fsx1) * scale);
        }
    }

    inline __host__ __device__ bool isIntegerArea(float scaleX, float scaleY) const
    {
        return static_cast<float>(cuda::round<RoundMode::UP, int>(scaleX)) == scaleX
            && static_cast<float>(cuda::round<RoundMode::UP, int>(scaleY)) == scaleY;
    }

    const float m_scaleX        = {};
    const float m_scaleY        = {};
    const bool  m_isIntegerArea = {};
};

/**@}*/

/**
 * Factory function to create an NHW interpolation wrap given a tensor data.
 *
 * The output \ref InterpolationWrap wraps an NHW 3D tensor allowing to access data per batch (N), per row (H) and
 * per column (W) of the input tensor interpolation-border aware in rows (or height H) and columns (or width W).
 * The input tensor data must have either NHWC or HWC layout, where the channel C is inside the given template type
 * \p T, e.g. T=uchar4 for RGBA8.  The active dimensions are H (second) and W (third).
 *
 * @sa NVCV_CPP_CUDATOOLS_INTERPOLATIONWRAP
 *
 * @tparam T Type of the values to be accessed in the interpolation wrap.
 * @tparam B Border extension to be used when accessing H and W, one of \ref NVCVBorderType
 * @tparam I Interpolation to be used when accessing H and W, one of \ref NVCVInterpolationType
 * @tparam StrideType Stride type used when accessing underlying tensor data
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 * @param[in] borderValue Border value to be used when accessing outside elements in constant border type
 * @param[in] scaleX Scale X value to be used when interpolation elements with area type
 * @param[in] scaleY Scale Y value to be used when interpolation elements with area type
 *
 * @return Interpolation wrap useful to access tensor data interpolation-border aware in H and W in CUDA kernels.
 */
template<typename T, NVCVBorderType B, NVCVInterpolationType I, typename StrideType = int64_t,
         class = Require<HasTypeTraits<T>>>
__host__ auto CreateInterpolationWrapNHW(const TensorDataStridedCuda &tensor, T borderValue = {}, float scaleX = {},
                                         float scaleY = {})
{
    auto borderWrap = CreateBorderWrapNHW<T, B, StrideType>(tensor, borderValue);

    return InterpolationWrap<decltype(borderWrap), I>(borderWrap, scaleX, scaleY);
}

/**
 * Factory function to create an NHWC interpolation wrap given a tensor data.
 *
 * The output \ref InterpolationWrap wraps an NHWC 4D tensor allowing to access data per batch (N), per row (H),
 * per column (W) and per channel (C) of the input tensor interpolation-border aware in rows (or height H) and
 * columns (or width W).  The input tensor data must have either NHWC or HWC layout, where the channel C is of type
 * \p T, e.g. T=uchar for each channel of either RGB8 or RGBA8.  The active dimensions are H (second) and W
 * (third).
 *
 * @sa NVCV_CPP_CUDATOOLS_INTERPOLATIONWRAP
 *
 * @tparam T Type of the values to be accessed in the interpolation wrap.
 * @tparam B Border extension to be used when accessing H and W, one of \ref NVCVBorderType
 * @tparam I Interpolation to be used when accessing H and W, one of \ref NVCVInterpolationType
 * @tparam StrideType Stride type used when accessing underlying tensor data
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 * @param[in] borderValue Border value to be used when accessing outside elements in constant border type
 * @param[in] scaleX Scale X value to be used when interpolation elements with area type
 * @param[in] scaleY Scale Y value to be used when interpolation elements with area type
 *
 * @return Interpolation wrap useful to access tensor data interpolation-border aware in H and W in CUDA kernels.
 */
template<typename T, NVCVBorderType B, NVCVInterpolationType I, typename StrideType = int64_t,
         class = Require<HasTypeTraits<T>>>
__host__ auto CreateInterpolationWrapNHWC(const TensorDataStridedCuda &tensor, T borderValue = {}, float scaleX = {},
                                          float scaleY = {})
{
    auto borderWrap = CreateBorderWrapNHWC<T, B, StrideType>(tensor, borderValue);

    return InterpolationWrap<decltype(borderWrap), I>(borderWrap, scaleX, scaleY);
}

} // namespace nvcv::cuda

#endif // NVCV_CUDA_INTERPOLATION_WRAP_HPP
