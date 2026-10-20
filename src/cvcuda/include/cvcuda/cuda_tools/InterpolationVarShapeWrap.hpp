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

#ifndef NVCV_CUDA_INTERPOLATION_VAR_SHAPE_WRAP_HPP
#define NVCV_CUDA_INTERPOLATION_VAR_SHAPE_WRAP_HPP

#include "BorderVarShapeWrap.hpp" // for BorderVarShapeWrap, etc.
#include "InterpolationWrap.hpp"  // for GetIndexForInterpolation, etc.

#include <nvcv/ImageBatchData.hpp> // for ImageBatchVarShapeDataStridedCuda, etc.

namespace nvcv::cuda {

namespace detail {

template<typename T, NVCVBorderType B, NVCVInterpolationType I, bool ReadTapByValue>
class InterpolationVarShapeWrapImpl
{
public:
    using BorderWrapper     = BorderVarShapeWrap<T, B>;
    using ImageBatchWrapper = typename BorderWrapper::ImageBatchWrapper;
    using ValueType         = typename BorderWrapper::ValueType;

    static constexpr int                   kNumDimensions     = BorderWrapper::kNumDimensions;
    static constexpr NVCVInterpolationType kInterpolationType = I;

    static constexpr auto kActiveDimensions    = BorderWrapper::kActiveDimensions;
    static constexpr int  kNumActiveDimensions = BorderWrapper::kNumActiveDimensions;

    InterpolationVarShapeWrapImpl() = default;

    explicit __host__ InterpolationVarShapeWrapImpl(const ImageBatchVarShapeDataStridedCuda &images,
                                                    ValueType                                borderValue = {})
        : m_borderWrap(images, borderValue)
    {
    }

    template<typename... Args>
    explicit __host__ __device__ InterpolationVarShapeWrapImpl(ImageBatchWrapper imageBatchWrap, ValueType borderValue,
                                                               Args... tensorShape)
        : m_borderWrap(imageBatchWrap, borderValue, tensorShape...)
    {
    }

    explicit __host__ __device__ InterpolationVarShapeWrapImpl(BorderWrapper borderWrap)
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
    /*
     * Which constant-border form is cheaper here is a property of the pixel width, not of the
     * border mode; both return the same element (see
     * nvcv::cuda::detail::ReadConstantBorder). `value()` re-materializes the border value per
     * byte-component, so its cost scales with the pixel width, while what it saves -- the
     * wrapper's stack frame -- is a fixed size. Measured on H100 CI the crossover sits between
     * three and four components. The tensor interpolation ladder has its own measured policy;
     * direct BorderVarShapeWrap consumers opt in only for measured scalar paths. Measured:
     * with `value()` Remap var-shape uchar4 DENSE regressed
     * 1198.6 -> 1343.5 us (+12.1%), while restricting it to one component put WarpAffine
     * var-shape uchar3 LINEAR CONSTANT back on the indexing form at +13.3% (H100) and +10.0%
     * (A100).
     */
    static constexpr bool kReadTapByValue = ReadTapByValue && B == NVCV_BORDER_CONSTANT && NumElements<ValueType> <= 3;

    // The one place this interpolation policy is applied. Only the NVCV_BORDER_CONSTANT
    // specialization has value(); every other case retains operator[].
    template<typename IntCoord>
    inline __host__ __device__ decltype(auto) readTap(IntCoord c) const
    {
        if constexpr (kReadTapByValue)
        {
            return m_borderWrap.value(c);
        }
        else
        {
            return m_borderWrap[c];
        }
    }

    template<typename DimType,
             class = Require<
                 std::is_same_v<BaseType<DimType>, float> && (NumElements<DimType> == 3 || NumElements<DimType> == 4)>>
    inline __host__ __device__ decltype(auto) doGetValue(DimType c) const
    {
        return readTap(StaticCast<int>(c));
    }

    inline __host__ __device__ decltype(auto) doGetValue(float3 c, int x, int y) const
    {
        return readTap(int3{x, y, static_cast<int>(c.z)});
    }

    inline __host__ __device__ decltype(auto) doGetValue(float4 c, int x, int y) const
    {
        return readTap(int4{x, y, static_cast<int>(c.z), static_cast<int>(c.w)});
    }

    const BorderWrapper m_borderWrap = {};
};

} // namespace detail

/**
 * @defgroup NVCV_CPP_CUDATOOLS_INTERPOLATIONVARSHAPEWRAP InterpolationVarShapeWrap classes
 * @{
 */

/**
 * Interpolation var-shape wrapper class used to wrap a \ref BorderVarShapeWrap adding interpolation handling to it.
 *
 * This class wraps a \ref BorderVarShapeWrap to add interpolation handling functionality.  It provides the \ref operator[]
 * to do the same semantic value access in the wrapped BorderVarShapeWrap but interpolation aware.
 *
 * @note Each interpolation wrap class below is specialized for one interpolation type.
 *
 * @sa NVCV_CPP_CUDATOOLS_INTERPOLATIONVARSHAPEWRAPS
 *
 * @tparam T Type (it can be const) of each element inside the border var-shape wrapper.
 * @tparam B Border extension used by the wrapped images.
 * @tparam I It is a \ref NVCVInterpolationType defining the interpolation type to be used.
 * @tparam ReadTapByValue Whether to opt measured constant-border taps with at most three
 *                       components into \ref BorderVarShapeWrap::value. Wider elements retain
 *                       reference reads; the tensor \ref InterpolationWrap policy has no width clamp.
 */
template<typename T, NVCVBorderType B, NVCVInterpolationType I, bool ReadTapByValue = false>
class InterpolationVarShapeWrap : public detail::InterpolationVarShapeWrapImpl<T, B, I, ReadTapByValue>
{
};

/**
 * Interpolation var-shape wrapper class specialized for \ref NVCV_INTERP_NEAREST.
 *
 * @tparam T Type (it can be const) of each element inside the border var-shape wrapper.
 */
template<typename T, NVCVBorderType B, bool ReadTapByValue>
class InterpolationVarShapeWrap<T, B, NVCV_INTERP_NEAREST, ReadTapByValue>
    : public detail::InterpolationVarShapeWrapImpl<T, B, NVCV_INTERP_NEAREST, ReadTapByValue>
{
    using Base = detail::InterpolationVarShapeWrapImpl<T, B, NVCV_INTERP_NEAREST, ReadTapByValue>;

public:
    using typename Base::BorderWrapper;
    using typename Base::ImageBatchWrapper;
    using typename Base::ValueType;

    using Base::kInterpolationType;
    using Base::kNumDimensions;

    InterpolationVarShapeWrap() = default;

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping \p images.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ InterpolationVarShapeWrap(const ImageBatchVarShapeDataStridedCuda &images,
                                                ValueType borderValue = {}, float scaleX = {}, float scaleY = {})
        : Base(images, borderValue)
    {
    }

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping an \p imageBatchWrap.
     *
     * @param[in] imageBatchWrap A \ref ImageBatchVarShapeWrap object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    template<typename... Args>
    explicit __host__ __device__ InterpolationVarShapeWrap(ImageBatchWrapper imageBatchWrap, ValueType borderValue,
                                                           float scaleX, float scaleY, Args... tensorShape)
        : Base(imageBatchWrap, borderValue, tensorShape...)
    {
    }

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping a \p borderWrap.
     *
     * @param[in] borderWrap A \ref BorderVarShapeWrap object to be wrapped.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ __device__ InterpolationVarShapeWrap(BorderWrapper borderWrap, float scaleX = {},
                                                           float scaleY = {})
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
     * @param[in] c Either 3D (z sample, y row and x column) or
     *              4D coordinate (w sample, z plane, y row and x column) to be accessed with (x, y) interpolation.
     *
     * @return Accessed interpolated value.
     */
    template<typename DimType,
             class = Require<
                 std::is_same_v<BaseType<DimType>, float> && (NumElements<DimType> == 3 || NumElements<DimType> == 4)>>
    inline __host__ __device__ ValueType operator[](DimType c) const
    {
        c.x = GetIndexForInterpolation<kInterpolationType>(c.x + .5f);
        c.y = GetIndexForInterpolation<kInterpolationType>(c.y + .5f);

        return doGetValue(c);
    }
};

/**
 * Interpolation var-shape wrapper class specialized for \ref NVCV_INTERP_LINEAR.
 *
 * @tparam T Type (it can be const) of each element inside the border var-shape wrapper.
 */
template<typename T, NVCVBorderType B, bool ReadTapByValue>
class InterpolationVarShapeWrap<T, B, NVCV_INTERP_LINEAR, ReadTapByValue>
    : public detail::InterpolationVarShapeWrapImpl<T, B, NVCV_INTERP_LINEAR, ReadTapByValue>
{
    using Base = detail::InterpolationVarShapeWrapImpl<T, B, NVCV_INTERP_LINEAR, ReadTapByValue>;

public:
    using typename Base::BorderWrapper;
    using typename Base::ImageBatchWrapper;
    using typename Base::ValueType;

    using Base::kInterpolationType;
    using Base::kNumDimensions;

    InterpolationVarShapeWrap() = default;

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping \p images.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ InterpolationVarShapeWrap(const ImageBatchVarShapeDataStridedCuda &images,
                                                ValueType borderValue = {}, float scaleX = {}, float scaleY = {})
        : Base(images, borderValue)
    {
    }

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping an \p imageBatchWrap.
     *
     * @param[in] imageBatchWrap A \ref ImageBatchVarShapeWrap object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    template<typename... Args>
    explicit __host__ __device__ InterpolationVarShapeWrap(ImageBatchWrapper imageBatchWrap, ValueType borderValue,
                                                           float scaleX, float scaleY, Args... tensorShape)
        : Base(imageBatchWrap, borderValue, tensorShape...)
    {
    }

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping a \p borderWrap.
     *
     * @param[in] borderWrap A \ref BorderVarShapeWrap object to be wrapped.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ __device__ InterpolationVarShapeWrap(BorderWrapper borderWrap, float scaleX = {},
                                                           float scaleY = {})
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
     * @param[in] c Either 3D (z sample, y row and x column) or
     *              4D coordinate (w sample, z plane, y row and x column) to be accessed with (x, y) interpolation.
     *
     * @return Accessed interpolated value.
     */
    template<typename DimType,
             class = Require<
                 std::is_same_v<BaseType<DimType>, float> && (NumElements<DimType> == 3 || NumElements<DimType> == 4)>>
    inline __host__ __device__ ValueType operator[](DimType c) const
    {
        // IndexType must match the storage width (`int`): letting
        // GetIndexForInterpolation return the default int64 would bypass its
        // overflow-prevention clamp and re-expose x1 + 1 to int32 overflow.
        const int x1 = GetIndexForInterpolation<kInterpolationType, 1, int>(c.x);
        const int x2 = x1 + 1;
        const int y1 = GetIndexForInterpolation<kInterpolationType, 1, int>(c.y);
        const int y2 = y1 + 1;

        auto out = SetAll<ConvertBaseTypeTo<float, std::remove_cv_t<ValueType>>>(0);

        out += Base::doGetValue(c, x1, y1) * (x2 - c.x) * (y2 - c.y);
        out += Base::doGetValue(c, x2, y1) * (c.x - x1) * (y2 - c.y);
        out += Base::doGetValue(c, x1, y2) * (x2 - c.x) * (c.y - y1);
        out += Base::doGetValue(c, x2, y2) * (c.x - x1) * (c.y - y1);

        return SaturateCast<ValueType>(out);
    }
};

/**
 * Interpolation var-shape wrapper class specialized for \ref NVCV_INTERP_CUBIC.
 *
 * @tparam T Type (it can be const) of each element inside the border var-shape wrapper.
 */
template<typename T, NVCVBorderType B, bool ReadTapByValue>
class InterpolationVarShapeWrap<T, B, NVCV_INTERP_CUBIC, ReadTapByValue>
    : public detail::InterpolationVarShapeWrapImpl<T, B, NVCV_INTERP_CUBIC, ReadTapByValue>
{
    using Base = detail::InterpolationVarShapeWrapImpl<T, B, NVCV_INTERP_CUBIC, ReadTapByValue>;

public:
    using typename Base::BorderWrapper;
    using typename Base::ImageBatchWrapper;
    using typename Base::ValueType;

    using Base::kInterpolationType;
    using Base::kNumDimensions;

    InterpolationVarShapeWrap() = default;

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping \p images.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ InterpolationVarShapeWrap(const ImageBatchVarShapeDataStridedCuda &images,
                                                ValueType borderValue = {}, float scaleX = {}, float scaleY = {})
        : Base(images, borderValue)
    {
    }

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping an \p imageBatchWrap.
     *
     * @param[in] imageBatchWrap A \ref ImageBatchVarShapeWrap object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    template<typename... Args>
    explicit __host__ __device__ InterpolationVarShapeWrap(ImageBatchWrapper imageBatchWrap, ValueType borderValue,
                                                           float scaleX, float scaleY, Args... tensorShape)
        : Base(imageBatchWrap, borderValue, tensorShape...)
    {
    }

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping a \p borderWrap.
     *
     * @param[in] borderWrap A \ref BorderVarShapeWrap object to be wrapped.
     * @param[in] scaleX The scale X value is ignored in non-Area interpolation types.
     * @param[in] scaleY The scale Y value is ignored in non-Area interpolation types.
     */
    explicit __host__ __device__ InterpolationVarShapeWrap(BorderWrapper borderWrap, float scaleX = {},
                                                           float scaleY = {})
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
     * @param[in] c Either 3D (z sample, y row and x column) or
     *              4D coordinate (w sample, z plane, y row and x column) to be accessed with (x, y) interpolation.
     *
     * @return Accessed interpolated value.
     */
    template<typename DimType,
             class = Require<
                 std::is_same_v<BaseType<DimType>, float> && (NumElements<DimType> == 3 || NumElements<DimType> == 4)>>
    inline __host__ __device__ ValueType operator[](DimType c) const
    {
        // Explicit int IndexType — see note at LINEAR above.
        const int ix = GetIndexForInterpolation<kInterpolationType, 1, int>(c.x);
        const int iy = GetIndexForInterpolation<kInterpolationType, 1, int>(c.y);

        using FT = ConvertBaseTypeTo<float, std::remove_cv_t<ValueType>>;
        auto sum = SetAll<FT>(0);

        float wx[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
        GetCubicCoeffs(c.x - ix, wx[0], wx[1], wx[2], wx[3]);
        float wy[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
        GetCubicCoeffs(c.y - iy, wy[0], wy[1], wy[2], wy[3]);

#pragma unroll
        for (int cy = -1; cy <= 2; cy++)
        {
#pragma unroll
            for (int cx = -1; cx <= 2; cx++)
            {
                sum += Base::doGetValue(c, ix + cx, iy + cy) * (wx[cx + 1] * wy[cy + 1]);
            }
        }

        return SaturateCast<ValueType>(sum);
    }
};

/**
 * Interpolation var-shape wrapper class specialized for \ref NVCV_INTERP_AREA.
 *
 * @tparam T Type (it can be const) of each element inside the border var-shape wrapper.
 */
template<typename T, NVCVBorderType B, bool ReadTapByValue>
class InterpolationVarShapeWrap<T, B, NVCV_INTERP_AREA, ReadTapByValue>
    : public detail::InterpolationVarShapeWrapImpl<T, B, NVCV_INTERP_AREA, ReadTapByValue>
{
    using Base = detail::InterpolationVarShapeWrapImpl<T, B, NVCV_INTERP_AREA, ReadTapByValue>;

public:
    using typename Base::BorderWrapper;
    using typename Base::ImageBatchWrapper;
    using typename Base::ValueType;

    using Base::kInterpolationType;
    using Base::kNumDimensions;

    InterpolationVarShapeWrap() = default;

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping \p images.
     *
     * @param[in] tensor A \ref TensorDataStridedCuda object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value for Area interpolation.
     * @param[in] scaleY The scale Y value for Area interpolation.
     */
    explicit __host__ InterpolationVarShapeWrap(const ImageBatchVarShapeDataStridedCuda &images,
                                                ValueType borderValue = {}, float scaleX = {}, float scaleY = {})
        : Base(images, borderValue)
        , m_scaleX(scaleX)
        , m_scaleY(scaleY)
        , m_isIntegerArea(isIntegerArea(scaleX, scaleY))
    {
    }

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping an \p imageBatchWrap.
     *
     * @param[in] imageBatchWrap A \ref ImageBatchVarShapeWrap object to be wrapped.
     * @param[in] borderValue The border value.
     * @param[in] scaleX The scale X value for Area interpolation.
     * @param[in] scaleY The scale Y value for Area interpolation.
     */
    template<typename... Args>
    explicit __host__ __device__ InterpolationVarShapeWrap(ImageBatchWrapper imageBatchWrap, ValueType borderValue,
                                                           float scaleX, float scaleY, Args... tensorShape)
        : Base(imageBatchWrap, borderValue, tensorShape...)
        , m_scaleX(scaleX)
        , m_scaleY(scaleY)
        , m_isIntegerArea(isIntegerArea(scaleX, scaleY))
    {
    }

    /**
     * Constructs an InterpolationVarShapeWrap by wrapping a \p borderWrap.
     *
     * @param[in] borderWrap A \ref BorderVarShapeWrap object to be wrapped.
     * @param[in] scaleX The scale X value for Area interpolation.
     * @param[in] scaleY The scale Y value for Area interpolation.
     */
    explicit __host__ __device__ InterpolationVarShapeWrap(BorderWrapper borderWrap, float scaleX = {},
                                                           float scaleY = {})
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
     * @param[in] c Either 3D (z sample, y row and x column) or
     *              4D coordinate (w sample, z plane, y row and x column) to be accessed with (x, y) interpolation.
     *
     * @return Accessed interpolated value.
     */
    template<typename DimType,
             class = Require<
                 std::is_same_v<BaseType<DimType>, float> && (NumElements<DimType> == 3 || NumElements<DimType> == 4)>>
    inline __host__ __device__ ValueType operator[](DimType c) const
    {
        const AreaBounds bounds = MakeAreaBounds(c);

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
    struct AreaBounds
    {
        float fsx1;
        float fsy1;
        float fsx2;
        float fsy2;
        int   xmin;
        int   xmax;
        int   ymin;
        int   ymax;
    };

    template<typename DimType>
    inline __host__ __device__ AreaBounds MakeAreaBounds(DimType c) const
    {
        const float fsx1 = c.x * m_scaleX;
        const float fsy1 = c.y * m_scaleY;
        const float fsx2 = fsx1 + m_scaleX;
        const float fsy2 = fsy1 + m_scaleY;

        // Explicit int IndexType — see note at LINEAR above.
        return {fsx1,
                fsy1,
                fsx2,
                fsy2,
                GetIndexForInterpolation<kInterpolationType, 1, int>(fsx1),
                GetIndexForInterpolation<kInterpolationType, 2, int>(fsx2),
                GetIndexForInterpolation<kInterpolationType, 1, int>(fsy1),
                GetIndexForInterpolation<kInterpolationType, 2, int>(fsy2)};
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddIntegerArea(DimType c, const AreaBounds &bounds, AccumType &out) const
    {
        const float scale = 1.f / (m_scaleX * m_scaleY);

        for (int cy = bounds.ymin; cy < bounds.ymax; ++cy)
        {
            for (int cx = bounds.xmin; cx < bounds.xmax; ++cx)
            {
                out += Base::doGetValue(c, cx, cy) * scale;
            }
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddFractionalArea(DimType c, const AreaBounds &bounds, AccumType &out) const
    {
        const float scale = FractionalAreaScale(c, bounds);

        AddInteriorRows(c, bounds, scale, out);

        if (bounds.ymin > bounds.fsy1)
        {
            AddTopEdgeRow(c, bounds, scale, out);
        }

        if (bounds.ymax < bounds.fsy2)
        {
            AddBottomEdgeRow(c, bounds, scale, out);
        }
    }

    template<typename DimType>
    inline __host__ __device__ float FractionalAreaScale(DimType c, const AreaBounds &bounds) const
    {
        const int w = ImageWidth(c);
        const int h = ImageHeight(c);

        return 1.f / (min(m_scaleX, w - bounds.fsx1) * min(m_scaleY, h - bounds.fsy1));
    }

    template<typename DimType>
    inline __host__ __device__ int ImageWidth(DimType c) const
    {
        if constexpr (NumElements<DimType> == 3)
        {
            return Base::m_borderWrap.imageBatchWrap().width(static_cast<int>(c.z));
        }
        else
        {
            return Base::m_borderWrap.imageBatchWrap().width(static_cast<int>(c.w), static_cast<int>(c.z));
        }
    }

    template<typename DimType>
    inline __host__ __device__ int ImageHeight(DimType c) const
    {
        if constexpr (NumElements<DimType> == 3)
        {
            return Base::m_borderWrap.imageBatchWrap().height(static_cast<int>(c.z));
        }
        else
        {
            return Base::m_borderWrap.imageBatchWrap().height(static_cast<int>(c.w), static_cast<int>(c.z));
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddInteriorRows(DimType c, const AreaBounds &bounds, float scale,
                                                    AccumType &out) const
    {
        for (int cy = bounds.ymin; cy < bounds.ymax; ++cy)
        {
            for (int cx = bounds.xmin; cx < bounds.xmax; ++cx)
            {
                out += Base::doGetValue(c, cx, cy) * scale;
            }

            AddHorizontalEdgeCells(c, bounds, cy, scale, out);
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddHorizontalEdgeCells(DimType c, const AreaBounds &bounds, int cy, float scale,
                                                           AccumType &out) const
    {
        if (bounds.xmin > bounds.fsx1)
        {
            out += Base::doGetValue(c, (bounds.xmin - 1), cy) * ((bounds.xmin - bounds.fsx1) * scale);
        }

        if (bounds.xmax < bounds.fsx2)
        {
            out += Base::doGetValue(c, bounds.xmax, cy) * ((bounds.fsx2 - bounds.xmax) * scale);
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddTopEdgeRow(DimType c, const AreaBounds &bounds, float scale,
                                                  AccumType &out) const
    {
        for (int cx = bounds.xmin; cx < bounds.xmax; ++cx)
        {
            out += Base::doGetValue(c, cx, (bounds.ymin - 1)) * ((bounds.ymin - bounds.fsy1) * scale);
        }

        if (bounds.xmin > bounds.fsx1)
        {
            out += Base::doGetValue(c, (bounds.xmin - 1), (bounds.ymin - 1))
                 * ((bounds.ymin - bounds.fsy1) * (bounds.xmin - bounds.fsx1) * scale);
        }

        if (bounds.xmax < bounds.fsx2)
        {
            out += Base::doGetValue(c, bounds.xmax, (bounds.ymin - 1))
                 * ((bounds.ymin - bounds.fsy1) * (bounds.fsx2 - bounds.xmax) * scale);
        }
    }

    template<typename DimType, typename AccumType>
    inline __host__ __device__ void AddBottomEdgeRow(DimType c, const AreaBounds &bounds, float scale,
                                                     AccumType &out) const
    {
        for (int cx = bounds.xmin; cx < bounds.xmax; ++cx)
        {
            out += Base::doGetValue(c, cx, bounds.ymax) * ((bounds.fsy2 - bounds.ymax) * scale);
        }

        if (bounds.xmax < bounds.fsx2)
        {
            out += Base::doGetValue(c, bounds.xmax, bounds.ymax)
                 * ((bounds.fsy2 - bounds.ymax) * (bounds.fsx2 - bounds.xmax) * scale);
        }

        if (bounds.xmin > bounds.fsx1)
        {
            out += Base::doGetValue(c, (bounds.xmin - 1), bounds.ymax)
                 * ((bounds.fsy2 - bounds.ymax) * (bounds.xmin - bounds.fsx1) * scale);
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

} // namespace nvcv::cuda

#endif // NVCV_CUDA_INTERPOLATION_VAR_SHAPE_WRAP_HPP
