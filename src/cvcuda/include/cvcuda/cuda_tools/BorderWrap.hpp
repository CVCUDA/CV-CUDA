/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <type_traits>

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
constexpr bool __host__ __device__ IsOutside(T c, T s)
{
    if constexpr (Active)
    {
        return (c < 0) || (c >= s);
    }
    return false;
}

namespace detail {

template<typename T>
constexpr T __host__ __device__ ReplicateBorderIndex(T c, T s)
{
    if (c < 0)
    {
        return 0;
    }

    if (c >= s)
    {
        return s - 1;
    }

    return c;
}

template<typename T>
constexpr T __host__ __device__ WrapBorderIndex(T c, T s)
{
    c = c % s;
    return (c < 0) ? c + s : c;
}

template<typename T>
constexpr T __host__ __device__ ReflectBorderIndex(T c, T s)
{
    T s2 = s * 2;
    c    = WrapBorderIndex(c, s2);
    return s - 1 - (abs(2 * c + 1 - s2) >> 1);
}

template<typename T>
constexpr T __host__ __device__ Reflect101BorderIndex(T c, T s)
{
    if (s == 1)
    {
        return 0;
    }

    c = WrapBorderIndex(c, 2 * s - 2);
    return s - 1 - abs(s - 1 - c);
}

} // namespace detail

/**
 * Function to get a border-aware index considering the range defined by given size.
 *
 * @note This function does not work for constant border modes.
 *
 * @tparam B It is a \ref NVCVBorderType indicating the border to be used.
 * @tparam Active Flag to turn this function active.
 * @tparam T Type of the values given to this function.
 *
 * @param[in] c Coordinate (input index) to put back inside valid range [0, s).
 * @param[in] s Size that defines the valid range [0, s).
 */
template<NVCVBorderType B, bool Active = true, typename T>
constexpr T __host__ __device__ GetIndexWithBorder(T c, T s)
{
    static_assert(B != NVCV_BORDER_CONSTANT && B != NVCV_BORDER_CONSTANT_COVERAGE,
                  "GetIndexWithBorder cannot be used with constant border modes");

    if constexpr (Active)
    {
        assert(s > 0);

        if constexpr (B == NVCV_BORDER_REPLICATE)
        {
            c = detail::ReplicateBorderIndex(c, s);
        }
        else if constexpr (B == NVCV_BORDER_WRAP)
        {
            c = detail::WrapBorderIndex(c, s);
        }
        else if constexpr (B == NVCV_BORDER_REFLECT)
        {
            c = detail::ReflectBorderIndex(c, s);
        }
        else if constexpr (B == NVCV_BORDER_REFLECT101)
        {
            c = detail::Reflect101BorderIndex(c, s);
        }

        assert(c >= 0 && c < s);
    }

    return c;
}

/**@}*/

namespace detail {

/*
 * Shared rationale for the three helpers below.
 *
 * Each helper reads a constant-border element, given the pointer the wrap resolved for a
 * coordinate and the wrap's border value. \p p is null exactly when the coordinate is outside the
 * tensor.
 *
 * This is the one place the constant-border read decides its shape, and the reason it has a shape
 * at all is codegen. A wrap's `operator[]` must return a reference, and out of bounds the only
 * address it can hand back is that of the wrap's own border-value member -- which lives inside an
 * object kernels take by value as a parameter. Up to sm_90, taking that address makes the compiler
 * stage the whole wrapper on the thread's stack and demote every in-bounds tap from a global load
 * (`LDG`) to a generic one (`LD`), because the loaded pointer may now alias local memory. On the
 * B32 1920x1080 uchar3 LINEAR WarpAffine that was 72 bytes of stack per thread and turned a 199 MB
 * output into 4.71 GB of DRAM writes. Reading the element out by value never takes that address.
 *
 * On sm_100, sm_120 and sm_121 the compiler keeps the wrapper in the constant bank on its own --
 * each compiles the unfixed kernel to `STACK:0` with twelve `LDG` taps and no local stores -- so
 * there is no spill left to avoid there, and for a 3-byte element the reference form is the
 * cheaper of the two because it loads the element straight from the selected address instead of
 * assembling it from per-byte selects. Measured on RTX PRO 6000 (sm_120), forcing the by-value
 * form on that architecture costs 13% to 15% on the same WarpAffine case. Both forms return the
 * same value; each architecture uses the one that is faster on it.
 *
 * All three are __forceinline__ rather than plain inline: an outlined call would take the address
 * of the wrap's own border-value member at this boundary and reintroduce exactly the staging
 * described above, so inlining is a requirement here rather than an inliner heuristic. That pin is
 * also what lets the by-value form take its border value by `const` reference: inlining collapses
 * the reference, so no address survives, whereas passing the element by value materializes a copy
 * at every tap. On sm_80 that copy costs eight registers in the uchar3 LINEAR CONSTANT var-shape
 * WarpPerspective kernel, which at the 32x8 block is one occupancy step.
 */
/** @internal @brief Select by value, never taking the address of \p borderValue. */
template<typename T>
__host__ __device__ __forceinline__ T ReadConstantBorderViaValueSelect(const T *p, const T &borderValue)
{
    if (p == nullptr)
    {
        return borderValue;
    }
    return *p;
}

/** @internal @brief Select an address, then load the value through it. */
template<typename T>
__host__ __device__ __forceinline__ T ReadConstantBorderViaAddressSelect(const T *p, const T &borderValue)
{
    const T &v = (p == nullptr) ? borderValue : *p;
    return v;
}

/**
 * @brief Read a constant-border element, in whichever of the two forms above is faster here.
 *
 * Both forms must return the same value; `TestBorderWrap` pins that.
 */
template<typename T>
__host__ __device__ __forceinline__ T ReadConstantBorder(const T *p, const T &borderValue)
{
// An allowlist, not a `>= 1000` floor: sm_100 and sm_121 match the measured sm_120 compile
// shape, while Thor (sm_110) lacks even that static evidence. A floor would also opt in every
// future architecture on today's evidence.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
    return ReadConstantBorderViaAddressSelect(p, borderValue);
#else
    return ReadConstantBorderViaValueSelect(p, borderValue);
#endif
}

template<class TW, NVCVBorderType B, bool... ActiveDimensions>
class BorderWrapImpl
{
public:
    using TensorWrapper = TW;
    using ValueType     = typename TensorWrapper::ValueType;
    using StrideType    = typename TensorWrapper::StrideType;

    static constexpr int            kNumDimensions = TensorWrapper::kNumDimensions;
    static constexpr NVCVBorderType kBorderType    = B;

    static_assert(kNumDimensions == sizeof...(ActiveDimensions));

    static constexpr bool kActiveDimensions[]  = {ActiveDimensions...}; // NOSONAR: device metadata.
    static constexpr int  kNumActiveDimensions = ((ActiveDimensions ? 1 : 0) + ...);

    struct ActiveMap
    {
        int from[kNumDimensions] = {}; // NOSONAR: constexpr device lookup table.

        constexpr ActiveMap()
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
        , m_tensorShape{static_cast<StrideType>(tensorShape)...}
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
            static_assert((IsIndexType<Args, StrideType> && ...));
            static_assert(sizeof...(Args) == kNumActiveDimensions);
        }
    }

    explicit __host__ BorderWrapImpl(const TensorDataStridedCuda &tensor)
        : m_tensorWrap(tensor)
    {
        assert(tensor.rank() >= kNumDimensions);

        int j = 0;
#pragma unroll
        for (int i = 0; i < kNumDimensions; ++i)
        {
            if (kActiveDimensions[i])
            {
                assert(tensor.shape(i) <= TypeTraits<int>::max);

                m_tensorShape[j++] = tensor.shape(i);
            }
        }
    }

    inline const __host__ __device__ TensorWrapper &tensorWrap() const
    {
        return m_tensorWrap;
    }

    inline const __host__ __device__ StrideType *tensorShape() const
    {
        return m_tensorShape;
    }

    inline __host__ __device__ ValueType borderValue() const
    {
        return ValueType{};
    }

private:
    const TensorWrapper m_tensorWrap                        = {};
    StrideType          m_tensorShape[kNumActiveDimensions] = {0}; // NOSONAR: device storage exposed as pointer.
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
    using typename Base::StrideType;
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
        else
        {
            static_assert(N == 4);
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
        return doGetPtr(std::index_sequence_for<Args...>{}, c...);
    }

private:
    template<typename... Args, std::size_t... Is>
    inline __host__ __device__ ValueType *doGetPtr(std::index_sequence<Is...>, Args... c) const
    {
        return Base::tensorWrap().ptr(GetIndexWithBorder<kBorderType, kActiveDimensions[Is]>(
            static_cast<StrideType>(c), Base::tensorShape()[kMap.from[Is]])...);
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
    using typename Base::StrideType;
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
        const ValueType *p = doGetPtrForCoord(c);

        if (p == nullptr)
        {
            return m_borderValue;
        }

        return *p;
    }

    /**
     * Read-only access returning the accessed element by value.
     *
     * @note Prefer this over `operator[]` in device code: `operator[]` has to return a reference,
     * and what that costs is described on \ref detail::ReadConstantBorder.
     *
     * @note This overload returns by value, while the non-constant border modes return a `const`
     * reference. Consume the result through `auto` / `decltype(auto)`; do not bind a long-lived
     * reference to it.
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed.
     *
     * @return Accessed value, or the border value when \p c is outside the wrapped tensor.
     */
    template<typename DimType, class = Require<std::is_same_v<int, BaseType<DimType>>>>
    inline __host__ __device__ ValueType value(DimType c) const
    {
        return detail::ReadConstantBorder(doGetPtrForCoord(c), m_borderValue);
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
        return doGetPtr(std::index_sequence_for<Args...>{}, c...);
    }

protected:
    // Shared by this specialization's `operator[]` and `value()`, which differ only in what they do
    // with the result, not in how they resolve it. The non-constant specialization still carries
    // its own copy of this ladder. Null means outside the tensor.
    template<typename DimType, class = Require<std::is_same_v<int, BaseType<DimType>>>>
    inline __host__ __device__ ValueType *doGetPtrForCoord(DimType c) const
    {
        constexpr int N = NumElements<DimType>;
        static_assert(kNumDimensions >= N);

        constexpr auto Is = std::make_index_sequence<N>{};

        if constexpr (N == 1)
        {
            if constexpr (NumComponents<DimType> == 0)
            {
                return doGetPtr(Is, c);
            }
            else
            {
                return doGetPtr(Is, c.x);
            }
        }
        else if constexpr (N == 2)
        {
            return doGetPtr(Is, c.y, c.x);
        }
        else if constexpr (N == 3)
        {
            return doGetPtr(Is, c.z, c.y, c.x);
        }
        else
        {
            static_assert(N == 4);
            return doGetPtr(Is, c.w, c.z, c.y, c.x);
        }
    }

private:
    template<typename... Args, std::size_t... Is>
    inline __host__ __device__ ValueType *doGetPtr(std::index_sequence<Is...>, Args... c) const
    {
        if ((IsOutside<kActiveDimensions[Is]>(static_cast<StrideType>(c), Base::tensorShape()[kMap.from[Is]]) || ...))
        {
            return nullptr;
        }
        return Base::tensorWrap().ptr(c...);
    }

    const ValueType m_borderValue = {};
};

/**
 * Border wrapper class specialized for \ref NVCV_BORDER_CONSTANT_COVERAGE.
 *
 * Reads as a zero border so an interpolation over it yields the zero-padded value, and keeps the
 * caller's border value for the coverage blend that \ref InterpolationWrap applies. Sampling with
 * this wrap on its own, without going through an interpolation wrap, is the same as sampling a
 * zero constant border.
 *
 * @tparam TW It is a \ref TensorWrap class with any dimension and type.
 * @tparam ActiveDimensions Flags to inform active (true) or inactive (false) dimensions.
 */
template<class TW, bool... ActiveDimensions>
class BorderWrap<TW, NVCV_BORDER_CONSTANT_COVERAGE, ActiveDimensions...>
    : public BorderWrap<TW, NVCV_BORDER_CONSTANT, ActiveDimensions...>
{
    using Base = BorderWrap<TW, NVCV_BORDER_CONSTANT, ActiveDimensions...>;

public:
    using typename Base::StrideType;
    using typename Base::TensorWrapper;
    using typename Base::ValueType;

    // Preserve fractional fills until the interpolated result is converted to the pixel type.
    using BlendType = ConvertBaseTypeTo<float, std::remove_cv_t<ValueType>>;

    static constexpr auto kBorderType = NVCV_BORDER_CONSTANT_COVERAGE;
    using Base::kActiveDimensions;
    using Base::kMap;
    using Base::kNumActiveDimensions;
    using Base::kNumDimensions;

    BorderWrap() = default;

    template<typename... Args>
    explicit __host__ __device__ BorderWrap(TensorWrapper tensorWrap, BlendType borderValue, Args... tensorShape)
        : Base(tensorWrap, SetAll<std::remove_cv_t<ValueType>>(0), tensorShape...)
        , m_borderValue(borderValue)
    {
    }

    explicit __host__ BorderWrap(const TensorDataStridedCuda &tensor, BlendType borderValue = {})
        : Base(tensor, SetAll<std::remove_cv_t<ValueType>>(0))
        , m_borderValue(borderValue)
    {
    }

    using Base::tensorShape;
    using Base::tensorWrap;

    /**
     * Get the border value blended in by coverage.
     *
     * @return The border value.
     */
    inline const __host__ __device__ BlendType &borderValue() const
    {
        return m_borderValue;
    }

    template<typename DimType>
    inline __host__ __device__ ValueType *ptr(DimType c) const
    {
        return Base::doGetPtrForCoord(c);
    }

private:
    const BlendType m_borderValue = SetAll<BlendType>(0);
};

/**@}*/

/**
 * Type a border value is supplied in for a given pixel type and border mode.
 *
 * \ref NVCV_BORDER_CONSTANT_COVERAGE blends its border value in the float domain and rounds the
 * blended result once, so it takes the value in float; every other mode feeds the border value
 * through as an ordinary pixel and takes it in the pixel type.
 */
template<typename T, NVCVBorderType B>
using BorderValueType
    = std::conditional_t<B == NVCV_BORDER_CONSTANT_COVERAGE, ConvertBaseTypeTo<float, std::remove_cv_t<T>>, T>;

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
 * @tparam StrideType Type of the strdies used in the underlying TensorWrap.
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 * @param[in] borderValue Border value to be used when accessing outside elements in constant border type
 *
 * @return Border wrap useful to access tensor data border aware in H and W in CUDA kernels.
 */
template<typename T, NVCVBorderType B, typename StrideType = int64_t, class = Require<HasTypeTraits<T>>>
__host__ auto CreateBorderWrapNHW(const TensorDataStridedCuda &tensor, BorderValueType<T, B> borderValue = {})
{
    auto tensorAccess = TensorDataAccessStridedImagePlanar::Create(tensor);
    assert(tensorAccess);
    assert(tensorAccess->numRows() <= TypeTraits<StrideType>::max);
    assert(tensorAccess->numCols() <= TypeTraits<StrideType>::max);

    auto tensorWrap = CreateTensorWrapNHW<T, StrideType>(tensor);

    return BorderWrap<decltype(tensorWrap), B, false, true, true>(tensorWrap, borderValue,
                                                                  static_cast<StrideType>(tensorAccess->numRows()),
                                                                  static_cast<StrideType>(tensorAccess->numCols()));
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
 * @tparam StrideType Type of the strdies used in the underlying TensorWrap.
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 * @param[in] borderValue Border value to be used when accessing outside elements in constant border type
 *
 * @return Border wrap useful to access tensor data border aware in H and W in CUDA kernels.
 */
template<typename T, NVCVBorderType B, typename StrideType = int64_t, class = Require<HasTypeTraits<T>>>
__host__ auto CreateBorderWrapNHWC(const TensorDataStridedCuda &tensor, BorderValueType<T, B> borderValue = {})
{
    auto tensorAccess = TensorDataAccessStridedImagePlanar::Create(tensor);
    assert(tensorAccess);
    assert(tensorAccess->numRows() <= TypeTraits<StrideType>::max);
    assert(tensorAccess->numCols() <= TypeTraits<StrideType>::max);

    auto tensorWrap = CreateTensorWrapNHWC<T, StrideType>(tensor);

    return BorderWrap<decltype(tensorWrap), B, false, true, true, false>(
        tensorWrap, borderValue, static_cast<StrideType>(tensorAccess->numRows()),
        static_cast<StrideType>(tensorAccess->numCols()));
}

} // namespace nvcv::cuda

#endif // NVCV_CUDA_BORDER_WRAP_HPP
