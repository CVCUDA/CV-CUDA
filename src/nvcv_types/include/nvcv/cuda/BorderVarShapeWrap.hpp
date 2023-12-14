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
 * @file BorderVarShapeWrap.hpp
 *
 * @brief Defines border of variable shapes wrapper over image batch for border handling.
 */

#ifndef NVCV_CUDA_BORDER_VAR_SHAPE_WRAP_HPP
#define NVCV_CUDA_BORDER_VAR_SHAPE_WRAP_HPP

#include "BorderWrap.hpp"             // for IsOutside, etc.
#include "ImageBatchVarShapeWrap.hpp" // for ImageBatchVarShapeWrap, etc.
#include "TypeTraits.hpp"             // for NumElements, etc.

#include <nvcv/ImageBatchData.hpp> // for ImageBatchVarShapeDataStridedCuda, etc.

namespace nvcv::cuda {

namespace detail {

template<class IW, NVCVBorderType B>
class BorderIWImpl
{
public:
    using ImageBatchWrapper = IW;
    using ValueType         = typename ImageBatchWrapper::ValueType;

    static constexpr int            kNumDimensions = ImageBatchWrapper::kNumDimensions;
    static constexpr NVCVBorderType kBorderType    = B;

    static constexpr bool kActiveDimensions[]  = {false, false, true, true};
    static constexpr int  kNumActiveDimensions = 2;

    BorderIWImpl() = default;

    explicit __host__ __device__ BorderIWImpl(ImageBatchWrapper imageBatchWrap)
        : m_imageBatchWrap(imageBatchWrap)
    {
    }

    explicit __host__ BorderIWImpl(const ImageBatchVarShapeDataStridedCuda &images)
        : m_imageBatchWrap(images)
    {
    }

    explicit __host__ BorderIWImpl(const ImageBatchVarShapeDataStridedCuda &images, int numChannels)
        : m_imageBatchWrap(images, numChannels)
    {
    }

    inline const __host__ __device__ ImageBatchWrapper &imageBatchWrap() const
    {
        return m_imageBatchWrap;
    }

    inline __host__ __device__ ValueType borderValue() const
    {
        return ValueType{};
    }

protected:
    const ImageBatchWrapper m_imageBatchWrap = {};
};

template<typename T, NVCVBorderType B>
using BorderVarShapeWrapImpl = BorderIWImpl<ImageBatchVarShapeWrap<T>, B>;

template<typename T, NVCVBorderType B>
using BorderVarShapeWrapNHWCImpl = BorderIWImpl<ImageBatchVarShapeWrapNHWC<T>, B>;

} // namespace detail

/**
 * @defgroup NVCV_CPP_CUDATOOLS_BORDERVARSHAPEWRAP BorderVarShapeWrap classes
 * @{
 */

/**
 * Border var-shape wrapper class used to wrap an \ref ImageBatchVarShapeWrap adding border handling to it.
 *
 * This class wraps an \ref ImageBatchVarShapeWrap to add border handling functionality.  It provides the methods
 * \p ptr and \p operator[] to do the same semantic access (pointer or reference) in the wrapped \ref
 * ImageBatchVarShapeWrap but border aware on width and height as active dimensions.
 *
 * @code
 * using PixelType = ...;
 * using ImageBatchWrap = ImageBatchVarShapeWrap<PixelType>;
 * using BorderVarShape = BorderVarShapeWrap<PixelType, NVCV_BORDER_REPLICATE>;
 * ImageBatchWrap dst(...);
 * ImageBatchWrap srcImageBatch(...);
 * BorderVarShape src(srcImageBatch);
 * dim3 grid{...}, block{...};
 * int2 fillBorderSize{2, 2};
 * FillBorder<<<grid, block>>>(dst, src, src.numImages(), fillBorderSize);
 *
 * template<typename T, NVCVBorderType B>
 * __global__ void FillBorder(ImageBatchVarShapeWrap<T> dst, BorderVarShapeWrap<T, B> src, int ns, int2 bs)
 * {
 *     int3 dstCoord = StaticCast<int>(blockIdx * blockDim + threadIdx);
 *     if (dstCoord.x >= dst.width(dstCoord.z) || dstCoord.y >= dst.height(dstCoord.z) || dstCoord.z >= ns)
 *         return;
 *     int3 srcCoord = {dstCoord.x - bs.x, dstCoord.y - bs.y, dstCoord.z};
 *     dst[dstCoord] = src[srcCoord];
 * }
 * @endcode
 *
 * @tparam T Type (it can be const) of each element inside the image batch var-shape wrapper.
 * @tparam B It is a \ref NVCVBorderType indicating the border to be used.
 */
template<typename T, NVCVBorderType B>
class BorderVarShapeWrap : public detail::BorderVarShapeWrapImpl<T, B>
{
    using Base = detail::BorderVarShapeWrapImpl<T, B>;

public:
    using typename Base::ImageBatchWrapper;
    using typename Base::ValueType;

    using Base::kBorderType;
    using Base::kNumDimensions;

    BorderVarShapeWrap() = default;

    /**
     * Constructs a BorderVarShapeWrap by wrapping an \p imageBatchWrap.
     *
     * @param[in] imageBatchWrap An \ref ImageBatchVarShapeWrap object to be wrapped.
     * @param[in] borderValue The border value is ignored in non-constant border types.
     */
    explicit __host__ __device__ BorderVarShapeWrap(ImageBatchWrapper imageBatchWrap, ValueType borderValue = {})
        : Base(imageBatchWrap)
    {
    }

    /**
     * Constructs a BorderVarShapeWrap by wrapping \p images.
     *
     * @param[in] images A \p ImageBatchVarShapeDataStridedCuda with image batch information.
     * @param[in] borderValue The border value is ignored in non-constant border types.
     */
    explicit __host__ BorderVarShapeWrap(const ImageBatchVarShapeDataStridedCuda &images, ValueType borderValue = {})
        : Base(images)
    {
    }

    // Get the image batch var-shape wrapped by this border var-shape wrap.
    using Base::imageBatchWrap;

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type).
     *
     * @param[in] c 4D coordinate (w sample, z plane, y row and x column) to be accessed.
     *
     * @return Accessed (const) reference.
     */
    inline __host__ __device__ ValueType &operator[](int4 c) const
    {
        return *doGetPtr(c.w, c.z, c.y, c.x);
    }

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type, considering plane=0).
     *
     * @param[in] c 3D coordinate (z sample, y row and x column) to be accessed.
     *
     * @return Accessed (const) reference.
     */
    inline __host__ __device__ ValueType &operator[](int3 c) const
    {
        return *doGetPtr(c.z, 0, c.y, c.x);
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the given coordinates.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] p Plane index in the selected image.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The (const) pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ ValueType *ptr(int s, int p, int y, int x) const
    {
        return doGetPtr(s, p, y, x);
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the given coordinates (considering plane=0).
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The (const) pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ ValueType *ptr(int s, int y, int x) const
    {
        return doGetPtr(s, 0, y, x);
    }

private:
    inline __host__ __device__ ValueType *doGetPtr(int s, int p, int y, int x) const
    {
        y = GetIndexWithBorder<kBorderType>(y, Base::m_imageBatchWrap.height(s, p));
        x = GetIndexWithBorder<kBorderType>(x, Base::m_imageBatchWrap.width(s, p));
        return Base::m_imageBatchWrap.ptr(s, p, y, x);
    }
};

/**
 * Border var-shape wrapper class specialized for \ref NVCV_BORDER_CONSTANT.
 *
 * @tparam T Type (it can be const) of each element inside the image batch var-shape wrapper.
 */
template<typename T>
class BorderVarShapeWrap<T, NVCV_BORDER_CONSTANT> : public detail::BorderVarShapeWrapImpl<T, NVCV_BORDER_CONSTANT>
{
    using Base = detail::BorderVarShapeWrapImpl<T, NVCV_BORDER_CONSTANT>;

public:
    using typename Base::ImageBatchWrapper;
    using typename Base::ValueType;

    using Base::kBorderType;
    using Base::kNumDimensions;

    BorderVarShapeWrap() = default;

    /**
     * Constructs a BorderVarShapeWrap by wrapping an \p imageBatchWrap.
     *
     * @param[in] imageBatchWrap An \ref ImageBatchVarShapeWrap object to be wrapped.
     * @param[in] borderValue The border value to be used when accessing outside the image batch.
     */
    explicit __host__ __device__ BorderVarShapeWrap(ImageBatchWrapper imageBatchWrap, ValueType borderValue = {})
        : Base(imageBatchWrap)
        , m_borderValue(borderValue)
    {
    }

    /**
     * Constructs a BorderVarShapeWrap by wrapping \p images.
     *
     * @param[in] images An \p ImageBatchVarShapeDataStridedCuda with image batch information.
     * @param[in] borderValue The border value to be used when accessing outside the tensor.
     */
    explicit __host__ BorderVarShapeWrap(const ImageBatchVarShapeDataStridedCuda &images, ValueType borderValue = {})
        : Base(images)
        , m_borderValue(borderValue)
    {
    }

    // Get the image batch var-shape wrapped by this border var-shape wrap.
    using Base::imageBatchWrap;

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
     * Subscript operator for read-only or read-and-write access (depending on value type).
     *
     * @param[in] c 4D coordinate (w sample, z plane, y row and x column) to be accessed.
     *
     * @return Accessed const reference.
     */
    inline const __host__ __device__ ValueType &operator[](int4 c) const
    {
        const ValueType *p = doGetPtr(c.w, c.z, c.y, c.x);

        if (p == nullptr)
        {
            return m_borderValue;
        }

        return *p;
    }

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type, considering plane=0).
     *
     * @param[in] c 3D coordinate (z sample, y row and x column) to be accessed.
     *
     * @return Accessed const reference.
     */
    inline const __host__ __device__ ValueType &operator[](int3 c) const
    {
        const ValueType *p = doGetPtr(c.z, 0, c.y, c.x);

        if (p == nullptr)
        {
            return m_borderValue;
        }

        return *p;
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the given coordinates.
     *
     * @note This method may return a nullptr pointer when accessing outside the wrapped \ref
     * ImageBatchVarShapeWrap since this border wrap is for constant border and there is no pointer representation
     * for the constant border value.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] p Plane index in the selected image.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The (const) pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ ValueType *ptr(int s, int p, int y, int x) const
    {
        return doGetPtr(s, p, y, x);
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the given coordinates (considering plane=0).
     *
     * @note This method may return a nullptr pointer when accessing outside the wrapped \ref
     * ImageBatchVarShapeWrap since this border wrap is for constant border and there is no pointer representation
     * for the constant border value.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The (const) pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ ValueType *ptr(int s, int y, int x) const
    {
        return doGetPtr(s, 0, y, x);
    }

private:
    inline __host__ __device__ ValueType *doGetPtr(int s, int p, int y, int x) const
    {
        if (IsOutside(y, Base::m_imageBatchWrap.height(s, p)) || IsOutside(x, Base::m_imageBatchWrap.width(s, p)))
        {
            return nullptr;
        }
        return Base::m_imageBatchWrap.ptr(s, p, y, x);
    }

    const ValueType m_borderValue = SetAll<ValueType>(0);
};

template<typename T, NVCVBorderType B>
class BorderVarShapeWrapNHWC : public detail::BorderVarShapeWrapNHWCImpl<T, B>
{
    using Base = detail::BorderVarShapeWrapNHWCImpl<T, B>;

public:
    using typename Base::ImageBatchWrapper;
    using typename Base::ValueType;

    using Base::kBorderType;
    using Base::kNumDimensions;

    BorderVarShapeWrapNHWC() = default;

    /**
     * Constructs a BorderVarShapeWrapNHWC by wrapping an \p imageBatchWrap.
     *
     * @param[in] imageBatchWrap An \ref ImageBatchVarShapeWrapNHWC object to be wrapped.
     * @param[in] borderValue The border value is ignored in non-constant border types.
     */
    explicit __host__ __device__ BorderVarShapeWrapNHWC(ImageBatchWrapper imageBatchWrap, ValueType borderValue = {})
        : Base(imageBatchWrap)
    {
    }

    /**
     * Constructs a BorderVarShapeWrapNHWC by wrapping \p images.
     *
     * @param[in] images A \p ImageBatchVarShapeDataStridedCuda with image batch information.
     * @param[in] numChannels The number of (interleaved) channels inside the wrapper.
     * @param[in] borderValue The border value is ignored in non-constant border types.
     */
    explicit __host__ BorderVarShapeWrapNHWC(const ImageBatchVarShapeDataStridedCuda &images, int numChannels,
                                             ValueType borderValue = {})
        : Base(images, numChannels)
    {
    }

    // Get the image batch var-shape wrapped by this border var-shape wrap.
    using Base::imageBatchWrap;

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type).
     *
     * @param[in] c 4D coordinate (x column, y row, z sample, w channel) to be accessed.
     *
     * @return Accessed (const) reference.
     */
    inline __host__ __device__ ValueType &operator[](int4 c) const
    {
        return *doGetPtr(c.z, c.y, c.x, c.w);
    }

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type, considering plane=0).
     *
     * @param[in] c 3D coordinate (x column, y row, z sample) (first channel) to be accessed.
     *
     * @return Accessed (const) reference.
     */
    inline __host__ __device__ ValueType &operator[](int3 c) const
    {
        return *doGetPtr(c.z, c.y, c.x, 0);
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the given coordinates.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     * @param[in] c Channel index in the image.
     *
     * @return The (const) pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ ValueType *ptr(int s, int y, int x, int c) const
    {
        return doGetPtr(s, y, x, c);
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the given coordinates (considering plane=0).
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The (const) pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ ValueType *ptr(int s, int y, int x) const
    {
        return doGetPtr(s, y, x, 0);
    }

private:
    inline __host__ __device__ ValueType *doGetPtr(int s, int y, int x, int c) const
    {
        y = GetIndexWithBorder<kBorderType>(y, Base::m_imageBatchWrap.height(s, 0));
        x = GetIndexWithBorder<kBorderType>(x, Base::m_imageBatchWrap.width(s, 0));
        return Base::m_imageBatchWrap.ptr(s, y, x, c);
    }
};

/**
 * Border var-shape wrapper class specialized for \ref NVCV_BORDER_CONSTANT.
 *
 * @tparam T Type (it can be const) of each element inside the image batch var-shape wrapper.
 */
template<typename T>
class BorderVarShapeWrapNHWC<T, NVCV_BORDER_CONSTANT>
    : public detail::BorderVarShapeWrapNHWCImpl<T, NVCV_BORDER_CONSTANT>
{
    using Base = detail::BorderVarShapeWrapNHWCImpl<T, NVCV_BORDER_CONSTANT>;

public:
    using typename Base::ImageBatchWrapper;
    using typename Base::ValueType;

    using Base::kBorderType;
    using Base::kNumDimensions;

    BorderVarShapeWrapNHWC() = default;

    /**
     * Constructs a BorderVarShapeNHWCWrap by wrapping an \p imageBatchWrap.
     *
     * @param[in] imageBatchWrap An \ref ImageBatchVarShapeWrapNHWC object to be wrapped.
     * @param[in] borderValue The border value to be used when accessing outside the image batch.
     */
    explicit __host__ __device__ BorderVarShapeWrapNHWC(ImageBatchWrapper imageBatchWrap, ValueType borderValue = {})
        : Base(imageBatchWrap)
        , m_borderValue(borderValue)
    {
    }

    /**
     * Constructs a BorderVarShapeWrapNHWC by wrapping \p images.
     *
     * @param[in] images An \p ImageBatchVarShapeDataStridedCuda with image batch information.
     * @param[in] numChannels The number of (interleaved) channels inside the wrapper.
     * @param[in] borderValue The border value to be used when accessing outside the tensor.
     */
    explicit __host__ BorderVarShapeWrapNHWC(const ImageBatchVarShapeDataStridedCuda &images, int numChannels,
                                             ValueType borderValue = {})
        : Base(images, numChannels)
        , m_borderValue(borderValue)
    {
    }

    // Get the image batch var-shape wrapped by this border var-shape wrap.
    using Base::imageBatchWrap;

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type).
     *
     * @param[in] c 4D coordinate (x column, y row, z sample, w channel) to be accessed.
     *
     * @return Accessed (const) reference.
     */
    inline __host__ __device__ ValueType &operator[](int4 c) const
    {
        ValueType *p = doGetPtr(c.z, c.y, c.x, c.w);

        if (p == nullptr)
        {
            return m_borderValue;
        }

        return *p;
    }

    /**
     * Subscript operator for read-only or read-and-write access (depending on value type, considering plane=0).
     *
     * @param[in] c 3D coordinate (x column, y row, z sample) (first channel) to be accessed.
     *
     * @return Accessed (const) reference.
     */
    inline __host__ __device__ ValueType &operator[](int3 c) const
    {
        ValueType *p = doGetPtr(c.z, c.y, c.x, 0);

        if (p == nullptr)
        {
            return m_borderValue;
        }

        return *p;
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the given coordinates.
     *
     * @note This method may return a nullptr pointer when accessing outside the wrapped \ref
     * ImageBatchVarShapeWrap since this border wrap is for constant border and there is no pointer representation
     * for the constant border value.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     * @param[in] c Channel index in the image.
     *
     * @return The (const) pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ ValueType *ptr(int s, int y, int x, int c) const
    {
        return doGetPtr(s, y, x, c);
    }

    /**
     * Get a read-only or read-and-write proxy (as pointer) at the given coordinates (considering plane=0).
     *
     * @note This method may return a nullptr pointer when accessing outside the wrapped \ref
     * ImageBatchVarShapeWrap since this border wrap is for constant border and there is no pointer representation
     * for the constant border value.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The (const) pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ ValueType *ptr(int s, int y, int x) const
    {
        return doGetPtr(s, y, x, 0);
    }

private:
    inline __host__ __device__ ValueType *doGetPtr(int s, int y, int x, int c) const
    {
        if (IsOutside(y, Base::m_imageBatchWrap.height(s, 0)) || IsOutside(x, Base::m_imageBatchWrap.width(s, 0)))
        {
            return nullptr;
        }
        return Base::m_imageBatchWrap.ptr(s, y, x, c);
    }

    ValueType m_borderValue = SetAll<ValueType>(0);
};

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_BORDER_VAR_SHAPE_WRAP_HPP
