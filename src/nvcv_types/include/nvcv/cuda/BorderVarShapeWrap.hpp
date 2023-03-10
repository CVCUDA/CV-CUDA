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

#include <nvcv/IImageBatchData.hpp> // for IImageBatchVarShapeDataStridedCuda, etc.

namespace nvcv::cuda {

namespace detail {

template<typename T, NVCVBorderType B>
class BorderVarShapeWrapImpl
{
public:
    using ImageBatchWrap = ImageBatchVarShapeWrap<T>;
    using ValueType      = typename ImageBatchWrap::ValueType;

    static constexpr int            kNumDimensions = ImageBatchWrap::kNumDimensions;
    static constexpr NVCVBorderType kBorderType    = B;

    BorderVarShapeWrapImpl() = default;

    explicit __host__ __device__ BorderVarShapeWrapImpl(ImageBatchWrap imageBatchWrap)
        : m_imageBatchWrap(imageBatchWrap)
    {
    }

    explicit __host__ BorderVarShapeWrapImpl(const IImageBatchVarShapeDataStridedCuda &images)
        : m_imageBatchWrap(images)
    {
    }

    inline const __host__ __device__ ImageBatchWrap &imageBatchWrap() const
    {
        return m_imageBatchWrap;
    }

protected:
    const ImageBatchWrap m_imageBatchWrap = {};
};

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
    using typename Base::ImageBatchWrap;
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
    explicit __host__ __device__ BorderVarShapeWrap(ImageBatchWrap imageBatchWrap, ValueType borderValue = {})
        : Base(imageBatchWrap)
    {
    }

    /**
     * Constructs a BorderVarShapeWrap by wrapping \p images.
     *
     * @param[in] images A \p IImageBatchVarShapeDataStridedCuda with image batch information.
     * @param[in] borderValue The border value is ignored in non-constant border types.
     */
    explicit __host__ BorderVarShapeWrap(const IImageBatchVarShapeDataStridedCuda &images, ValueType borderValue = {})
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
    using typename Base::ImageBatchWrap;
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
    explicit __host__ __device__ BorderVarShapeWrap(ImageBatchWrap imageBatchWrap, ValueType borderValue = {})
        : Base(imageBatchWrap)
        , m_borderValue(borderValue)
    {
    }

    /**
     * Constructs a BorderVarShapeWrap by wrapping \p images.
     *
     * @param[in] images An \p IImageBatchVarShapeDataStridedCuda with image batch information.
     * @param[in] borderValue The border value to be used when accessing outside the tensor.
     */
    explicit __host__ BorderVarShapeWrap(const IImageBatchVarShapeDataStridedCuda &images, ValueType borderValue = {})
        : Base(images)
        , m_borderValue(borderValue)
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
        ValueType *p = doGetPtr(c.w, c.z, c.y, c.x);

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
     * @return Accessed (const) reference.
     */
    inline __host__ __device__ ValueType &operator[](int3 c) const
    {
        ValueType *p = doGetPtr(c.z, 0, c.y, c.x);

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

    ValueType m_borderValue = SetAll<ValueType>(0);
};

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_BORDER_VAR_SHAPE_WRAP_HPP
