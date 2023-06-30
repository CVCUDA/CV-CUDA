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
 * @file ImageBatchVarShapeWrap.hpp
 *
 * @brief Defines a wrapper of an image batch (or a list of images) of variable shapes.
 */

#ifndef NVCV_CUDA_IMAGE_BATCH_VAR_SHAPE_WRAP_HPP
#define NVCV_CUDA_IMAGE_BATCH_VAR_SHAPE_WRAP_HPP

#include "TypeTraits.hpp" // for HasTypeTraits, etc.

#include <nvcv/ImageBatchData.hpp> // for ImageBatchVarShapeDataStridedCuda, etc.

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_IMAGEWRAP Image Wrapper classes
 * @{
 */

/**
 * Image batch var-shape wrapper class to wrap ImageBatchVarShapeDataStridedCuda.
 *
 * ImageBatchVarShapeWrap is a wrapper of an image batch (or a list of images) of variable shapes.  The template
 * parameter \p T is the type of each element inside the wrapper, and it can be compound type to represent a pixel
 * type, e.g. uchar4 for RGBA images.
 *
 * @code
 * cudaStream_t stream;
 * cudaStreamCreate(&stream);
 * nvcv::ImageBatchVarShape imageBatch(samples);
  auto *imageBatchData  = imageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream)
 * nvcv::cuda::ImageBatchVarShapeWrap<uchar4> wrap(*imageBatchData);
 * // Now wrap can be used in device code to access elements of the image batch via operator[] or ptr method.
 * @endcode
 *
 * @tparam T Type (it can be const) of each element inside the image batch var-shape wrapper.
 */
template<typename T>
class ImageBatchVarShapeWrap;

template<typename T>
class ImageBatchVarShapeWrap<const T>
{
    // It is a requirement of this class that its type has type traits.
    static_assert(HasTypeTraits<T>, "ImageBatchVarShapeWrap<T> can only be used if T has type traits");

public:
    // The type provided as template parameter is the value type, i.e. the type of each element inside this wrapper.
    using ValueType = const T;

    // The number of dimensions is fixed as 4, one for each image in the batch and one for each plane of a 2D image.
    static constexpr int kNumDimensions = 4;
    // The number of variable strides is fixed as 3, meaning it uses 3 run-time strides.
    static constexpr int kVariableStrides = 3;
    // The number of constant strides is fixed as 1, meaning it uses 1 compile-time stride.
    static constexpr int kConstantStrides = 1;

    ImageBatchVarShapeWrap() = default;

    /**
     * Constructs a constant ImageBatchVarShapeWrap by wrapping an \p images argument.
     *
     * @param[in] images Reference to the list of images that will be wrapped.
     */
    __host__ ImageBatchVarShapeWrap(const ImageBatchVarShapeDataStridedCuda &images)
        : m_imageList(images.imageList())
    {
    }

    /**
     * Get plane \p p of image \p s sample.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] p Plane index in the image.
     *
     * @return The plane of the given image sample in batch.
     */
    inline const __host__ __device__ NVCVImagePlaneStrided plane(int s, int p = 0) const
    {
        return m_imageList[s].planes[p];
    }

    /**
     * Get width of plane \p p of image \p s sample.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] p Plane index in the image.
     *
     * @return The width of the given image plane in batch.
     */
    inline __host__ __device__ int width(int s, int p = 0) const
    {
        return m_imageList[s].planes[p].width;
    }

    /**
     * Get height of plane \p p of image \p s sample.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] p Plane index in the image.
     *
     * @return The height of the given image plane in batch.
     */
    inline __host__ __device__ int height(int s, int p = 0) const
    {
        return m_imageList[s].planes[p].height;
    }

    /**
     * Get row stride in bytes of plane \p p of image \p s sample.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] p Plane index in the image.
     *
     * @return The row stride in bytes of the given image plane in batch.
     */
    inline __host__ __device__ int rowStride(int s, int p = 0) const
    {
        return m_imageList[s].planes[p].rowStride;
    }

    /**
     * Subscript operator for read-only access.
     *
     * @param[in] c 4D coordinates (w sample, z plane, y row and x column) to be accessed.
     *
     * @return Accessed const reference.
     */
    inline const __host__ __device__ T &operator[](int4 c) const
    {
        return *doGetPtr(c.w, c.z, c.y, c.x);
    }

    /**
     * Subscript operator for read-only access (considering plane=0).
     *
     * @param[in] c 3D coordinates (z sample, y row and x column) to be accessed.
     *
     * @return Accessed const reference.
     */
    inline const __host__ __device__ T &operator[](int3 c) const
    {
        return *doGetPtr(c.z, 0, c.y, c.x);
    }

    /**
     * Get a read-only proxy (as pointer) of the given coordinates.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] p Plane index in the image.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The const pointer to the beginning of the given coordinates.
     */
    inline const __host__ __device__ T *ptr(int s, int p, int y, int x) const
    {
        return doGetPtr(s, p, y, x);
    }

    /**
     * Get a read-only proxy (as pointer) of the given coordinates (considering plane=0).
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The const pointer to the beginning of the given coordinates.
     */
    inline const __host__ __device__ T *ptr(int s, int y, int x) const
    {
        return doGetPtr(s, 0, y, x);
    }

    /**
     * Get a read-only proxy (as pointer) of the given coordinates (considering plane=0).
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     *
     * @return The const pointer to the beginning of the given coordinates.
     */
    inline const __host__ __device__ T *ptr(int s, int y) const
    {
        return doGetPtr(s, 0, y, 0);
    }

protected:
    inline const __host__ __device__ T *doGetPtr(int s, int p, int y, int x) const
    {
        int offset = y * this->rowStride(s, p) + x * sizeof(T);

        return reinterpret_cast<const T *>(m_imageList[s].planes[p].basePtr + offset);
    }

private:
    const NVCVImageBufferStrided *m_imageList = nullptr;
};

/**
 * Image batch var-shape wrapper class to wrap ImageBatchVarShapeDataStridedCuda.
 *
 * This class is specialized for non-constant value type.
 *
 * @tparam T Type (non-const) of each element inside the image batch var-shape wrapper.
 */
template<typename T>
class ImageBatchVarShapeWrap : public ImageBatchVarShapeWrap<const T>
{
    using Base = ImageBatchVarShapeWrap<const T>;

public:
    using ValueType = T;

    using Base::kConstantStrides;
    using Base::kNumDimensions;
    using Base::kVariableStrides;

    ImageBatchVarShapeWrap() = default;

    /**
     * Constructs a ImageBatchVarShapeWrap by wrapping an \p images argument.
     *
     * @param[in] images Reference to the list of images that will be wrapped.
     */
    __host__ ImageBatchVarShapeWrap(const ImageBatchVarShapeDataStridedCuda &images)
        : Base(images)
    {
    }

    using Base::height;
    using Base::plane;
    using Base::rowStride;
    using Base::width;

    /**
     * Subscript operator for read-and-write access.
     *
     * @param[in] c 4D coordinates (w sample, z plane, y row and x column) to be accessed.
     *
     * @return Accessed reference.
     */
    inline __host__ __device__ T &operator[](int4 c) const
    {
        return *doGetPtr(c.w, c.z, c.y, c.x);
    }

    /**
     * Subscript operator for read-and-write access (considering plane = 0).
     *
     * @param[in] c 3D coordinates (z sample, y row and x column) to be accessed.
     *
     * @return Accessed reference.
     */
    inline __host__ __device__ T &operator[](int3 c) const
    {
        return *doGetPtr(c.z, 0, c.y, c.x);
    }

    /**
     * Get a read-and-write proxy (as pointer) of the given coordinates.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] p Plane index in the image.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ T *ptr(int s, int p, int y, int x) const
    {
        return doGetPtr(s, p, y, x);
    }

    /**
     * Get a read-and-write proxy (as pointer) of the given coordinates (considering plane=0).
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ T *ptr(int s, int y, int x) const
    {
        return doGetPtr(s, 0, y, x);
    }

    /**
     * Get a read-and-write proxy (as pointer) of the given coordinates (considering plane=0).
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     *
     * @return The pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ T *ptr(int s, int y) const
    {
        return doGetPtr(s, 0, y, 0);
    }

protected:
    inline __host__ __device__ T *doGetPtr(int s, int p, int y, int x) const
    {
        return const_cast<T *>(Base::doGetPtr(s, p, y, x));
    }
};

/**
 * Image batch var-shape wrapper NHWC class to wrap ImageBatchVarShapeDataStridedCuda and number of channels.
 *
 * This class handles number of channels as a separate run-time parameter instead of built-in \p T.  It considers
 * interleaved channels, where they appear in a packed sequence at the last dimension (thus NHWC).  It also
 * considers each image in the batch has a single plane.
 *
 * @note The class \ref ImageBatchVarShapeWrap can be used with its template parameter \p T type as a compound
 * type, where its number of elements yield the number of channels.
 *
 * @tparam T Type (it can be const) of each element inside this wrapper.
 */
template<typename T>
class ImageBatchVarShapeWrapNHWC : ImageBatchVarShapeWrap<T>
{
    using Base = ImageBatchVarShapeWrap<T>;

public:
    using ValueType = T;

    using Base::kConstantStrides;
    using Base::kNumDimensions;
    using Base::kVariableStrides;

    ImageBatchVarShapeWrapNHWC() = default;

    /**
     * Constructs a ImageBatchVarShapeWrapNHWC by wrapping an \p images and \p numChannels arguments.
     *
     * @param[in] images Reference to the list of images that will be wrapped.
     * @param[in] numChannels The number of (interleaved) channels inside the wrapper.
     */
    __host__ ImageBatchVarShapeWrapNHWC(const ImageBatchVarShapeDataStridedCuda &images, int numChannels)
        : Base(images)
        , m_numChannels(numChannels)
    {
#ifndef NDEBUG
        auto formats = images.hostFormatList();
        for (int i = 0; i < images.numImages(); ++i)
        {
            assert(1 == nvcv::ImageFormat{formats[i]}.numPlanes()
                   && "This wrapper class is only for single-plane images in batch");
        }
#endif
    }

    // Get the number of channels.
    inline __host__ __device__ const int &numChannels() const
    {
        return m_numChannels;
    }

    using Base::height;
    using Base::plane;
    using Base::rowStride;
    using Base::width;

    /**
     * Subscript operator for either read-only or read-and-write access.
     *
     * @param[in] c 4D coordinates (x column, y row, z sample, w channel) to be accessed.
     *
     * @return Accessed reference.
     */
    inline __host__ __device__ T &operator[](int4 c) const
    {
        return *doGetPtr(c.z, c.y, c.x, c.w);
    }

    /**
     * Subscript operator for either read-only or read-and-write access.
     *
     * @param[in] c 3D coordinates (x column, y row, z sample) (first channel) to be accessed.
     *
     * @return Accessed reference.
     */
    inline __host__ __device__ T &operator[](int3 c) const
    {
        return *doGetPtr(c.z, c.y, c.x, 0);
    }

    /**
     * Get either read-only or read-and-write proxy (as pointer) of the given coordinates.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     * @param[in] c Channel index in the image.
     *
     * @return The pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ T *ptr(int s, int y, int x, int c) const
    {
        return doGetPtr(s, y, x, c);
    }

    /**
     * Get either read-only or read-and-write proxy (as pointer) of the given coordinates.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ T *ptr(int s, int y, int x) const
    {
        return doGetPtr(s, y, x, 0);
    }

    /**
     * Get either read-only or read-and-write proxy (as pointer) of the given coordinates.
     *
     * @param[in] s Sample image index in the list.
     * @param[in] y Row index in the selected image.
     * @param[in] x Column index in the selected image.
     *
     * @return The pointer to the beginning of the given coordinates.
     */
    inline __host__ __device__ T *ptr(int s, int y) const
    {
        return doGetPtr(s, y, 0, 0);
    }

private:
    inline __host__ __device__ T *doGetPtr(int s, int y, int x, int c) const
    {
        return Base::doGetPtr(s, 0, y, 0) + x * m_numChannels + c;
    }

    int m_numChannels = 1;
};

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_IMAGE_BATCH_VAR_SHAPE_HPP
