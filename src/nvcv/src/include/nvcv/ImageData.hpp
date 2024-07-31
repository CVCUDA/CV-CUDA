/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_IMAGEDATA_HPP
#define NVCV_IMAGEDATA_HPP

#include "ImageData.h"
#include "ImageFormat.hpp"
#include "Optional.hpp"
#include "Size.hpp"

namespace nvcv {

/**
 * @class ImageData
 * @brief Represents image data encapsulated in a convenient interface.
 *
 * This class provides methods to access and manipulate image data.
 * It abstracts the underlying image data representation and provides an interface for higher-level operations.
 */
class ImageData
{
public:
    ImageData() = default;

    /**
     * @brief Construct from an existing NVCVImageData.
     *
     * @param data The NVCVImageData to use for initialization.
     */
    ImageData(const NVCVImageData &data);

    /**
     * @brief Get the image format.
     *
     * @return The format of the image as ImageFormat.
     */
    ImageFormat format() const;

    /**
     * @brief Get a mutable reference to the underlying NVCVImageData.
     *
     * This method provides direct access to the internal data representation, allowing for more specific operations.
     *
     * @return A reference to the NVCVImageData.
     */
    NVCVImageData       &cdata();
    const NVCVImageData &cdata() const;

    /**
     * @brief Casts the image data to a derived type.
     *
     * This template method allows the caller to attempt casting the image data to a more specific type.
     * If the cast is successful, an Optional containing the casted type is returned. Otherwise, an empty Optional is returned.
     *
     * @tparam Derived The target data type to which the cast should be attempted.
     * @return An Optional containing the casted type, or an empty Optional if the cast fails.
     */
    template<typename Derived>
    Optional<Derived> cast() const;

private:
    NVCVImageData m_data{};
};

/**
 * @class ImageDataCudaArray
 * @brief Represents image data stored in a CUDA array format.
 *
 * This class extends the ImageData class, providing additional methods and attributes specific to the CUDA array format.
 */
class ImageDataCudaArray : public ImageData
{
public:
    using Buffer = NVCVImageBufferCudaArray;

    /**
     * @brief Constructor that initializes the image data from an ImageFormat and a CUDA array buffer.
     *
     * @param format The image format.
     * @param buffer The CUDA array buffer.
     */
    explicit ImageDataCudaArray(ImageFormat format, const Buffer &buffer);

    /**
     * @brief Constructor that initializes the image data from an existing NVCVImageData.
     *
     * @param data The NVCVImageData to use for initialization.
     */
    explicit ImageDataCudaArray(const NVCVImageData &data);

    /**
     * @brief Get the number of planes in the image.
     *
     * @return The number of planes.
     */
    int numPlanes() const;

    /**
     * @brief Get a specific plane of the image.
     *
     * @param p The index of the plane.
     * @return The CUDA array corresponding to the specified plane.
     */
    cudaArray_t plane(int p) const;

    /**
     * @brief Check if a given image buffer type is compatible with the CUDA array format.
     *
     * @param kind The image buffer type.
     * @return true if the type is NVCV_IMAGE_BUFFER_CUDA_ARRAY, false otherwise.
     */
    static constexpr bool IsCompatibleKind(NVCVImageBufferType kind)
    {
        return kind == NVCV_IMAGE_BUFFER_CUDA_ARRAY;
    }
};

using ImageBufferStrided = NVCVImageBufferStrided;
using ImagePlaneStrided  = NVCVImagePlaneStrided;

/**
 * @class ImageDataStrided
 * @brief Represents strided image data.
 *
 * This class extends the ImageData class, providing additional methods and attributes specific to the strided image format.
 */
class ImageDataStrided : public ImageData
{
public:
    /**
     * @brief Constructor that initializes the strided image data from an existing NVCVImageData.
     *
     * @param data The NVCVImageData to use for initialization.
     */
    explicit ImageDataStrided(const NVCVImageData &data);

    using Buffer = ImageBufferStrided;

    /**
     * @brief Get the size of the image.
     *
     * @return The size of the image in the form of a Size2D object.
     */
    Size2D size() const;

    /**
     * @brief Get the number of planes in the image.
     *
     * @return The number of planes.
     */
    int numPlanes() const;

    /**
     * @brief Get a specific plane of the image in a strided format.
     *
     * @param p The index of the plane.
     * @return The ImagePlaneStrided corresponding to the specified plane.
     */
    const ImagePlaneStrided &plane(int p) const;

    /**
     * @brief Check if a given image buffer type is compatible with the strided format.
     *
     * @param kind The image buffer type.
     * @return true if the type is either NVCV_IMAGE_BUFFER_STRIDED_CUDA or NVCV_IMAGE_BUFFER_STRIDED_HOST, false otherwise.
     */
    static constexpr bool IsCompatibleKind(NVCVImageBufferType kind)
    {
        return kind == NVCV_IMAGE_BUFFER_STRIDED_CUDA || kind == NVCV_IMAGE_BUFFER_STRIDED_HOST;
    }

protected:
    ImageDataStrided() = default;
};

/**
 * @class ImageDataStridedCuda
 * @brief Represents strided image data specifically for CUDA.
 *
 * This class extends the ImageDataStrided class, offering methods and functionalities tailored for CUDA.
 */
class ImageDataStridedCuda : public ImageDataStrided
{
public:
    /**
     * @brief Constructor that initializes the strided CUDA image data from a format and buffer.
     *
     * @param format The format of the image.
     * @param buffer The buffer containing the image data.
     */
    explicit ImageDataStridedCuda(ImageFormat format, const Buffer &buffer);

    /**
     * @brief Constructor that initializes the strided CUDA image data from an existing NVCVImageData.
     *
     * @param data The NVCVImageData to use for initialization.
     */
    explicit ImageDataStridedCuda(const NVCVImageData &data);

    /**
     * @brief Check if a given image buffer type is compatible with the strided CUDA format.
     *
     * @param kind The image buffer type.
     * @return true if the type is NVCV_IMAGE_BUFFER_STRIDED_CUDA, false otherwise.
     */
    static constexpr bool IsCompatibleKind(NVCVImageBufferType kind)
    {
        return kind == NVCV_IMAGE_BUFFER_STRIDED_CUDA;
    }
};

/**
 * @class ImageDataStridedHost
 * @brief Represents strided image data specifically for host.
 *
 * This class extends the ImageDataStrided class, offering methods and functionalities tailored for the host environment.
 */
class ImageDataStridedHost : public ImageDataStrided
{
public:
    /**
     * @brief Constructor that initializes the strided host image data from a format and buffer.
     *
     * @param format The format of the image.
     * @param buffer The buffer containing the image data.
     */
    explicit ImageDataStridedHost(ImageFormat format, const Buffer &buffer);

    /**
     * @brief Constructor that initializes the strided host image data from an existing NVCVImageData.
     *
     * @param data The NVCVImageData to use for initialization.
     */
    explicit ImageDataStridedHost(const NVCVImageData &data);

    /**
     * @brief Check if a given image buffer type is compatible with the strided host format.
     *
     * @param kind The image buffer type.
     * @return true if the type is NVCV_IMAGE_BUFFER_STRIDED_HOST, false otherwise.
     */
    static constexpr bool IsCompatibleKind(NVCVImageBufferType kind)
    {
        return kind == NVCV_IMAGE_BUFFER_STRIDED_HOST;
    }
};

} // namespace nvcv

#include "detail/ImageDataImpl.hpp"

#endif // NVCV_DETAIL_IMAGEDATA_HPP
