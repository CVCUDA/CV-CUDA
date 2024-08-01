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

#ifndef NVCV_IMAGEBATCHDATA_HPP
#define NVCV_IMAGEBATCHDATA_HPP

#include "ImageBatchData.h"
#include "ImageData.hpp"
#include "ImageFormat.hpp"
#include "Optional.hpp"
#include "detail/CudaFwd.h"

namespace nvcv {

/**
 * @class ImageBatchData
 * @brief Represents the underlying data of an image batch.
 *
 * This class provides an interface to access and manipulate the data associated
 * with a batch of images. It also allows casting the data to a derived type.
 */
class ImageBatchData
{
public:
    /**
     * @brief Cast the image batch data to a specific derived type.
     *
     * @tparam Derived The target derived type.
     * @return An Optional containing the casted data if successful, otherwise NullOpt.
     */
    template<typename Derived>
    Optional<Derived> cast() const;

    /**
     * @brief Check if a specific buffer type is compatible with this class.
     *
     * @param type The buffer type to check.
     * @return True if the type is not NVCV_IMAGE_BATCH_BUFFER_NONE, false otherwise.
     */
    static constexpr bool IsCompatibleKind(NVCVImageBatchBufferType type)
    {
        return type != NVCV_IMAGE_BATCH_BUFFER_NONE;
    }

    ImageBatchData() = default;

    /**
     * @brief Construct from an existing NVCVImageBatchData.
     *
     * @param data The NVCVImageBatchData to use for initialization.
     */
    ImageBatchData(const NVCVImageBatchData &data)
        : m_data(data)
    {
    }

    /**
     * @brief Access the underlying constant NVCVImageBatchData.
     *
     * @return A reference to the underlying NVCVImageBatchData.
     */
    const NVCVImageBatchData &cdata() const &;

    /**
     * @brief Move the underlying NVCVImageBatchData.
     *
     * @return The moved NVCVImageBatchData.
     */
    NVCVImageBatchData cdata() &&;

    /**
     * @brief Get the number of images in the batch.
     *
     * @return The number of images.
     */
    int32_t numImages() const;

protected:
    NVCVImageBatchData &data() &;

private:
    NVCVImageBatchData m_data{};
};

/**
 * @class ImageBatchVarShapeData
 * @brief Represents the data of a variable shaped image batch.
 *
 * This class provides an interface to access and manipulate the data associated
 * with a batch of variable shaped images. It extends the base ImageBatchData class.
 */
class ImageBatchVarShapeData : public ImageBatchData
{
public:
    using ImageBatchData::ImageBatchData;

    /**
     * @brief Construct from an existing NVCVImageBatchData.
     *
     * @param data The NVCVImageBatchData to use for initialization.
     */
    explicit ImageBatchVarShapeData(const NVCVImageBatchData &data);

    /**
     * @brief Get the list of image formats.
     *
     * @return A pointer to the list of NVCVImageFormat.
     */
    const NVCVImageFormat *formatList() const;

    /**
     * @brief Get the host-side list of image formats.
     *
     * @return A pointer to the list of NVCVImageFormat on the host.
     */
    const NVCVImageFormat *hostFormatList() const;

    /**
     * @brief Get the maximum size across all images in the batch.
     *
     * @return The maximum size as Size2D.
     */
    Size2D maxSize() const;

    /**
     * @brief Get the unique format of images in the batch, if all images have the same format.
     *
     * @return The unique ImageFormat, or undefined if formats differ.
     */
    ImageFormat uniqueFormat() const;

    /**
     * @brief Check if a specific buffer type is compatible with this class.
     *
     * @param type The buffer type to check.
     * @return True if the type matches the expected type for variable shaped image batches, false otherwise.
     */
    static constexpr bool IsCompatibleKind(NVCVImageBatchBufferType type)
    {
        return type == NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA;
    }
};

/**
 * @class ImageBatchVarShapeDataStrided
 * @brief Represents the strided data of a variable shaped image batch.
 *
 * This class extends ImageBatchVarShapeData to provide access to strided data.
 */
class ImageBatchVarShapeDataStrided : public ImageBatchVarShapeData
{
public:
    using ImageBatchVarShapeData::ImageBatchVarShapeData;

    /**
     * @brief Construct from an existing NVCVImageBatchData.
     *
     * @param data The NVCVImageBatchData to use for initialization.
     */
    explicit ImageBatchVarShapeDataStrided(const NVCVImageBatchData &data);

    /**
     * @brief Get the list of strided image buffers.
     *
     * @return A pointer to the list of NVCVImageBufferStrided.
     */
    const NVCVImageBufferStrided *imageList() const;
};

/**
 * @class ImageBatchVarShapeDataStridedCuda
 * @brief Represents the strided data of a variable shaped image batch in CUDA.
 *
 * This class extends ImageBatchVarShapeDataStrided to provide access to CUDA-specific strided data.
 */
class ImageBatchVarShapeDataStridedCuda : public ImageBatchVarShapeDataStrided
{
public:
    using Buffer = NVCVImageBatchVarShapeBufferStrided;

    using ImageBatchVarShapeDataStrided::ImageBatchVarShapeDataStrided;

    /**
     * @brief Construct with image count and buffer.
     *
     * @param numImages The number of images.
     * @param buffer The buffer to use for initialization.
     */
    explicit ImageBatchVarShapeDataStridedCuda(int32_t numImages, const Buffer &buffer);

    /**
     * @brief Construct from an existing NVCVImageBatchData.
     *
     * @param data The NVCVImageBatchData to use for initialization.
     */
    explicit ImageBatchVarShapeDataStridedCuda(const NVCVImageBatchData &data);
};

} // namespace nvcv

#include "detail/ImageBatchDataImpl.hpp"

#endif // NVCV_IMAGEBATCHDATA_HPP
