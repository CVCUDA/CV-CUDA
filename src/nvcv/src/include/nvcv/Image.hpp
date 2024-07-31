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

#ifndef NVCV_IMAGE_HPP
#define NVCV_IMAGE_HPP

#include "CoreResource.hpp"
#include "HandleWrapper.hpp"
#include "Image.h"
#include "ImageData.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"
#include "alloc/Allocator.hpp"
#include "detail/Callback.hpp"

#include <functional>

namespace nvcv {

NVCV_IMPL_SHARED_HANDLE(Image);

/**
 * @class Image
 * @brief Represents an image resource managed by NVCV.
 *
 * This class wraps the NVCVImageHandle and provides a high-level interface to
 * manage images, including their allocation, deallocation, and querying of
 * various properties like size and format.
 */
class Image : public CoreResource<NVCVImageHandle, Image>
{
public:
    using HandleType   = NVCVImageHandle;
    using Base         = CoreResource<NVCVImageHandle, Image>;
    using Requirements = NVCVImageRequirements;

    /**
     * @brief Calculate the requirements for image allocation.
     *
     * @param size The size of the image.
     * @param fmt The image format.
     * @param bufAlign The memory alignment (optional).
     * @return The requirements for the image.
     */
    static Requirements CalcRequirements(const Size2D &size, ImageFormat fmt, const MemAlignment &bufAlign = {});

    NVCV_IMPLEMENT_SHARED_RESOURCE(Image, Base);

    /**
     * @brief Construct an Image with specific requirements.
     *
     * @param reqs The requirements for the image.
     * @param alloc The allocator to use (optional).
     */
    explicit Image(const Requirements &reqs, const Allocator &alloc = nullptr);

    /**
     * @brief Construct an Image with specified size, format, and alignment.
     *
     * @param size The size of the image.
     * @param fmt The image format.
     * @param alloc The allocator to use (optional).
     * @param bufAlign The memory alignment (optional).
     */
    explicit Image(const Size2D &size, ImageFormat fmt, const Allocator &alloc = nullptr,
                   const MemAlignment &bufAlign = {});

    /**
     * @brief Get the size of the image.
     *
     * @return The size of the image.
     */
    Size2D size() const;

    /**
     * @brief Get the format of the image.
     *
     * @return The image format.
     */
    ImageFormat format() const;

    /**
     * @brief Export the underlying data of the image.
     *
     * @return The image data.
     */
    ImageData exportData() const;

    /**
     * @brief Export the underlying data of the image with a specific type.
     *
     * @tparam DATA The type to cast the data to.
     * @return The image data casted to the specified type.
     */
    template<typename DATA>
    Optional<DATA> exportData() const;

    /**
     * @brief Set a user-defined pointer to the image.
     *
     * @param ptr The pointer to set.
     */
    void setUserPointer(void *ptr);

    /**
     * @brief Retrieve the user-defined pointer associated with the image.
     *
     * @return The user pointer.
     */
    void *userPointer() const;
};

// ImageWrapData definition -------------------------------------
// Image that wraps an image data allocated outside NVCV

using ImageDataCleanupFunc = void(const ImageData &);

struct TranslateImageDataCleanup
{
    template<typename CppCleanup>
    void operator()(CppCleanup &&c, const NVCVImageData *data) const noexcept
    {
        c(ImageData(*data));
    }
};

using ImageDataCleanupCallback
    = CleanupCallback<ImageDataCleanupFunc, detail::RemovePointer_t<NVCVImageDataCleanupFunc>,
                      TranslateImageDataCleanup>;

// For API backward-compatibility
inline Image ImageWrapData(const ImageData &data, ImageDataCleanupCallback &&cleanup = ImageDataCleanupCallback{});

using ImageWrapHandle = NonOwningResource<Image>;

} // namespace nvcv

#include "detail/ImageImpl.hpp"

#endif // NVCV_IMAGE_HPP
