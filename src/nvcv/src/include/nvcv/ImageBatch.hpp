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

#ifndef NVCV_IMAGEBATCH_HPP
#define NVCV_IMAGEBATCH_HPP

#include "CoreResource.hpp"
#include "Image.hpp"
#include "ImageBatch.h"
#include "ImageBatchData.hpp"
#include "Optional.hpp"

namespace nvcv {

NVCV_IMPL_SHARED_HANDLE(ImageBatch);

/**
 * @class ImageBatch
 * @brief Represents a batch of images managed by NVCV.
 *
 * This class wraps the NVCVImageBatchHandle and provides a high-level interface to
 * manage batches of images, including querying of various properties like capacity
 * and number of images.
 */
class ImageBatch : public CoreResource<NVCVImageBatchHandle, ImageBatch>
{
public:
    using Base = CoreResource<NVCVImageBatchHandle, ImageBatch>;

    NVCV_IMPLEMENT_SHARED_RESOURCE(ImageBatch, Base);

    using HandleType = NVCVImageBatchHandle;

    /**
     * @brief Get the capacity of the image batch, i.e., the maximum number of images it can hold.
     *
     * @return The capacity of the image batch.
     */
    int32_t capacity() const;

    /**
     * @brief Get the current number of images in the image batch.
     *
     * @return The number of images.
     */
    int32_t numImages() const;

    /**
     * @brief Get the type of the image batch.
     *
     * @return The type of the image batch.
     */
    NVCVTypeImageBatch type() const;

    /**
     * @brief Export the underlying data of the image batch.
     *
     * @param stream The CUDA stream.
     * @return The image batch data.
     */
    ImageBatchData exportData(CUstream stream) const;

    /**
     * @brief Export the underlying data of the image batch with a specific type.
     *
     * @tparam Data The type to cast the data to.
     * @param stream The CUDA stream.
     * @return The image batch data casted to the specified type.
     */
    template<typename Data>
    Optional<Data> exportData(CUstream stream) const;

    /**
     * @brief Set a user-defined pointer to the image batch.
     *
     * @param ptr The pointer to set.
     */
    void setUserPointer(void *ptr);

    /**
     * @brief Retrieve the user-defined pointer associated with the image batch.
     *
     * @return The user pointer.
     */
    void *userPointer() const;

    /**
     * @brief Check if a kind is compatible with the ImageBatch class.
     *
     * @return True since all kinds are currently supported.
     */
    static bool IsCompatibleKind(NVCVTypeImageBatch)
    {
        return true;
    }
};

struct TranslateImageToHandle
{
    template<typename CppGetImage>
    NVCVImageHandle operator()(CppGetImage &&c) const noexcept
    {
        return c().handle();
    }
};

using ImageDataCleanupCallback
    = CleanupCallback<ImageDataCleanupFunc, detail::RemovePointer_t<NVCVImageDataCleanupFunc>,
                      TranslateImageDataCleanup>;

/**
 * @class ImageBatchVarShape
 * @brief Represents a batch of images with variable shapes.
 *
 * Extends the functionality provided by the `ImageBatch` class to support batches
 * of images where each image might have a different shape or size.
 */
class ImageBatchVarShape : public ImageBatch
{
public:
    using Requirements = NVCVImageBatchVarShapeRequirements;

    /**
     * @brief Calculate requirements for a variable-shaped image batch with a specific capacity.
     *
     * @param capacity The capacity for which requirements need to be calculated.
     * @return The requirements for creating a batch with the specified capacity.
     */
    static Requirements CalcRequirements(int32_t capacity);

    NVCV_IMPLEMENT_SHARED_RESOURCE(ImageBatchVarShape, ImageBatch);

    explicit ImageBatchVarShape(NVCVImageBatchHandle &&handle); ///< Construct from an existing NVCV handle.
    ImageBatchVarShape(const ImageBatch &batch);                ///< Construct from an existing `ImageBatch`.
    ImageBatchVarShape(ImageBatch &&batch);                     ///< Move construct from an existing `ImageBatch`.
    explicit ImageBatchVarShape(const Requirements &reqs,
                                const Allocator    &alloc = nullptr); ///< Construct with specific requirements.
    explicit ImageBatchVarShape(int32_t          capacity,
                                const Allocator &alloc = nullptr); ///< Construct with a specified capacity.

    ImageBatchVarShape &operator=(const ImageBatch &batch);
    ImageBatchVarShape &operator=(ImageBatch &&batch);

    template<class IT>
    void pushBack(IT itBeg, IT itend);
    void pushBack(const Image &img);
    void popBack(int32_t imgCount = 1);

    /**
     *
     * @brief Adds images produced by the callback.
     *
     * This function appends images produced by the callback until it returns an image with a null handle.
     *
     * @param cb    A callable objects that takes no parameters and returns nvcv::Image, reference to nvcv::Image,
     *              an object convertible to nvcv::Image.
     *
     * @note When the callback returns NVCVImageHandle, it is assumed that the callback transfers the ownership of
     *       the handle reference - the reference count will not be incremented when the object is added to the batch.
     */
    template<class F, class = decltype(std::declval<F>()())>
    void pushBack(F &&cb);

    /**
     * @brief Clear all images from the batch.
     */
    void clear();

    /**
     * @brief Get the maximum size among all images in the batch.
     *
     * @return The maximum size.
     */
    Size2D maxSize() const;

    /**
     * @brief Determine the unique format of the images in the batch if applicable.
     *
     * @return The unique format, or an invalid format if images have different formats.
     */
    ImageFormat uniqueFormat() const;

    /**
     * @brief Access a specific image in the batch.
     *
     * @param n The index of the image.
     * @return The image at the specified index.
     */
    Image operator[](ptrdiff_t n) const;

    class Iterator;

    using ConstIterator = Iterator;

    ConstIterator begin() const;
    ConstIterator end() const;

    ConstIterator cbegin() const;
    ConstIterator cend() const;

    using ImageBatch::exportData;

    /**
     * @brief Export the underlying data of the image batch.
     *
     * @param stream The CUDA stream.
     * @return The image batch data.
     */
    ImageBatchVarShapeData exportData(CUstream stream) const
    {
        return *ImageBatch::template exportData<ImageBatchVarShapeData>(stream);
    }

    /**
     * @brief Check if a specific kind is compatible with the ImageBatchVarShape class.
     *
     * @param kind The kind to check.
     * @return True if the kind is NVCV_TYPE_IMAGEBATCH_VARSHAPE, false otherwise.
     */
    static bool IsCompatibleKind(NVCVTypeImageBatch kind)
    {
        return kind == NVCV_TYPE_IMAGEBATCH_VARSHAPE;
    }
};

// For API backward-compatibility
using ImageBatchWrapHandle         = NonOwningResource<ImageBatch>;
using ImageBatchVarShapeWrapHandle = NonOwningResource<ImageBatchVarShape>;

class ImageBatchVarShape::Iterator
{
public:
    using value_type        = Image;
    using reference         = const Image &;
    using pointer           = const Image *;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = int32_t;

    Iterator() = default;

    reference operator*() const;
    pointer   operator->() const;

    Iterator  operator++(int);
    Iterator &operator++();
    Iterator  operator--(int);
    Iterator &operator--();

    Iterator operator+(difference_type diff) const;
    Iterator operator-(difference_type diff) const;

    difference_type operator-(const Iterator &that) const;

    bool operator==(const Iterator &that) const;
    bool operator!=(const Iterator &that) const;
    bool operator<(const Iterator &that) const;
    bool operator>(const Iterator &that) const;
    bool operator<=(const Iterator &that) const;
    bool operator>=(const Iterator &that) const;

private:
    const ImageBatchVarShape *m_batch        = nullptr;
    int                       m_curIndex     = 0;
    mutable Image             m_currentImage = {};

    void updateCurrentItem() const;
    void invalidateCurrentItem();

    friend class ImageBatchVarShape;
    Iterator(const ImageBatchVarShape &batch, int32_t idxImage);

    friend Iterator operator+(difference_type diff, const Iterator &it);
};

// Image Batch const ref optional definition ---------------------------

using OptionalImageBatchVarShapeConstRef = nvcv::Optional<std::reference_wrapper<const nvcv::ImageBatchVarShape>>;

#define NVCV_IMAGE_BATCH_VAR_SHAPE_HANDLE_TO_OPTIONAL(X) \
    X ? nvcv::OptionalImageBatchVarShapeConstRef(nvcv::ImageBatchVarShapeWrapHandle{X}) : nvcv::NullOpt

} // namespace nvcv

#include "detail/ImageBatchImpl.hpp"

#endif // NVCV_IMAGEBATCH_HPP
