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

#ifndef NVCV_IMAGEBATCH_HPP
#define NVCV_IMAGEBATCH_HPP

#include "CoreResource.hpp"
#include "Image.hpp"
#include "ImageBatch.h"
#include "ImageBatchData.hpp"

namespace nvcv {

NVCV_IMPL_SHARED_HANDLE(ImageBatch);

class ImageBatch : public CoreResource<NVCVImageBatchHandle, ImageBatch>
{
public:
    using Base = CoreResource<NVCVImageBatchHandle, ImageBatch>;

    NVCV_IMPLEMENT_SHARED_RESOURCE(ImageBatch, Base);

    using HandleType = NVCVImageBatchHandle;

    int32_t capacity() const;
    int32_t numImages() const;

    NVCVTypeImageBatch type() const;

    ImageBatchData exportData(CUstream stream) const;

    template<typename Data>
    Optional<Data> exportData(CUstream stream) const;

    void  setUserPointer(void *ptr);
    void *userPointer() const;

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

class ImageBatchVarShape : public ImageBatch
{
public:
    using Requirements = NVCVImageBatchVarShapeRequirements;
    static Requirements CalcRequirements(int32_t capacity);

    NVCV_IMPLEMENT_SHARED_RESOURCE(ImageBatchVarShape, ImageBatch);

    explicit ImageBatchVarShape(NVCVImageBatchHandle &&handle);
    ImageBatchVarShape(const ImageBatch &batch);
    ImageBatchVarShape(ImageBatch &&batch);
    explicit ImageBatchVarShape(const Requirements &reqs, const Allocator &alloc = nullptr);
    explicit ImageBatchVarShape(int32_t capacity, const Allocator &alloc = nullptr);

    ImageBatchVarShape &operator=(const ImageBatch &batch);
    ImageBatchVarShape &operator=(ImageBatch &&batch);

    template<class IT>
    void pushBack(IT itBeg, IT itend);
    void pushBack(const Image &img);
    void popBack(int32_t imgCount = 1);

    /** Adds images produced by the callback.
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

    void clear();

    Size2D      maxSize() const;
    ImageFormat uniqueFormat() const;

    Image operator[](ptrdiff_t n) const;

    class Iterator;

    using ConstIterator = Iterator;

    ConstIterator begin() const;
    ConstIterator end() const;

    ConstIterator cbegin() const;
    ConstIterator cend() const;

    using ImageBatch::exportData;

    ImageBatchVarShapeData exportData(CUstream stream) const
    {
        return *ImageBatch::template exportData<ImageBatchVarShapeData>(stream);
    }

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

} // namespace nvcv

#include "detail/ImageBatchImpl.hpp"

#endif // NVCV_IMAGEBATCH_HPP
