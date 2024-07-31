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

#ifndef NVCV_IMAGEBATCH_IMPL_HPP
#define NVCV_IMAGEBATCH_IMPL_HPP

#ifndef NVCV_IMAGEBATCH_IMPL_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// ImageBatch implementation ---------------------

inline int32_t ImageBatch::capacity() const
{
    int32_t out;
    detail::CheckThrow(nvcvImageBatchGetCapacity(this->handle(), &out));
    return out;
}

inline int32_t ImageBatch::numImages() const
{
    int32_t out;
    detail::CheckThrow(nvcvImageBatchGetNumImages(this->handle(), &out));
    return out;
}

inline NVCVTypeImageBatch ImageBatch::type() const
{
    NVCVTypeImageBatch type;
    detail::CheckThrow(nvcvImageBatchGetType(this->handle(), &type));
    return type;
}

inline ImageBatchData ImageBatch::exportData(CUstream stream) const
{
    // ImageBatches are mutable, we can't cache previously exported data.

    NVCVImageBatchData batchData;
    detail::CheckThrow(nvcvImageBatchExportData(this->handle(), stream, &batchData));

    if (batchData.bufferType != NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION,
                        "Image batch data cannot be exported, buffer type not supported");
    }

    return ImageBatchData(batchData);
}

template<typename DATA>
Optional<DATA> ImageBatch::exportData(CUstream stream) const
{
    return exportData(stream).cast<DATA>();
}

inline void ImageBatch::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvImageBatchSetUserPointer(this->handle(), ptr));
}

inline void *ImageBatch::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvImageBatchGetUserPointer(this->handle(), &ptr));
    return ptr;
}

// ImageBatchVarShape implementation ----------------------------------

namespace detail {

inline NVCVImageHandle GetImageHandleForPushBack(Image img)
{
    return img.release();
}

inline NVCVImageHandle GetImageHandleForPushBack(std::reference_wrapper<Image> img)
{
    return GetImageHandleForPushBack(img.get());
}

inline NVCVImageHandle GetImageHandleForPushBack(NVCVImageHandle imgHandle)
{
    return imgHandle;
}

} // namespace detail

inline ImageBatchVarShape::ImageBatchVarShape(NVCVImageBatchHandle &&handle)
    : ImageBatchVarShape(ImageBatch(std::move(handle))) // we take the ownership first, create a wrapper and cast
{
}

inline ImageBatchVarShape::ImageBatchVarShape(const ImageBatch &batch)
{
    *this = batch;
}

inline ImageBatchVarShape::ImageBatchVarShape(ImageBatch &&batch)
{
    *this = std::move(batch);
}

inline ImageBatchVarShape &ImageBatchVarShape::operator=(const ImageBatch &batch)
{
    if (*this && !IsCompatibleKind(batch.type()))
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "The handle doesn't point to a variable-shape image batch.");
    ImageBatch::operator=(batch);
    return *this;
}

inline ImageBatchVarShape &ImageBatchVarShape::operator=(ImageBatch &&batch)
{
    if (*this && !IsCompatibleKind(batch.type()))
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "The handle doesn't point to a variable-shape image batch.");
    ImageBatch::operator=(std::move(batch));
    return *this;
}

template<class IT>
void ImageBatchVarShape::pushBack(IT itBeg, IT itEnd)
{
    auto cb = [itBeg, &itEnd]() mutable
    {
        if (itBeg != itEnd)
        {
            return detail::GetImageHandleForPushBack(*(itBeg++));
        }
        else
        {
            return NVCVImageHandle{};
        }
    };

    pushBack(cb);
}

inline void ImageBatchVarShape::pushBack(const Image &img)
{
    NVCVImageHandle himg = img.handle();
    detail::CheckThrow(nvcvImageBatchVarShapePushImages(this->handle(), &himg, 1));
}

template<class F, class>
inline void ImageBatchVarShape::pushBack(F &&cb)
{
    auto *pcb = &cb;
    auto  ccb = [](void *ctx) -> NVCVImageHandle
    {
        return detail::GetImageHandleForPushBack((*decltype(pcb)(ctx))());
    };
    detail::CheckThrow(nvcvImageBatchVarShapePushImagesCallback(this->handle(), ccb, pcb));
}

inline void ImageBatchVarShape::popBack(int32_t imgCount)
{
    detail::CheckThrow(nvcvImageBatchVarShapePopImages(this->handle(), imgCount));
}

inline Image ImageBatchVarShape::operator[](ptrdiff_t n) const
{
    NVCVImageHandle himg;
    detail::CheckThrow(nvcvImageBatchVarShapeGetImages(this->handle(), n, &himg, 1));
    return Image(std::move(himg));
}

inline void ImageBatchVarShape::clear()
{
    detail::CheckThrow(nvcvImageBatchVarShapeClear(this->handle()));
}

inline Size2D ImageBatchVarShape::maxSize() const
{
    Size2D s;
    detail::CheckThrow(nvcvImageBatchVarShapeGetMaxSize(this->handle(), &s.w, &s.h));
    return s;
}

inline ImageFormat ImageBatchVarShape::uniqueFormat() const
{
    NVCVImageFormat out;
    detail::CheckThrow(nvcvImageBatchVarShapeGetUniqueFormat(this->handle(), &out));
    return ImageFormat{out};
}

inline auto ImageBatchVarShape::begin() const -> ConstIterator
{
    return ConstIterator(*this, 0);
}

inline auto ImageBatchVarShape::end() const -> ConstIterator
{
    return ConstIterator(*this, this->numImages());
}

inline auto ImageBatchVarShape::cbegin() const -> ConstIterator
{
    return this->begin();
}

inline auto ImageBatchVarShape::cend() const -> ConstIterator
{
    return this->end();
}

inline auto ImageBatchVarShape::CalcRequirements(int32_t capacity) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvImageBatchVarShapeCalcRequirements(capacity, &reqs));
    return reqs;
}

inline ImageBatchVarShape::ImageBatchVarShape(const Requirements &reqs, const Allocator &alloc)
{
    NVCVImageBatchHandle handle = nullptr;
    detail::CheckThrow(nvcvImageBatchVarShapeConstruct(&reqs, alloc.handle(), &handle));
    reset(std::move(handle));
}

inline ImageBatchVarShape::ImageBatchVarShape(int32_t capacity, const Allocator &alloc)
    : ImageBatchVarShape(CalcRequirements(capacity), alloc)
{
}

// ImageBatchVarShape::Iterator implementation ------------------------

inline ImageBatchVarShape::Iterator::Iterator(const ImageBatchVarShape &batch, int32_t idxImage)
    : m_batch(&batch)
    , m_curIndex(idxImage)
{
}

inline void ImageBatchVarShape::Iterator::updateCurrentItem() const
{
    if (m_currentImage)
        return;

    if (m_batch == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Iterator doesn't point to an image batch object");
    }
    if (m_curIndex >= m_batch->numImages())
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Iterator points to an invalid image in the image batch");
    }
    m_currentImage = (*m_batch)[m_curIndex];
}

inline void ImageBatchVarShape::Iterator::invalidateCurrentItem()
{
    m_currentImage.reset();
}

inline auto ImageBatchVarShape::Iterator::operator*() const -> reference
{
    updateCurrentItem();
    return m_currentImage;
}

inline auto ImageBatchVarShape::Iterator::operator->() const -> pointer
{
    updateCurrentItem();
    return &m_currentImage;
}

inline auto ImageBatchVarShape::Iterator::operator++(int) -> Iterator
{
    Iterator cur(*this);
    ++(*this);
    return cur;
}

inline auto ImageBatchVarShape::Iterator::operator++() -> Iterator &
{
    ++m_curIndex;
    invalidateCurrentItem();
    return *this;
}

inline auto ImageBatchVarShape::Iterator::operator--(int) -> Iterator
{
    Iterator cur(*this);
    --(*this);
    return cur;
}

inline auto ImageBatchVarShape::Iterator::operator--() -> Iterator &
{
    --m_curIndex;
    invalidateCurrentItem();
    return *this;
}

inline auto ImageBatchVarShape::Iterator::operator+(difference_type diff) const -> Iterator
{
    return {*m_batch, m_curIndex + diff};
}

inline auto ImageBatchVarShape::Iterator::operator-(difference_type diff) const -> Iterator
{
    return {*m_batch, m_curIndex - diff};
}

inline ImageBatchVarShape::Iterator operator+(ImageBatchVarShape::Iterator::difference_type diff,
                                              const ImageBatchVarShape::Iterator           &it)
{
    return it + diff;
}

inline bool ImageBatchVarShape::Iterator::operator==(const Iterator &that) const
{
    if (m_batch == nullptr && that.m_batch == nullptr)
    {
        return true;
    }
    else if (m_batch == that.m_batch)
    {
        return m_curIndex == that.m_curIndex;
    }
    else
    {
        return false;
    }
}

inline bool ImageBatchVarShape::Iterator::operator!=(const Iterator &that) const
{
    return !(*this == that);
}

inline bool ImageBatchVarShape::Iterator::operator<(const Iterator &that) const
{
    return std::make_pair(m_batch, m_curIndex) < std::make_pair(that.m_batch, that.m_curIndex);
}

inline bool ImageBatchVarShape::Iterator::operator>(const Iterator &that) const
{
    return that < *this;
}

inline bool ImageBatchVarShape::Iterator::operator<=(const Iterator &that) const
{
    return !(that < *this);
}

inline bool ImageBatchVarShape::Iterator::operator>=(const Iterator &that) const
{
    return !(*this < that);
}

inline auto ImageBatchVarShape::Iterator::operator-(const Iterator &that) const -> difference_type
{
    if (m_batch != that.m_batch)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION,
                        "Cannot calculate a difference between iterators from different batch objects");
    }

    return m_curIndex - that.m_curIndex;
}

} // namespace nvcv

#endif // NVCV_IMAGEBATCH_IMPL_HPP
