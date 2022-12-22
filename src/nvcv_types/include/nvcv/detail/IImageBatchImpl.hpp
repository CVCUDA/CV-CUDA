/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_IIMAGEBATCH_IMPL_HPP
#define NVCV_IIMAGEBATCH_IMPL_HPP

#ifndef NVCV_IIMAGEBATCH_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// IImageBatch implementation ---------------------

inline NVCVImageBatchHandle IImageBatch::handle() const
{
    return doGetHandle();
}

inline int32_t IImageBatch::capacity() const
{
    int32_t out;
    detail::CheckThrow(nvcvImageBatchGetCapacity(this->handle(), &out));
    return out;
}

inline int32_t IImageBatch::numImages() const
{
    int32_t out;
    detail::CheckThrow(nvcvImageBatchGetNumImages(this->handle(), &out));
    return out;
}

inline const IImageBatchData *IImageBatch::exportData(CUstream stream) const
{
    // ImageBatches are mutable, we can't cache previously exported data.

    NVCVImageBatchData batchData;
    detail::CheckThrow(nvcvImageBatchExportData(this->handle(), stream, &batchData));

    if (batchData.bufferType != NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION,
                        "Image batch data cannot be exported, buffer type not supported");
    }

    m_cacheData.emplace(batchData.numImages, batchData.buffer.varShapeStrided);

    return &*m_cacheData;
}

inline void IImageBatch::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvImageBatchSetUserPointer(this->handle(), ptr));
}

inline void *IImageBatch::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvImageBatchGetUserPointer(this->handle(), &ptr));
    return ptr;
}

inline IImageBatch *IImageBatch::cast(HandleType h)
{
    if (h != nullptr)
    {
        // Must get the concrete type to cast to the proper interface.
        NVCVTypeImageBatch type;
        detail::CheckThrow(nvcvImageBatchGetType(h, &type));
        switch (type)
        {
        case NVCV_TYPE_IMAGEBATCH_VARSHAPE:
            return detail::CastImpl<IImageBatchVarShape>(&nvcvImageBatchGetUserPointer, &nvcvImageBatchSetUserPointer,
                                                         h);
        default:
            return nullptr;
        }
    }
    else
    {
        return nullptr;
    }
}

// IImageBatchVarShape implementation ----------------------------------

inline const IImageBatchVarShapeData *IImageBatchVarShape::exportData(CUstream stream) const
{
    return static_cast<const IImageBatchVarShapeData *>(IImageBatch::exportData(stream));
}

template<class IT>
void IImageBatchVarShape::pushBack(IT itBeg, IT itEnd)
{
    auto cb = [it = itBeg, &itEnd]() mutable -> auto *
    {
        if (it != itEnd)
        {
            return &*it++;
        }
        else
        {
            return static_cast<decltype(&*it)>(nullptr);
        }
    };

    pushBack(cb);
}

namespace detail {

// For any pointer-like type
template<class T, class = typename std::enable_if<std::is_same<
                      NVCVImageHandle, typename std::decay<decltype(std::declval<T>()->handle())>::type>::value>::type>
NVCVImageHandle GetImageHandle(const T &ptr)
{
    if (ptr == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image must not be NULL");
    }
    return ptr->handle();
}

inline NVCVImageHandle GetImageHandle(const IImage &img)
{
    return img.handle();
}

inline NVCVImageHandle GetImageHandle(NVCVImageHandle h)
{
    if (h == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image handle must not be NULL");
    }
    return h;
}

template<class T>
NVCVImageHandle GetImageHandle(const std::reference_wrapper<T> &h)
{
    return h.get().handle();
}

template<class T>
inline NVCVImageHandle GetImageHandle(const T *ptr)
{
    if (ptr == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image must not be NULL");
    }
    return GetImageHandle(*ptr);
}

} // namespace detail

template<class F, class>
void IImageBatchVarShape::pushBack(F &&cbGetImage)
{
    static auto cbpriv = [](void *ctx) -> NVCVImageHandle
    {
        assert(ctx != nullptr);
        F &getImage = *static_cast<typename std::decay<F>::type *>(ctx);

        if (const auto &img = getImage())
        {
            return detail::GetImageHandle(img);
        }
        else
        {
            return nullptr;
        }
    };

    detail::CheckThrow(nvcvImageBatchVarShapePushImagesCallback(this->handle(), cbpriv, &cbGetImage));
}

inline void IImageBatchVarShape::pushBack(const IImage &img)
{
    NVCVImageHandle himg = img.handle();
    detail::CheckThrow(nvcvImageBatchVarShapePushImages(this->handle(), &himg, 1));
}

inline void IImageBatchVarShape::popBack(int32_t imgCount)
{
    detail::CheckThrow(nvcvImageBatchVarShapePopImages(this->handle(), imgCount));
}

inline IImage &IImageBatchVarShape::operator[](ptrdiff_t n) const
{
    NVCVImageHandle himg;
    detail::CheckThrow(nvcvImageBatchVarShapeGetImages(this->handle(), n, &himg, 1));
    return StaticCast<IImage>(himg);
}

inline void IImageBatchVarShape::clear()
{
    detail::CheckThrow(nvcvImageBatchVarShapeClear(this->handle()));
}

inline Size2D IImageBatchVarShape::maxSize() const
{
    Size2D s;
    detail::CheckThrow(nvcvImageBatchVarShapeGetMaxSize(this->handle(), &s.w, &s.h));
    return s;
}

inline ImageFormat IImageBatchVarShape::uniqueFormat() const
{
    NVCVImageFormat out;
    detail::CheckThrow(nvcvImageBatchVarShapeGetUniqueFormat(this->handle(), &out));
    return ImageFormat{out};
}

inline auto IImageBatchVarShape::begin() const -> ConstIterator
{
    return ConstIterator(*this, 0);
}

inline auto IImageBatchVarShape::end() const -> ConstIterator
{
    return ConstIterator(*this, this->numImages());
}

inline auto IImageBatchVarShape::cbegin() const -> ConstIterator
{
    return this->begin();
}

inline auto IImageBatchVarShape::cend() const -> ConstIterator
{
    return this->end();
}

// IImageBatchVarShape::Iterator implementation ------------------------

inline IImageBatchVarShape::Iterator::Iterator(const IImageBatchVarShape &batch, int32_t idxImage)
    : m_batch(&batch)
    , m_curIndex(idxImage)
{
}

inline IImageBatchVarShape::Iterator::Iterator()
    : m_batch(nullptr)
    , m_curIndex(0)
{
}

inline IImageBatchVarShape::Iterator::Iterator(const Iterator &that)
    : m_batch(that.m_batch)
    , m_curIndex(that.m_curIndex)
{
}

inline auto IImageBatchVarShape::Iterator::operator=(const Iterator &that) -> Iterator &
{
    if (this != &that)
    {
        m_batch    = that.m_batch;
        m_curIndex = that.m_curIndex;
    }
    return *this;
}

inline auto IImageBatchVarShape::Iterator::operator*() const -> reference
{
    if (m_batch == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Iterator doesn't point to an image batch object");
    }
    if (m_curIndex >= m_batch->numImages())
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Iterator points to an invalid image in the image batch");
    }

    return (*m_batch)[m_curIndex];
}

inline auto IImageBatchVarShape::Iterator::operator->() const -> pointer
{
    return &*(*this);
}

inline auto IImageBatchVarShape::Iterator::operator++(int) -> Iterator
{
    Iterator cur(*this);
    ++(*this);
    return cur;
}

inline auto IImageBatchVarShape::Iterator::operator++() -> Iterator &
{
    ++m_curIndex;
    return *this;
}

inline bool IImageBatchVarShape::Iterator::operator==(const Iterator &that) const
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

inline bool IImageBatchVarShape::Iterator::operator!=(const Iterator &that) const
{
    return !(*this == that);
}

} // namespace nvcv

#endif // NVCV_IIMAGEBATCH_IMPL_HPP
