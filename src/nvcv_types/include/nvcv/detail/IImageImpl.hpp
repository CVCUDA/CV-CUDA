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

#ifndef NVCV_IIMAGE_IMPL_HPP
#define NVCV_IIMAGE_IMPL_HPP

#ifndef NVCV_IIMAGE_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

inline IImage::IImage()
    : m_cacheDataPtr(nullptr)
{
}

inline IImage::~IImage()
{
    if (m_cacheDataPtr != nullptr)
    {
        m_cacheDataPtr->~IImageData();
    }
}

inline NVCVImageHandle IImage::handle() const
{
    NVCVImageHandle h = doGetHandle();
    assert(h != nullptr && "Post-condition failed");
    return h;
}

inline IImage *IImage::cast(HandleType h)
{
    return detail::CastImpl<IImage>(&nvcvImageGetUserPointer, &nvcvImageSetUserPointer, h);
}

inline Size2D IImage::size() const
{
    Size2D out;
    detail::CheckThrow(nvcvImageGetSize(this->handle(), &out.w, &out.h));
    return out;
}

inline ImageFormat IImage::format() const
{
    NVCVImageFormat out;
    detail::CheckThrow(nvcvImageGetFormat(this->handle(), &out));
    return ImageFormat{out};
}

inline const IImageData *IImage::exportData() const
{
    NVCVImageData imgData;
    detail::CheckThrow(nvcvImageExportData(this->handle(), &imgData));

    if (m_cacheDataPtr != nullptr)
    {
        m_cacheDataPtr->~IImageData();
        m_cacheDataPtr = nullptr;
    }

    switch (imgData.bufferType)
    {
    case NVCV_IMAGE_BUFFER_STRIDED_HOST:
    case NVCV_IMAGE_BUFFER_NONE:
        break; // will return nullptr as per current semantics

    case NVCV_IMAGE_BUFFER_STRIDED_CUDA:
        m_cacheDataPtr
            = ::new (&m_cacheDataArena) ImageDataStridedCuda(ImageFormat{imgData.format}, imgData.buffer.strided);
        break;

    case NVCV_IMAGE_BUFFER_CUDA_ARRAY:
        m_cacheDataPtr
            = ::new (&m_cacheDataArena) ImageDataCudaArray(ImageFormat{imgData.format}, imgData.buffer.cudaarray);
        break;
    }

    return m_cacheDataPtr;
}

inline void IImage::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvImageSetUserPointer(this->handle(), ptr));
}

inline void *IImage::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvImageGetUserPointer(this->handle(), &ptr));
    return ptr;
}

} // namespace nvcv

#endif // NVCV_IIMAGE_IMPL_HPP
