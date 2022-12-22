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

#ifndef NVCV_IMAGE_IMPL_HPP
#define NVCV_IMAGE_IMPL_HPP

#ifndef NVCV_IMAGE_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// Image implementation -------------------------------------

inline auto Image::CalcRequirements(const Size2D &size, ImageFormat fmt, const MemAlignment &bufAlign) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvImageCalcRequirements(size.w, size.h, fmt, bufAlign.baseAddr(), bufAlign.rowAddr(), &reqs));
    return reqs;
}

inline Image::Image(const Requirements &reqs, IAllocator *alloc)
{
    detail::CheckThrow(nvcvImageConstruct(&reqs, alloc ? alloc->handle() : nullptr, &m_handle));
    detail::SetObjectAssociation(nvcvImageSetUserPointer, this, m_handle);
}

inline Image::Image(const Size2D &size, ImageFormat fmt, IAllocator *alloc, const MemAlignment &bufAlign)
    : Image(CalcRequirements(size, fmt, bufAlign), alloc)
{
}

inline Image::~Image()
{
    nvcvImageDestroy(m_handle);
}

inline NVCVImageHandle Image::doGetHandle() const
{
    return m_handle;
}

// ImageWrapData implementation -------------------------------------

inline ImageWrapData::ImageWrapData(const IImageData &data, std::function<ImageDataCleanupFunc> cleanup)
    : m_cleanup(std::move(cleanup))
{
    detail::CheckThrow(nvcvImageWrapDataConstruct(&data.cdata(), m_cleanup ? &doCleanup : nullptr, this, &m_handle));
    detail::SetObjectAssociation(nvcvImageSetUserPointer, this, m_handle);
}

inline ImageWrapData::~ImageWrapData()
{
    nvcvImageDestroy(m_handle);
}

inline NVCVImageHandle ImageWrapData::doGetHandle() const
{
    return m_handle;
}

inline void ImageWrapData::doCleanup(void *ctx, const NVCVImageData *data)
{
    assert(data != nullptr);

    auto *this_ = reinterpret_cast<ImageWrapData *>(ctx);
    assert(this_ != nullptr);

    // exportData refers to 'data'
    const IImageData *imgData = this_->exportData();
    assert(imgData != nullptr);

    assert(this_->m_cleanup != nullptr);
    this_->m_cleanup(*imgData);
}

} // namespace nvcv

#endif // NVCV_IMAGE_IMPL_HPP
