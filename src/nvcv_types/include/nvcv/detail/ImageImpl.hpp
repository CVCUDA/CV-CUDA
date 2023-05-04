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

inline Image::Image(const Requirements &reqs, const Allocator &alloc)
{
    detail::CheckThrow(nvcvImageConstruct(&reqs, alloc.handle(), &m_handle));
    detail::SetObjectAssociation(nvcvImageSetUserPointer, this, m_handle);
}

inline Image::Image(const Size2D &size, ImageFormat fmt, const Allocator &alloc, const MemAlignment &bufAlign)
    : Image(CalcRequirements(size, fmt, bufAlign), alloc)
{
}

inline Image::~Image()
{
    nvcvImageDecRef(m_handle, nullptr);
}

inline NVCVImageHandle Image::doGetHandle() const
{
    return m_handle;
}

// ImageWrapData implementation -------------------------------------

inline ImageWrapData::ImageWrapData(const ImageData &data, ImageDataCleanupCallback &&cleanup)
{
    detail::CheckThrow(
        nvcvImageWrapDataConstruct(&data.cdata(), cleanup.targetFunc(), cleanup.targetHandle(), &m_handle));
    cleanup.release(); // the handle now owns the callback
    try
    {
        detail::SetObjectAssociation(nvcvImageSetUserPointer, this, m_handle);
    }
    catch (...)
    {
        nvcvImageDecRef(m_handle, nullptr);
        throw;
    }
}

inline ImageWrapData::~ImageWrapData()
{
    nvcvImageDecRef(m_handle, nullptr);
}

inline NVCVImageHandle ImageWrapData::doGetHandle() const
{
    return m_handle;
}

} // namespace nvcv

#endif // NVCV_IMAGE_IMPL_HPP
