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

#ifndef NVCV_IMAGE_IMPL_HPP
#define NVCV_IMAGE_IMPL_HPP

#ifndef NVCV_IMAGE_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// Image implementation -------------------------------------

inline Size2D Image::size() const
{
    Size2D out;
    detail::CheckThrow(nvcvImageGetSize(this->handle(), &out.w, &out.h));
    return out;
}

inline ImageFormat Image::format() const
{
    NVCVImageFormat out;
    detail::CheckThrow(nvcvImageGetFormat(this->handle(), &out));
    return ImageFormat{out};
}

inline ImageData Image::exportData() const
{
    if (auto handle = this->handle())
    {
        ImageData data;
        detail::CheckThrow(nvcvImageExportData(handle, &data.cdata()));
        return data;
    }
    else
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Cannot export data from a NULL handle.");
    }
}

template<typename DATA>
inline Optional<DATA> Image::exportData() const
{
    return exportData().cast<DATA>();
}

inline void Image::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvImageSetUserPointer(this->handle(), ptr));
}

inline void *Image::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvImageGetUserPointer(this->handle(), &ptr));
    return ptr;
}

inline auto Image::CalcRequirements(const Size2D &size, ImageFormat fmt, const MemAlignment &bufAlign) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvImageCalcRequirements(size.w, size.h, fmt, bufAlign.baseAddr(), bufAlign.rowAddr(), &reqs));
    return reqs;
}

inline Image::Image(const Requirements &reqs, const Allocator &alloc)
{
    NVCVImageHandle handle;
    detail::CheckThrow(nvcvImageConstruct(&reqs, alloc.handle(), &handle));
    reset(std::move(handle));
}

inline Image::Image(const Size2D &size, ImageFormat fmt, const Allocator &alloc, const MemAlignment &bufAlign)
    : Image(CalcRequirements(size, fmt, bufAlign), alloc)
{
}

// ImageWrapData implementation -------------------------------------

inline Image ImageWrapData(const ImageData &data, ImageDataCleanupCallback &&cleanup)
{
    NVCVImageHandle handle = nullptr;
    detail::CheckThrow(
        nvcvImageWrapDataConstruct(&data.cdata(), cleanup.targetFunc(), cleanup.targetHandle(), &handle));
    (void)cleanup.release(); // The cleanup callback is now owned by the image object.
    return Image(std::move(handle));
}

} // namespace nvcv

#endif // NVCV_IMAGE_IMPL_HPP
