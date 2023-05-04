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

#ifndef NVCV_IIMAGE_IMPL_HPP
#define NVCV_IIMAGE_IMPL_HPP

#ifndef NVCV_IIMAGE_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

inline IImage::IImage() {}

inline IImage::~IImage() {}

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

inline ImageData IImage::exportData() const
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
inline Optional<DATA> IImage::exportData() const
{
    return exportData().cast<DATA>();
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
