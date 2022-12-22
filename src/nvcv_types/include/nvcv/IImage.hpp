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

#ifndef NVCV_IIMAGE_HPP
#define NVCV_IIMAGE_HPP

#include "Casts.hpp"
#include "IImageData.hpp"
#include "Image.h"
#include "ImageData.hpp"
#include "Size.hpp"

#include <nvcv/ImageFormat.hpp>

#include <type_traits> // for std::aligned_storage

namespace nvcv {

class IImage
{
public:
    using HandleType    = NVCVImageHandle;
    using BaseInterface = IImage;

    virtual ~IImage();

    HandleType     handle() const;
    static IImage *cast(HandleType h);

    Size2D      size() const;
    ImageFormat format() const;

    const IImageData *exportData() const;

    void  setUserPointer(void *ptr);
    void *userPointer() const;

protected:
    IImage();

private:
    virtual NVCVImageHandle doGetHandle() const = 0;

    // Where the concrete class for exported image data will be allocated
    // Should be an std::variant in C++17.
    union Arena
    {
        ImageDataCudaArray   cudaArray;
        ImageDataStridedCuda devStrided;
    };

    mutable std::aligned_storage<sizeof(Arena), alignof(Arena)>::type m_cacheDataArena;
    mutable IImageData                                               *m_cacheDataPtr;
};

} // namespace nvcv

#include "detail/IImageImpl.hpp"

#endif // NVCV_IIMAGE_HPP
