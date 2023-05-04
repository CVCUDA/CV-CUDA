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

#ifndef NVCV_IMAGE_HPP
#define NVCV_IMAGE_HPP

#include "IImage.hpp"
#include "ImageData.hpp"
#include "Size.hpp"
#include "alloc/Allocator.hpp"
#include "detail/Callback.hpp"

#include <nvcv/ImageFormat.hpp>

#include <functional>

namespace nvcv {

// Image definition -------------------------------------
// Image allocated by cv-cuda
class Image : public IImage
{
public:
    using Requirements = NVCVImageRequirements;
    static Requirements CalcRequirements(const Size2D &size, ImageFormat fmt, const MemAlignment &bufAlign = {});

    explicit Image(const Requirements &reqs, const Allocator &alloc = nullptr);
    explicit Image(const Size2D &size, ImageFormat fmt, const Allocator &alloc = nullptr,
                   const MemAlignment &bufAlign = {});
    ~Image();

    Image(const Image &) = delete;

private:
    NVCVImageHandle doGetHandle() const final override;

    NVCVImageHandle m_handle;
};

// ImageWrapData definition -------------------------------------
// Image that wraps an image data allocated outside cv-cuda

using ImageDataCleanupFunc = void(const ImageData &);

struct TranslateImageDataCleanup
{
    template<typename CppCleanup>
    void operator()(CppCleanup &&c, const NVCVImageData *data) const noexcept
    {
        c(ImageData(*data));
    }
};

using ImageDataCleanupCallback
    = CleanupCallback<ImageDataCleanupFunc, detail::RemovePointer_t<NVCVImageDataCleanupFunc>,
                      TranslateImageDataCleanup>;

class ImageWrapData : public IImage
{
public:
    explicit ImageWrapData(const ImageData &data, ImageDataCleanupCallback &&cleanup = ImageDataCleanupCallback{});
    ~ImageWrapData();

    ImageWrapData(const Image &) = delete;

private:
    NVCVImageHandle doGetHandle() const final override;

    NVCVImageHandle m_handle;
};

// For API backward-compatibility
using ImageWrapHandle = detail::WrapHandle<IImage>;

} // namespace nvcv

#include "detail/ImageImpl.hpp"

#endif // NVCV_IMAGE_HPP
