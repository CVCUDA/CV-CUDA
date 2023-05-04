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

#ifndef NVCV_IMAGEDATA_HPP
#define NVCV_IMAGEDATA_HPP

#include "ImageData.h"
#include "ImageFormat.hpp"
#include "Optional.hpp"
#include "Size.hpp"

namespace nvcv {

// Interface hierarchy of image contents
class ImageData
{
public:
    ImageData() = default;
    ImageData(const NVCVImageData &data);

    ImageFormat format() const;

    NVCVImageData       &cdata();
    const NVCVImageData &cdata() const;

    template<typename Derived>
    Optional<Derived> cast() const;

private:
    NVCVImageData m_data{};
};

class ImageDataCudaArray : public ImageData
{
public:
    using Buffer = NVCVImageBufferCudaArray;

    explicit ImageDataCudaArray(ImageFormat format, const Buffer &buffer);
    explicit ImageDataCudaArray(const NVCVImageData &data);

    int         numPlanes() const;
    cudaArray_t plane(int p) const;

    static constexpr bool IsCompatibleKind(NVCVImageBufferType kind)
    {
        return kind == NVCV_IMAGE_BUFFER_CUDA_ARRAY;
    }
};

using ImageBufferStrided = NVCVImageBufferStrided;
using ImagePlaneStrided  = NVCVImagePlaneStrided;

class ImageDataStrided : public ImageData
{
public:
    explicit ImageDataStrided(const NVCVImageData &data);

    using Buffer = ImageBufferStrided;
    Size2D size() const;

    int                      numPlanes() const;
    const ImagePlaneStrided &plane(int p) const;

    static constexpr bool IsCompatibleKind(NVCVImageBufferType kind)
    {
        return kind == NVCV_IMAGE_BUFFER_STRIDED_CUDA || kind == NVCV_IMAGE_BUFFER_STRIDED_HOST;
    }

protected:
    ImageDataStrided() = default;
};

class ImageDataStridedCuda : public ImageDataStrided
{
public:
    explicit ImageDataStridedCuda(ImageFormat format, const Buffer &buffer);
    explicit ImageDataStridedCuda(const NVCVImageData &data);

    static constexpr bool IsCompatibleKind(NVCVImageBufferType kind)
    {
        return kind == NVCV_IMAGE_BUFFER_STRIDED_CUDA;
    }
};

class ImageDataStridedHost : public ImageDataStrided
{
public:
    explicit ImageDataStridedHost(ImageFormat format, const Buffer &buffer);
    explicit ImageDataStridedHost(const NVCVImageData &data);

    static constexpr bool IsCompatibleKind(NVCVImageBufferType kind)
    {
        return kind == NVCV_IMAGE_BUFFER_STRIDED_HOST;
    }
};

} // namespace nvcv

#include "detail/ImageDataImpl.hpp"

#endif // NVCV_DETAIL_IMAGEDATA_HPP
