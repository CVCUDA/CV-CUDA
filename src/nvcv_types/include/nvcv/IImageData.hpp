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

#ifndef NVCV_IIMAGEDATA_HPP
#define NVCV_IIMAGEDATA_HPP

#include "ImageData.h"
#include "Size.hpp"

#include <nvcv/ImageFormat.hpp>

namespace nvcv {

// Interface hierarchy of image contents
class IImageData
{
public:
    virtual ~IImageData() = 0;

    ImageFormat format() const;

    const NVCVImageData &cdata() const;

protected:
    IImageData() = default;
    IImageData(const NVCVImageData &data);

    NVCVImageData &cdata();

private:
    NVCVImageData m_data;
};

class IImageDataCudaArray : public IImageData
{
public:
    virtual ~IImageDataCudaArray() = 0;

    int         numPlanes() const;
    cudaArray_t plane(int p) const;

protected:
    using IImageData::IImageData;
};

using ImagePlaneStrided = NVCVImagePlaneStrided;

class IImageDataStrided : public IImageData
{
public:
    virtual ~IImageDataStrided() = 0;

    Size2D size() const;

    int                      numPlanes() const;
    const ImagePlaneStrided &plane(int p) const;

protected:
    using IImageData::IImageData;
};

class IImageDataStridedCuda : public IImageDataStrided
{
public:
    virtual ~IImageDataStridedCuda() = 0;

protected:
    using IImageDataStrided::IImageDataStrided;
};

class IImageDataStridedHost : public IImageDataStrided
{
public:
    virtual ~IImageDataStridedHost() = 0;

protected:
    using IImageDataStrided::IImageDataStrided;
};

} // namespace nvcv

#include "detail/IImageDataImpl.hpp"

#endif // NVCV_IIMAGEDATA_HPP
