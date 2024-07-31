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

#ifndef NVCV_IMAGEDATA_IMPL_HPP
#define NVCV_IMAGEDATA_IMPL_HPP

#ifndef NVCV_IMAGEDATA_HPP
#    error "You must not include this header directly"
#endif

#include <stdexcept>

namespace nvcv {

// Implementation - ImageData ---------------------------------
inline ImageData::ImageData(const NVCVImageData &data)
    : m_data(data)
{
}

inline const NVCVImageData &ImageData::cdata() const
{
    return m_data;
}

inline NVCVImageData &ImageData::cdata()
{
    return m_data;
}

inline ImageFormat ImageData::format() const
{
    return ImageFormat{this->cdata().format};
}

template<typename Derived>
inline Optional<Derived> ImageData::cast() const
{
    static_assert(std::is_base_of<ImageData, Derived>::value, "Cannot cast ImageData to an unrelated type");

    static_assert(sizeof(Derived) == sizeof(NVCVImageData),
                  "The derived type is not a simple wrapper around NVCVImageData.");

    if (Derived::IsCompatibleKind(m_data.bufferType))
    {
        return Derived{m_data};
    }
    else
    {
        return NullOpt;
    }
}

// Implementation - ImageDataCudaArray -------------------------

inline ImageDataCudaArray::ImageDataCudaArray(const NVCVImageData &data)
    : ImageData(data)
{
    if (!IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image buffer format must be suitable for pitch-linear data");
    }
}

inline int32_t ImageDataCudaArray::numPlanes() const
{
    const NVCVImageBufferCudaArray &data = this->cdata().buffer.cudaarray;
    return data.numPlanes;
}

inline cudaArray_t ImageDataCudaArray::plane(int p) const
{
    const NVCVImageBufferCudaArray &data = this->cdata().buffer.cudaarray;

    if (p < 0 || p >= data.numPlanes)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Plane out of bounds");
    }
    return data.planes[p];
}

// Implementation - ImageDataStrided ------------------------------

inline ImageDataStrided::ImageDataStrided(const NVCVImageData &data)
    : ImageData(data)
{
    if (!IsCompatibleKind(data.bufferType))
        throw std::invalid_argument("Incompatible buffer type.");
}

inline Size2D ImageDataStrided::size() const
{
    const NVCVImageBufferStrided &data = this->cdata().buffer.strided;
    if (data.numPlanes > 0)
    {
        return {data.planes[0].width, data.planes[0].height};
    }
    else
    {
        return {0, 0};
    }
}

inline int32_t ImageDataStrided::numPlanes() const
{
    const NVCVImageBufferStrided &data = this->cdata().buffer.strided;
    return data.numPlanes;
}

inline const NVCVImagePlaneStrided &ImageDataStrided::plane(int p) const
{
    const NVCVImageBufferStrided &data = this->cdata().buffer.strided;
    if (p < 0 || p >= data.numPlanes)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Plane out of bounds");
    }
    return data.planes[p];
}

// ImageDataCudaArray implementation -----------------------

inline ImageDataCudaArray::ImageDataCudaArray(ImageFormat format, const Buffer &buffer)
{
    NVCVImageData &data = this->cdata();

    data.format           = format;
    data.bufferType       = NVCV_IMAGE_BUFFER_CUDA_ARRAY;
    data.buffer.cudaarray = buffer;
}

// ImageDataStridedCuda implementation -----------------------

inline ImageDataStridedCuda::ImageDataStridedCuda(ImageFormat format, const Buffer &buffer)
{
    NVCVImageData &data = this->cdata();

    data.format         = format;
    data.bufferType     = NVCV_IMAGE_BUFFER_STRIDED_CUDA;
    data.buffer.strided = buffer;
}

inline ImageDataStridedCuda::ImageDataStridedCuda(const NVCVImageData &data)
    : ImageDataStrided(data)
{
    if (!IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "Image buffer format must be suitable for pitch-linear CUDA-accessible data");
    }
}

// ImageDataStridedHost implementation -----------------------

inline ImageDataStridedHost::ImageDataStridedHost(ImageFormat format, const Buffer &buffer)
{
    NVCVImageData &data = this->cdata();

    data.format         = format;
    data.bufferType     = NVCV_IMAGE_BUFFER_STRIDED_HOST;
    data.buffer.strided = buffer;
}

inline ImageDataStridedHost::ImageDataStridedHost(const NVCVImageData &data)
    : ImageDataStrided(data)
{
    if (!IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "Image buffer format must be suitable for pitch-linear host-accessible data");
    }
}

} // namespace nvcv

#endif // NVCV_IMAGEDATA_IMPL_HPP
