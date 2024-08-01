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

#ifndef NVCV_IMAGEBATCHDATA_IMPL_HPP
#define NVCV_IMAGEBATCHDATA_IMPL_HPP

#ifndef NVCV_IMAGEBATCHDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// Implementation - ImageBatchData

inline int32_t ImageBatchData::numImages() const
{
    return this->cdata().numImages;
}

inline const NVCVImageBatchData &ImageBatchData::cdata() const &
{
    return m_data;
}

inline NVCVImageBatchData ImageBatchData::cdata() &&
{
    return m_data;
}

inline NVCVImageBatchData &ImageBatchData::data() &
{
    return m_data;
}

template<typename Derived>
Optional<Derived> ImageBatchData::cast() const
{
    static_assert(std::is_base_of<ImageBatchData, Derived>::value, "Cannot cast ImageData to an unrelated type");

    static_assert(sizeof(Derived) == sizeof(NVCVImageBatchData),
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

// Implementation - ImageBatchVarShapeData

inline ImageBatchVarShapeData::ImageBatchVarShapeData(const NVCVImageBatchData &data)
    : ImageBatchData(data)
{
    if (!IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible buffer type.");
    }
}

inline const NVCVImageFormat *ImageBatchVarShapeData::formatList() const
{
    return this->cdata().buffer.varShapeStrided.formatList;
}

inline const NVCVImageFormat *ImageBatchVarShapeData::hostFormatList() const
{
    return this->cdata().buffer.varShapeStrided.hostFormatList;
}

inline Size2D ImageBatchVarShapeData::maxSize() const
{
    const NVCVImageBatchVarShapeBufferStrided &buffer = this->cdata().buffer.varShapeStrided;

    return {buffer.maxWidth, buffer.maxHeight};
}

inline ImageFormat ImageBatchVarShapeData::uniqueFormat() const
{
    return ImageFormat{this->cdata().buffer.varShapeStrided.uniqueFormat};
}

// Implementation - ImageBatchVarShapeDataStrided

inline ImageBatchVarShapeDataStrided::ImageBatchVarShapeDataStrided(const NVCVImageBatchData &data)
    : ImageBatchVarShapeData(data)
{
    if (!IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible buffer type.");
    }
}

inline const NVCVImageBufferStrided *ImageBatchVarShapeDataStrided::imageList() const
{
    return this->cdata().buffer.varShapeStrided.imageList;
}

// Implementation - ImageBatchVarShapeDataStridedCuda

// ImageBatchVarShapeDataStridedCuda implementation -----------------------

inline ImageBatchVarShapeDataStridedCuda::ImageBatchVarShapeDataStridedCuda(const NVCVImageBatchData &data)
    : ImageBatchVarShapeDataStrided(data)
{
    if (!IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible buffer type.");
    }
}

inline ImageBatchVarShapeDataStridedCuda::ImageBatchVarShapeDataStridedCuda(int32_t numImages, const Buffer &buffer)
{
    NVCVImageBatchData &data = this->data();

    data.numImages              = numImages;
    data.bufferType             = NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA;
    data.buffer.varShapeStrided = buffer;
}

} // namespace nvcv

#endif // NVCV_IMAGEBATCHDATA_IMPL_HPP
