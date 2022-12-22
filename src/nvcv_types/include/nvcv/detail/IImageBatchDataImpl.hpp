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

#ifndef NVCV_IIMAGEBATCHDATA_IMPL_HPP
#define NVCV_IIMAGEBATCHDATA_IMPL_HPP

#ifndef NVCV_IIMAGEBATCHDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// Implementation - IImageBatchData

inline IImageBatchData::IImageBatchData(const NVCVImageBatchData &data)
    : m_data(data)
{
}

inline IImageBatchData::~IImageBatchData()
{
    // required dtor implementation
}

inline int32_t IImageBatchData::numImages() const
{
    return this->cdata().numImages;
}

inline const NVCVImageBatchData &IImageBatchData::cdata() const
{
    return m_data;
}

inline NVCVImageBatchData &IImageBatchData::cdata()
{
    return m_data;
}

// Implementation - IImageBatchVarShapeData

inline IImageBatchVarShapeData::~IImageBatchVarShapeData()
{
    // required dtor implementation
}

inline const NVCVImageFormat *IImageBatchVarShapeData::formatList() const
{
    return this->cdata().buffer.varShapeStrided.formatList;
}

inline const NVCVImageFormat *IImageBatchVarShapeData::hostFormatList() const
{
    return this->cdata().buffer.varShapeStrided.hostFormatList;
}

inline Size2D IImageBatchVarShapeData::maxSize() const
{
    const NVCVImageBatchVarShapeBufferStrided &buffer = this->cdata().buffer.varShapeStrided;

    return {buffer.maxWidth, buffer.maxHeight};
}

inline ImageFormat IImageBatchVarShapeData::uniqueFormat() const
{
    return ImageFormat{this->cdata().buffer.varShapeStrided.uniqueFormat};
}

// Implementation - IImageBatchVarShapeDataStrided

inline IImageBatchVarShapeDataStrided::~IImageBatchVarShapeDataStrided()
{
    // required dtor implementation
}

inline const NVCVImageBufferStrided *IImageBatchVarShapeDataStrided::imageList() const
{
    return this->cdata().buffer.varShapeStrided.imageList;
}

// Implementation - IImageBatchVarShapeDataStridedCuda

inline IImageBatchVarShapeDataStridedCuda::~IImageBatchVarShapeDataStridedCuda()
{
    // required dtor implementation
}

} // namespace nvcv

#endif // NVCV_IIMAGEBATCHDATA_IMPL_HPP
