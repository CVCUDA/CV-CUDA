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

#ifndef NVCV_IIMAGEBATCHDATA_HPP
#define NVCV_IIMAGEBATCHDATA_HPP

#include "ImageBatchData.h"
#include "ImageData.hpp"
#include "detail/CudaFwd.h"
#include "detail/Optional.hpp"

#include <nvcv/ImageFormat.hpp>

namespace nvcv {

// Interface hierarchy of image batch contents
class IImageBatchData
{
public:
    virtual ~IImageBatchData() = 0;

    int32_t numImages() const;

    const NVCVImageBatchData &cdata() const;

protected:
    IImageBatchData() = default;
    IImageBatchData(const NVCVImageBatchData &data);

    NVCVImageBatchData &cdata();

private:
    NVCVImageBatchData m_data;
};

class IImageBatchVarShapeData : public IImageBatchData
{
public:
    virtual ~IImageBatchVarShapeData() = 0;

    const NVCVImageFormat *formatList() const;
    const NVCVImageFormat *hostFormatList() const;
    Size2D                 maxSize() const;
    ImageFormat            uniqueFormat() const;

protected:
    using IImageBatchData::IImageBatchData;
};

class IImageBatchVarShapeDataStrided : public IImageBatchVarShapeData
{
public:
    virtual ~IImageBatchVarShapeDataStrided() = 0;

    const NVCVImageBufferStrided *imageList() const;

protected:
    using IImageBatchVarShapeData::IImageBatchVarShapeData;
};

class IImageBatchVarShapeDataStridedCuda : public IImageBatchVarShapeDataStrided
{
public:
    virtual ~IImageBatchVarShapeDataStridedCuda() = 0;

protected:
    using IImageBatchVarShapeDataStrided::IImageBatchVarShapeDataStrided;
};

} // namespace nvcv

#include "detail/IImageBatchDataImpl.hpp"

#endif // NVCV_IIMAGEBATCHDATA_HPP
