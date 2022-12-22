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

#ifndef NVCV_CORE_PRIV_IIMAGEBATCH_HPP
#define NVCV_CORE_PRIV_IIMAGEBATCH_HPP

#include "ICoreObject.hpp"
#include "ImageFormat.hpp"

#include <nvcv/ImageBatch.h>

namespace nvcv::priv {

class IAllocator;

class IImageBatch : public ICoreObjectHandle<IImageBatch, NVCVImageBatchHandle>
{
public:
    virtual int32_t capacity() const  = 0;
    virtual int32_t numImages() const = 0;

    virtual NVCVTypeImageBatch type() const = 0;

    virtual IAllocator &alloc() const = 0;

    virtual void exportData(CUstream stream, NVCVImageBatchData &data) const = 0;
};

class IImageBatchVarShape : public IImageBatch
{
public:
    virtual void pushImages(const NVCVImageHandle *images, int32_t numImages) = 0;
    virtual void pushImages(NVCVPushImageFunc cbPushImage, void *ctxCallback) = 0;

    virtual void popImages(int32_t numImages) = 0;

    virtual void clear() = 0;

    virtual Size2D      maxSize() const      = 0;
    virtual ImageFormat uniqueFormat() const = 0;

    virtual void getImages(int32_t begIndex, NVCVImageHandle *outImages, int32_t numImages) const = 0;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_IIMAGEBATCH_HPP
