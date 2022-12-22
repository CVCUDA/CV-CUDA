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

#ifndef NVCV_CORE_PRIV_IMAGEBATCHVARSHAPE_HPP
#define NVCV_CORE_PRIV_IMAGEBATCHVARSHAPE_HPP

#include "IImageBatch.hpp"

#include <cuda_runtime.h>

namespace nvcv::priv {

class ImageBatchVarShape final : public CoreObjectBase<IImageBatchVarShape>
{
public:
    explicit ImageBatchVarShape(NVCVImageBatchVarShapeRequirements reqs, IAllocator &alloc);
    ~ImageBatchVarShape();

    static NVCVImageBatchVarShapeRequirements CalcRequirements(int32_t capacity);

    int32_t capacity() const override;
    int32_t numImages() const override;

    Size2D      maxSize() const override;
    ImageFormat uniqueFormat() const override;

    NVCVTypeImageBatch type() const override;

    IAllocator &alloc() const override;

    void getImages(int32_t begIndex, NVCVImageHandle *outImages, int32_t numImages) const override;

    void exportData(CUstream stream, NVCVImageBatchData &data) const override;

    void pushImages(const NVCVImageHandle *images, int32_t numImages) override;
    void pushImages(NVCVPushImageFunc cbPushImage, void *ctxCallback) override;
    void popImages(int32_t numImages) override;
    void clear() override;

private:
    IAllocator                        &m_alloc;
    NVCVImageBatchVarShapeRequirements m_reqs;

    mutable int32_t m_dirtyStartingFromIndex;

    int32_t                 m_numImages;
    NVCVImageBufferStrided *m_hostImagesBuffer;
    NVCVImageBufferStrided *m_devImagesBuffer;

    NVCVImageFormat *m_hostFormatsBuffer;
    NVCVImageFormat *m_devFormatsBuffer;

    NVCVImageHandle *m_imgHandleBuffer;

    // Max width/height up to m_numImages.
    // If nullopt, must be recalculated from the beginning.
    mutable std::optional<Size2D>      m_cacheMaxSize;
    mutable std::optional<ImageFormat> m_cacheUniqueFormat;

    void doUpdateCache() const;

    // TODO: must be retrieved from the resource allocator;
    cudaEvent_t m_evPostFence;

    // Assumes there's enough space for image.
    // Does not update dirty count
    void doPushImage(NVCVImageHandle imgHandle);
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_IMAGEBATCHVARSHAPE_HPP
