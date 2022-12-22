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

#include "ImageBatchVarShape.hpp"

#include "DataType.hpp"
#include "IAllocator.hpp"
#include "IImage.hpp"
#include "ImageManager.hpp"
#include "Requirements.hpp"

#include <cuda_runtime.h>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nvcv::priv {

// ImageBatchVarShape implementation -------------------------------------------

NVCVImageBatchVarShapeRequirements ImageBatchVarShape::CalcRequirements(int32_t capacity)
{
    NVCVImageBatchVarShapeRequirements reqs;
    reqs.capacity = capacity;
    reqs.mem      = {};

    reqs.alignBytes = alignof(NVCVImageBufferStrided);
    reqs.alignBytes = std::lcm(alignof(NVCVImageHandle), reqs.alignBytes);
    reqs.alignBytes = std::lcm(alignof(NVCVImageFormat), reqs.alignBytes);

    reqs.alignBytes = util::RoundUpNextPowerOfTwo(reqs.alignBytes);

    if (reqs.alignBytes > NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment requirement of %d is larger than the maximum allowed %ld", reqs.alignBytes,
                        NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE);
    }

    AddBuffer(reqs.mem.cudaMem, capacity * sizeof(NVCVImageBufferStrided), reqs.alignBytes);
    AddBuffer(reqs.mem.cudaMem, capacity * sizeof(NVCVImageFormat), reqs.alignBytes);

    AddBuffer(reqs.mem.hostMem, capacity * sizeof(NVCVImageBufferStrided), reqs.alignBytes);
    AddBuffer(reqs.mem.hostMem, capacity * sizeof(NVCVImageFormat), reqs.alignBytes);

    AddBuffer(reqs.mem.hostMem, capacity * sizeof(NVCVImageHandle), reqs.alignBytes);

    return reqs;
}

ImageBatchVarShape::ImageBatchVarShape(NVCVImageBatchVarShapeRequirements reqs, IAllocator &alloc)
    : m_alloc{alloc}
    , m_reqs{std::move(reqs)}
    , m_dirtyStartingFromIndex(0)
    , m_numImages(0)
    , m_cacheMaxSize{Size2D{0,0}}
{
    m_evPostFence     = nullptr;
    m_devImagesBuffer = m_hostImagesBuffer = nullptr;
    m_devFormatsBuffer = m_hostFormatsBuffer = nullptr;
    m_imgHandleBuffer                        = nullptr;

    int64_t bufImagesSize  = m_reqs.capacity * sizeof(NVCVImageBufferStrided);
    int64_t bufFormatsSize = m_reqs.capacity * sizeof(NVCVImageFormat);
    int64_t imgHandlesSize = m_reqs.capacity * sizeof(NVCVImageHandle);

    try
    {
        m_devImagesBuffer
            = reinterpret_cast<NVCVImageBufferStrided *>(m_alloc.allocCudaMem(bufImagesSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_devImagesBuffer != nullptr);

        m_hostImagesBuffer
            = reinterpret_cast<NVCVImageBufferStrided *>(m_alloc.allocHostMem(bufImagesSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_devImagesBuffer != nullptr);

        m_devFormatsBuffer
            = reinterpret_cast<NVCVImageFormat *>(m_alloc.allocCudaMem(bufFormatsSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_devFormatsBuffer != nullptr);

        m_hostFormatsBuffer
            = reinterpret_cast<NVCVImageFormat *>(m_alloc.allocHostMem(bufFormatsSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_devFormatsBuffer != nullptr);

        m_imgHandleBuffer
            = reinterpret_cast<NVCVImageHandle *>(m_alloc.allocHostMem(imgHandlesSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_imgHandleBuffer != nullptr);

        NVCV_CHECK_THROW(cudaEventCreateWithFlags(&m_evPostFence, cudaEventDisableTiming));
    }
    catch (...)
    {
        if (m_evPostFence)
        {
            NVCV_CHECK_LOG(cudaEventDestroy(m_evPostFence));
        }

        m_alloc.freeCudaMem(m_devImagesBuffer, bufImagesSize, m_reqs.alignBytes);
        m_alloc.freeHostMem(m_hostImagesBuffer, bufImagesSize, m_reqs.alignBytes);

        m_alloc.freeCudaMem(m_devFormatsBuffer, bufFormatsSize, m_reqs.alignBytes);
        m_alloc.freeHostMem(m_hostFormatsBuffer, bufFormatsSize, m_reqs.alignBytes);

        m_alloc.freeHostMem(m_imgHandleBuffer, imgHandlesSize, m_reqs.alignBytes);
        throw;
    }
}

ImageBatchVarShape::~ImageBatchVarShape()
{
    NVCV_CHECK_LOG(cudaEventSynchronize(m_evPostFence));

    int64_t bufImagesSize  = m_reqs.capacity * sizeof(NVCVImageBufferStrided);
    int64_t bufFormatsSize = m_reqs.capacity * sizeof(NVCVImageFormat);
    int64_t imgHandlesSize = m_reqs.capacity * sizeof(NVCVImageHandle);

    m_alloc.freeCudaMem(m_devImagesBuffer, bufImagesSize, m_reqs.alignBytes);
    m_alloc.freeHostMem(m_hostImagesBuffer, bufImagesSize, m_reqs.alignBytes);

    m_alloc.freeCudaMem(m_devFormatsBuffer, bufFormatsSize, m_reqs.alignBytes);
    m_alloc.freeHostMem(m_hostFormatsBuffer, bufFormatsSize, m_reqs.alignBytes);

    m_alloc.freeHostMem(m_imgHandleBuffer, imgHandlesSize, m_reqs.alignBytes);

    NVCV_CHECK_LOG(cudaEventDestroy(m_evPostFence));
}

NVCVTypeImageBatch ImageBatchVarShape::type() const
{
    return NVCV_TYPE_IMAGEBATCH_VARSHAPE;
}

int32_t ImageBatchVarShape::capacity() const
{
    return m_reqs.capacity;
}

int32_t ImageBatchVarShape::numImages() const
{
    return m_numImages;
}

Size2D ImageBatchVarShape::maxSize() const
{
    doUpdateCache();
    return *m_cacheMaxSize;
}

ImageFormat ImageBatchVarShape::uniqueFormat() const
{
    doUpdateCache();
    return *m_cacheUniqueFormat;
}

IAllocator &ImageBatchVarShape::alloc() const
{
    return m_alloc;
}

void ImageBatchVarShape::doUpdateCache() const
{
    if (m_cacheMaxSize && m_cacheUniqueFormat)
    {
        return;
    }

    if (!m_cacheMaxSize)
    {
        m_cacheMaxSize = Size2D{0, 0};
    }

    for (int i = 0; i < m_numImages; ++i)
    {
        if (m_cacheMaxSize)
        {
            // first plane has the image size
            m_cacheMaxSize->w = std::max(m_cacheMaxSize->w, m_hostImagesBuffer[i].planes[0].width);
            m_cacheMaxSize->h = std::max(m_cacheMaxSize->h, m_hostImagesBuffer[i].planes[0].height);
        }

        constexpr ImageFormat fmt_none = ImageFormat{NVCV_IMAGE_FORMAT_NONE};

        if (!m_cacheUniqueFormat)
        {
            m_cacheUniqueFormat = ImageFormat{m_hostFormatsBuffer[i]};
        }
        else if (*m_cacheUniqueFormat != fmt_none && *m_cacheUniqueFormat != ImageFormat{m_hostFormatsBuffer[i]})
        {
            *m_cacheUniqueFormat = fmt_none;
        }
    }

    if (!m_cacheUniqueFormat)
    {
        NVCV_ASSERT(m_numImages == 0);
        m_cacheUniqueFormat = ImageFormat{NVCV_IMAGE_FORMAT_NONE};
    }
}

void ImageBatchVarShape::exportData(CUstream stream, NVCVImageBatchData &data) const
{
    data.numImages  = m_numImages;
    data.bufferType = NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA;

    NVCVImageBatchVarShapeBufferStrided &buf = data.buffer.varShapeStrided;
    buf.imageList                            = m_devImagesBuffer;
    buf.formatList                           = m_devFormatsBuffer;
    buf.hostFormatList                       = m_hostFormatsBuffer;

    NVCV_ASSERT(m_dirtyStartingFromIndex <= m_numImages);

    if (m_dirtyStartingFromIndex < m_numImages)
    {
        NVCV_CHECK_THROW(cudaMemcpyAsync(
            m_devImagesBuffer + m_dirtyStartingFromIndex, m_hostImagesBuffer + m_dirtyStartingFromIndex,
            (m_numImages - m_dirtyStartingFromIndex) * sizeof(*m_devImagesBuffer), cudaMemcpyHostToDevice, stream));

        NVCV_CHECK_THROW(cudaMemcpyAsync(
            m_devFormatsBuffer + m_dirtyStartingFromIndex, m_hostFormatsBuffer + m_dirtyStartingFromIndex,
            (m_numImages - m_dirtyStartingFromIndex) * sizeof(*m_devFormatsBuffer), cudaMemcpyHostToDevice, stream));

        // Signal that we finished reading from m_hostBuffer
        NVCV_CHECK_THROW(cudaEventRecord(m_evPostFence, stream));

        // up to m_numImages, we're all good
        m_dirtyStartingFromIndex = m_numImages;
    }

    doUpdateCache();

    NVCV_ASSERT(m_cacheMaxSize);
    buf.maxWidth  = m_cacheMaxSize->w;
    buf.maxHeight = m_cacheMaxSize->h;

    NVCV_ASSERT(m_cacheUniqueFormat);
    buf.uniqueFormat = m_cacheUniqueFormat->value();
}

void ImageBatchVarShape::pushImages(const NVCVImageHandle *images, int32_t numImages)
{
    if (images == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Images buffer cannot be NULL");
    }

    if (numImages + m_numImages > m_reqs.capacity)
    {
        throw Exception(NVCV_ERROR_OVERFLOW,
                        "Adding %d images to image batch would make its size %d exceed its capacity %d", numImages,
                        numImages + m_numImages, m_reqs.capacity);
    }

    // Wait till m_hostBuffer is free to be written to (all pending reads are finished).
    NVCV_CHECK_THROW(cudaEventSynchronize(m_evPostFence));

    int oldNumImages = m_numImages;

    try
    {
        for (int i = 0; i < numImages; ++i)
        {
            doPushImage(images[i]);
        }
    }
    catch (...)
    {
        m_numImages = oldNumImages;
        throw;
    }
}

void ImageBatchVarShape::pushImages(NVCVPushImageFunc cbPushImage, void *ctxCallback)
{
    if (cbPushImage == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Callback function that adds images to the image batch cannot be NULL");
    }

    // Wait till m_hostBuffer is free to be written to (all pending reads are finished).
    NVCV_CHECK_THROW(cudaEventSynchronize(m_evPostFence));

    int oldNumImages = m_numImages;

    try
    {
        while (NVCVImageHandle imgHandle = cbPushImage(ctxCallback))
        {
            if (m_numImages == m_reqs.capacity)
            {
                throw Exception(NVCV_ERROR_OVERFLOW,
                                "Adding one more image to image batch would make its size exceed its capacity %d",
                                m_reqs.capacity);
            }

            doPushImage(imgHandle);
        }
    }
    catch (...)
    {
        m_numImages = oldNumImages;
        throw;
    }
}

void ImageBatchVarShape::doPushImage(NVCVImageHandle imgHandle)
{
    NVCV_ASSERT(m_numImages < m_reqs.capacity);

    auto &img = ToStaticRef<IImage>(imgHandle);

    if (img.format().memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Format of image to be added, " << img.format()
                                                     << " must be pitch-linear, not " << img.format().memLayout();
    }

    NVCVImageData imgData;
    img.exportData(imgData);

    if (imgData.bufferType != NVCV_IMAGE_BUFFER_STRIDED_CUDA)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Data buffer of image to be added isn't gpu-accessible";
    }

    m_hostImagesBuffer[m_numImages]  = imgData.buffer.strided;
    m_hostFormatsBuffer[m_numImages] = imgData.format;
    m_imgHandleBuffer[m_numImages]   = imgHandle;

    Size2D imgSize = img.size();

    ++m_numImages;

    if (m_numImages == 1)
    {
        m_cacheMaxSize      = imgSize;
        m_cacheUniqueFormat = ImageFormat{imgData.format};
    }
    else
    {
        // Only update max size if data is valid
        if (m_cacheMaxSize)
        {
            m_cacheMaxSize->w = std::max(m_cacheMaxSize->w, imgSize.w);
            m_cacheMaxSize->h = std::max(m_cacheMaxSize->h, imgSize.h);
        }

        if (m_cacheUniqueFormat && *m_cacheUniqueFormat != ImageFormat{imgData.format})
        {
            m_cacheUniqueFormat = ImageFormat{NVCV_IMAGE_FORMAT_NONE};
        }
    }
}

void ImageBatchVarShape::popImages(int32_t numImages)
{
    if (m_numImages - numImages < 0)
    {
        throw Exception(NVCV_ERROR_UNDERFLOW,
                        "Cannot remove more images, %d, than the number of images, %d, in the image batch", numImages,
                        m_numImages);
    }

    m_numImages -= numImages;

    if (m_dirtyStartingFromIndex > m_numImages)
    {
        m_dirtyStartingFromIndex = m_numImages;
    }

    // Removing images invalidates size.
    m_cacheMaxSize = std::nullopt;
    // It *does not* always invalidate m_cacheUniqueFormat, though.
    // But if we're now empty, yeah, we reset it.
    if (m_numImages == 0)
    {
        m_cacheUniqueFormat = std::nullopt;
    }
}

void ImageBatchVarShape::getImages(int32_t begIndex, NVCVImageHandle *outImages, int32_t numImages) const
{
    if (begIndex + numImages > m_numImages)
    {
        throw Exception(NVCV_ERROR_OVERFLOW, "Cannot get images past end of image batch");
    }

    std::copy(m_imgHandleBuffer + begIndex, m_imgHandleBuffer + begIndex + numImages, outImages);
}

void ImageBatchVarShape::clear()
{
    m_numImages              = 0;
    m_dirtyStartingFromIndex = 0;
    m_cacheMaxSize           = {0, 0};
    m_cacheUniqueFormat      = std::nullopt;
}

} // namespace nvcv::priv
