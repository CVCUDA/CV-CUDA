/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PYTHON_PRIV_IMAGEBATCH_HPP
#define NVCV_PYTHON_PRIV_IMAGEBATCH_HPP

#include "BatchData.hpp"
#include "Container.hpp"
#include "Size.hpp"

#include <nvcv/ImageBatch.hpp>

#include <vector>

namespace nvcvpy::priv {
namespace py = nanobind;

class Image;
class ImageBatchPendingOwners;

class ImageBatchVarShape : public Container
{
    using ImageList = std::vector<std::shared_ptr<Image>>;

public:
    static void Export(py::module_ &m);

#ifdef NB_FREE_THREADED
    BatchMutex &batchMutex() const
    {
        return m_batchMutex;
    }

    BatchMutex *submissionBatchMutex() const override
    {
        return &m_batchMutex;
    }

    uint64_t               iterationVersion() const;
    std::shared_ptr<Image> iterationAt(size_t &index, uint64_t version) const;
    void                   visitSubmissionResources(LockMode mode, const SubmissionVisitor &visit) override;
#endif

    static std::shared_ptr<ImageBatchVarShape> Create(int capacity);

    struct OutputImage
    {
        Size2D            size;
        nvcv::ImageFormat format;
    };

    static std::shared_ptr<ImageBatchVarShape> CreateForOutput(int capacity, const std::vector<OutputImage> &images,
                                                               Stream &stream);
    static std::shared_ptr<ImageBatchVarShape> WrapExternalBufferVector(std::vector<py::object> buffer,
                                                                        nvcv::ImageFormat       fmt);

    const nvcv::ImageBatchVarShape &impl() const;
    nvcv::ImageBatchVarShape       &impl();

    // Let's simplify a bit and NOT export the base class ImageBatch,
    // as we currently have only one leaf class (this one).
    py::object uniqueFormat() const;
    int32_t    capacity() const;
    int32_t    numImages() const;
    Size2D     maxSize() const;

    int64_t        GetSizeInBytes() const override;
    RetainedMemory retainedMemory() const override;
    bool           allowsCacheReuse(bool pendingRelease) const override;
    void           submitSync(Stream &stream) override;
    void           submitSync(Stream &stream, int submissionDeviceId, StreamOwnerBag *retiredOwners = nullptr) override;

    void pushBack(Image &img);
    void pushBackMany(const std::vector<std::shared_ptr<Image>> &imgList);
    void popBack(int imgCount);
    void clear();

    ImageList::const_iterator begin() const;
    ImageList::const_iterator end() const;

#ifndef NB_FREE_THREADED
    std::shared_ptr<Resource> pendingOwners() const;
#endif

    class Key final : public IKey
    {
    public:
        explicit Key(int capacity)
            : m_capacity(capacity)
        {
        }

    private:
        int m_capacity;

        size_t doGetHash() const override;
        bool   doIsCompatible(const IKey &that) const override;
    };

    const Key &key() const override
    {
        return m_key;
    }

private:
    explicit ImageBatchVarShape(int capacity);

    int64_t doComputeSizeInBytes(const NVCVImageBatchVarShapeRequirements &reqs) const;

#ifdef NB_FREE_THREADED
    mutable BatchMutex m_batchMutex;
    uint64_t           m_version = 0;
#else
    inline static BatchMutex m_batchMutex;
#endif
    Key                                           m_key;
    BatchStorage<nvcv::ImageBatchVarShape, Image> m_data;
    int64_t                                       m_size_inbytes        = -1;
    mutable bool                                  m_retainedMemoryDirty = true;
    mutable RetainedMemory                        m_retainedMemory;
#ifndef NB_FREE_THREADED
    mutable std::shared_ptr<ImageBatchPendingOwners> m_pendingOwners;
#endif
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_IMAGEBATCH_HPP
