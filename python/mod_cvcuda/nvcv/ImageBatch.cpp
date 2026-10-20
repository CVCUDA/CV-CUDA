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

#include "ImageBatch.hpp"

#include "../NvtxRange.hpp"
#include "CastUtils.hpp"
#include "ExternalBuffer.hpp"
#include "Image.hpp"
#include "Stream.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <nanobind/make_iterator.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace nvcvpy::priv {

namespace {

class ImageBatchError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace

#ifndef NB_FREE_THREADED
// An unchanged batch shares one immutable child hold across its completion
// generations. Only activation/retirement touches individual child counters;
// cached, idle snapshots must not keep allocations marked as GPU-pending.
class ImageBatchPendingOwners final : public Resource
{
public:
    explicit ImageBatchPendingOwners(const std::vector<std::shared_ptr<Image>> &images)
        : Resource(false)
        , m_images(images)
    {
    }

    std::unique_lock<std::mutex> lockOwnership() const
    {
        return std::unique_lock{m_mutex};
    }

    bool idle() const
    {
        return m_holds == 0;
    }

    void acquirePendingReleaseOwner() noexcept override
    {
        auto lock = lockOwnership();
        if (m_holds++ == 0)
            for (const auto &image : m_images) image->acquirePendingReleaseOwner();
        Resource::acquirePendingReleaseOwner();
    }

    void blockPendingReleaseOwner() noexcept override
    {
        auto lock = lockOwnership();
        NVCV_ASSERT(m_blocked < m_holds);
        if (m_blocked++ == 0)
            for (const auto &image : m_images) image->blockPendingReleaseOwner();
        Resource::blockPendingReleaseOwner();
    }

    void releasePendingReleaseOwner(bool blocked) noexcept override
    {
        auto lock = lockOwnership();
        NVCV_ASSERT(m_holds > 0 && (!blocked || m_blocked > 0));
        if (--m_holds == 0)
        {
            for (const auto &image : m_images) image->releasePendingReleaseOwner(m_blocked != 0);
            m_indexed = false;
        }
        else if (blocked && m_blocked == 1)
        {
            for (const auto &image : m_images) image->unblockPendingReleaseOwner();
        }
        if (blocked)
            --m_blocked;
        Resource::releasePendingReleaseOwner(blocked);
    }

    void indexPendingResources() noexcept override
    {
        auto lock = lockOwnership();
        if (!m_indexed)
        {
            Cache::IndexPending(m_images);
            m_indexed = true;
        }
    }

private:
    const std::vector<std::shared_ptr<Image>> m_images;
    mutable std::mutex                        m_mutex;
    size_t                                    m_holds   = 0;
    size_t                                    m_blocked = 0;
    bool                                      m_indexed = false;
};

std::shared_ptr<Resource> ImageBatchVarShape::pendingOwners() const
{
    if (!m_pendingOwners)
        m_pendingOwners = std::make_shared<ImageBatchPendingOwners>(m_data->list);
    return m_pendingOwners;
}
#endif

size_t ImageBatchVarShape::Key::doGetHash() const
{
    using util::ComputeHash;
    return ComputeHash(m_capacity);
}

bool ImageBatchVarShape::Key::doIsCompatible(const IKey &ithat) const
{
    auto &that = static_cast<const Key &>(ithat);
    return m_capacity == that.m_capacity;
}

bool ImageBatchVarShape::allowsCacheReuse(bool pendingRelease) const
{
    std::lock_guard lock(m_batchMutex);
    return !pendingRelease && m_data->impl.refCount() == 1;
}

std::shared_ptr<ImageBatchVarShape> ImageBatchVarShape::Create(int capacity)
{
    std::shared_ptr<CacheItem> item = Cache::Instance().fetchOne(Key{capacity});

    // None found?
    if (!item)
    {
        std::shared_ptr<ImageBatchVarShape> batch(new ImageBatchVarShape(capacity)); // NOSONAR: constructor is private.
        Cache::Instance().add(*batch);
        return batch;
    }
    else
    {
        auto       batch = std::static_pointer_cast<ImageBatchVarShape>(item->acquireClaimedObject());
        Image::Key key;
        batch->clear(); // make sure it's in pristine state
        return batch;
    }
}

std::shared_ptr<ImageBatchVarShape> ImageBatchVarShape::CreateForOutput(int                             capacity,
                                                                        const std::vector<OutputImage> &images,
                                                                        [[maybe_unused]] Stream        &stream)
{
#ifndef NB_FREE_THREADED
    // Returning an exact batch unchanged avoids touching host descriptors
    // that may still be read by an asynchronous upload. Generic Create keeps
    // its clear/refill behavior and never opts into pending batch reuse.
    Cache::ReusePredicate matches = [&images, &stream](Resource &resource, bool pending)
    {
        auto &batch = static_cast<ImageBatchVarShape &>(resource);
        if (batch.m_data->impl.refCount() != 1 || batch.m_data->list.size() != images.size()
            || py::find(&batch).is_valid() || (pending && !batch.isOrderedOn(stream)))
            return false;
        auto ownershipLock
            = batch.m_pendingOwners ? batch.m_pendingOwners->lockOwnership() : std::unique_lock<std::mutex>{};
        const long idleSnapshotOwner = batch.m_pendingOwners && batch.m_pendingOwners->idle() ? 1 : 0;
        for (size_t index = 0; index < images.size(); ++index)
        {
            const auto &image = batch.m_data->list[index];
            if (!(image->size() == images[index].size) || image->format() != images[index].format
                || image->GetSizeInBytes() == 0 || image->impl().refCount() != 2 || py::find(image.get()).is_valid())
                return false;

            // Prove the cache owner exists; a raw count could otherwise
            // mistake an external owner for an evicted cache entry.
            auto cacheOwner    = Cache::PinItem(*image);
            auto pendingOwners = image->reusablePendingReleaseOwners();
            if (!cacheOwner || (image->hasPendingReleaseOwners() && !pendingOwners.has_value())
                || image.use_count() != static_cast<long>(2 + pendingOwners.value_or(0)) + idleSnapshotOwner
                || ((pending || pendingOwners.has_value()) && !image->isOrderedOn(stream))
                || !image->pendingReleaseOwnershipMatches(pendingOwners.value_or(0)))
                return false;
        }
        return true;
    };
    if (auto item = Cache::Instance().fetchOne(Key{capacity}, {}, &stream, matches))
        return std::static_pointer_cast<ImageBatchVarShape>(item->acquireClaimedObject());
#endif

    auto output = Create(capacity);
    for (const OutputImage &spec : images)
    {
        auto image = Image::Create(spec.size, spec.format, 0);
        output->pushBack(*image);
    }
    return output;
}

std::shared_ptr<ImageBatchVarShape> ImageBatchVarShape::WrapExternalBufferVector(std::vector<py::object> buffers,
                                                                                 nvcv::ImageFormat       fmt)
{
    std::vector<std::shared_ptr<ExternalBuffer>> buflist;
    buflist.reserve(buffers.size());

    for (py::object &obj : buffers)
    {
        std::shared_ptr<ExternalBuffer> buffer = cast_py_object_as<ExternalBuffer>(obj);
        if (!buffer)
        {
            throw ImageBatchError("Input buffer doesn't provide cuda_array_interface or DLPack interfaces");
        }
        buflist.push_back(buffer);
    }

    std::shared_ptr<ImageBatchVarShape> batch = Create(static_cast<int>(buffers.size()));
    batch->pushBackMany(Image::WrapExternalBufferMany(buflist, fmt));

    return batch;
}

ImageBatchVarShape::ImageBatchVarShape(int capacity)
    : m_key(capacity)
    , m_data(capacity)
    , m_size_inbytes(doComputeSizeInBytes(nvcv::ImageBatchVarShape::CalcRequirements(capacity)))
{
#ifdef NB_FREE_THREADED
    m_data->size = m_size_inbytes;
#endif
}

int64_t ImageBatchVarShape::doComputeSizeInBytes(const NVCVImageBatchVarShapeRequirements &reqs) const
{
    int64_t size_inbytes;
    util::CheckThrow(nvcvMemRequirementsCalcTotalSizeBytes(&(reqs.mem.cudaMem), &size_inbytes));
    return size_inbytes;
}

int64_t ImageBatchVarShape::GetSizeInBytes() const
{
    // m_size_inbytes == -1 indicates failure case and value has not been computed yet
    NVCV_ASSERT(m_size_inbytes != -1
                && "ImageBatchVarShape has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
    return m_size_inbytes;
}

RetainedMemory ImageBatchVarShape::retainedMemory() const
{
    std::lock_guard lock(m_batchMutex);
    if (m_retainedMemoryDirty)
    {
        m_retainedMemory = Container::retainedMemory();
        for (const std::shared_ptr<Image> &image : m_data->list)
        {
            if (image)
            {
                RetainedMemory child = image->retainedMemory();
                m_retainedMemory.knownBytes
                    = child.knownBytes > std::numeric_limits<uint64_t>::max() - m_retainedMemory.knownBytes
                        ? std::numeric_limits<uint64_t>::max()
                        : m_retainedMemory.knownBytes + child.knownBytes;
                m_retainedMemory.unknownResources
                    = child.unknownResources > std::numeric_limits<size_t>::max() - m_retainedMemory.unknownResources
                        ? std::numeric_limits<size_t>::max()
                        : m_retainedMemory.unknownResources + child.unknownResources;
            }
        }
        m_retainedMemoryDirty = false;
    }
    return m_retainedMemory;
}

void ImageBatchVarShape::submitSync(Stream &stream)
{
    submitSync(stream, stream.submissionDeviceId());
}

void ImageBatchVarShape::submitSync(Stream &stream, int submissionDevice, StreamOwnerBag *retiredOwners)
{
#ifdef NB_FREE_THREADED
    Resource::submitSync(stream, submissionDevice, retiredOwners);
    return;
#endif
    const auto synchronizedState = syncState();
    Resource::submitSync(stream, submissionDevice, retiredOwners);
    for (const std::shared_ptr<Image> &image : m_data->list)
    {
        if (image && !image->submitSyncThrough(stream, synchronizedState, submissionDevice, retiredOwners))
        {
            image->submitSync(stream, submissionDevice, retiredOwners);
        }
    }
}

const nvcv::ImageBatchVarShape &ImageBatchVarShape::impl() const
{
    return m_data->impl;
}

nvcv::ImageBatchVarShape &ImageBatchVarShape::impl()
{
    return m_data->impl;
}

py::object ImageBatchVarShape::uniqueFormat() const
{
    std::lock_guard   lock(m_batchMutex);
    nvcv::ImageFormat fmt = m_data->impl.uniqueFormat();
    if (fmt)
    {
        return py::cast(fmt);
    }
    else
    {
        return py::none();
    }
}

Size2D ImageBatchVarShape::maxSize() const
{
    std::lock_guard lock(m_batchMutex);
    nvcv::Size2D    s = m_data->impl.maxSize();
    return {s.w, s.h};
}

int32_t ImageBatchVarShape::capacity() const
{
    std::lock_guard lock(m_batchMutex);
    return m_data->impl.capacity();
}

int32_t ImageBatchVarShape::numImages() const
{
    std::lock_guard lock(m_batchMutex);
    NVCV_ASSERT(m_data->impl.numImages() == (int)m_data->list.size());
    return m_data->impl.numImages();
}

void ImageBatchVarShape::pushBack(Image &img)
{
    std::lock_guard lock(m_batchMutex);
#ifndef NB_FREE_THREADED
    m_pendingOwners.reset();
#endif
    m_data.writable();
    m_retainedMemoryDirty = true;
    m_data->impl.pushBack(img.impl());
    m_data->list.push_back(SharedContainerFrom(img));
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

void ImageBatchVarShape::pushBackMany(const std::vector<std::shared_ptr<Image>> &imgList)
{
    std::lock_guard lock(m_batchMutex);
#ifndef NB_FREE_THREADED
    m_pendingOwners.reset();
#endif
    m_data.writable();
    m_retainedMemoryDirty = true;
    std::vector<NVCVImageHandle> handlelist;
    handlelist.reserve(imgList.size());
    for (const auto &img : imgList)
    {
        if (img)
        {
            handlelist.push_back(img->impl().handle());
        }
        else
        {
            handlelist.push_back(nullptr);
        }
    }

    m_data->list.reserve(m_data->list.size() + imgList.size());

    nvcv::detail::CheckThrow(nvcvImageBatchVarShapePushImages(m_data->impl.handle(), handlelist.data(),
                                                              static_cast<int32_t>(handlelist.size())));
    m_data->list.insert(m_data->list.end(), imgList.begin(), imgList.end());
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

void ImageBatchVarShape::popBack(int imgCount)
{
    std::lock_guard lock(m_batchMutex);
#ifndef NB_FREE_THREADED
    m_pendingOwners.reset();
#endif
    m_data.writable();
    m_retainedMemoryDirty = true;
    m_data->impl.popBack(imgCount);
    m_data->list.erase(m_data->list.end() - imgCount, m_data->list.end());
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

void ImageBatchVarShape::clear()
{
    std::lock_guard lock(m_batchMutex);
#ifndef NB_FREE_THREADED
    m_pendingOwners.reset();
#endif
    m_data.writable();
    m_retainedMemoryDirty = true;
    m_data->impl.clear();
    m_data->list.clear();
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

auto ImageBatchVarShape::begin() const -> ImageList::const_iterator
{
    return m_data->list.begin();
}

auto ImageBatchVarShape::end() const -> ImageList::const_iterator
{
    return m_data->list.end();
}

#ifdef NB_FREE_THREADED
void ImageBatchVarShape::visitSubmissionResources(LockMode mode, const SubmissionVisitor &visit)
{
    std::lock_guard lock(m_batchMutex);
    auto            data = m_data.snapshot();
    visit(*data, mode == LOCK_MODE_NONE ? mode : LOCK_MODE_READWRITE, true);
    visit(data->submissionChildren(), mode, false);
}

uint64_t ImageBatchVarShape::iterationVersion() const
{
    std::lock_guard lock(m_batchMutex);
    return m_version;
}

std::shared_ptr<Image> ImageBatchVarShape::iterationAt(size_t &index, uint64_t version) const
{
    std::lock_guard lock(m_batchMutex);
    if (version != m_version)
        throw std::runtime_error("Batch structurally modified during iteration");
    if (index == m_data->list.size())
        throw py::stop_iteration();
    return m_data->list[index++];
}

#endif

void ImageBatchVarShape::Export(py::module_ &m)
{
    using namespace py::literals;
#ifdef NB_FREE_THREADED
    using Iterator = BatchIterator<ImageBatchVarShape, Image>;
    py::class_<Iterator>(m, "_ImageBatchVarShapeIterator")
        .def(
            "__iter__", [](Iterator &it) -> Iterator & { return it; }, py::rv_policy::reference_internal)
        .def("__next__", &Iterator::next, py::lock_self());
#endif

    py::class_<ImageBatchVarShape, Container>(m, "ImageBatchVarShape", "Batch of Images.", py::is_weak_referenceable())
        .def(py::new_(&ImageBatchVarShape::Create), "capacity"_a,
             "Create a new ImageBatchVarShape object with the specified capacity.")
        .def_prop_ro("uniqueformat", &ImageBatchVarShape::uniqueFormat,
                     "Return True if all the images have the same format, False otherwise.")
        .def_prop_ro("maxsize", &ImageBatchVarShape::maxSize,
                     "Return the maximum size of the ImageBatchVarShape in bytes.")
        .def_prop_ro("capacity", &ImageBatchVarShape::capacity,
                     "Return the capacity of the ImageBatchVarShape in number of images.")
        .def("__len__", &ImageBatchVarShape::numImages, "Return the number of images in the ImageBatchVarShape.")
        .def(
            "__iter__",
            [](const ImageBatchVarShape &batch)
            {
#ifdef NB_FREE_THREADED
                return Iterator(SharedContainerFrom(batch));
#else
                return py::make_iterator(py::type<ImageBatchVarShape>(), "Iterator", batch.begin(), batch.end());
#endif
            },
            py::keep_alive<0, 1>(), "Return an iterator over the images in the ImageBatchVarShape.")
        .def("pushback", &ImageBatchVarShape::pushBack, "Add a new image to the end of the ImageBatchVarShape.")
        .def("pushback", &ImageBatchVarShape::pushBackMany, "Add multiple images to the end of the ImageBatchVarShape.")
        .def("popback", &ImageBatchVarShape::popBack, "count"_a = 1,
             "Remove one or more images from the end of the ImageBatchVarShape.")
        .def("clear", &ImageBatchVarShape::clear, "Remove all images from the ImageBatchVarShape.");

    m.def("as_images", ::cvcudapy::NvtxTrace("cvcuda.as_images", &ImageBatchVarShape::WrapExternalBufferVector),
          "buffers"_a = std::vector<py::object>{}, "format"_a = nvcv::FMT_NONE, py::keep_alive<0, 1>(),
          "Wrap a vector of external buffers as a batch of images, and tie the buffers lifetime to it");
}

} // namespace nvcvpy::priv
