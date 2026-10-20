/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "TensorBatch.hpp"

#include "../NvtxRange.hpp"
#include "CastUtils.hpp"
#include "DataType.hpp"
#include "ExternalBuffer.hpp"
#include "Tensor.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <nanobind/make_iterator.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace nvcvpy::priv {

namespace {

class TensorBatchError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace

size_t TensorBatch::Key::doGetHash() const
{
    using util::ComputeHash;
    return ComputeHash(m_capacity);
}

bool TensorBatch::Key::doIsCompatible(const IKey &ithat) const
{
    auto &that = static_cast<const Key &>(ithat);
    return m_capacity == that.m_capacity;
}

std::shared_ptr<TensorBatch> TensorBatch::Create(int capacity)
{
    std::shared_ptr<CacheItem> item = Cache::Instance().fetchOne(Key{capacity});

    // None found?
    if (!item)
    {
        std::shared_ptr<TensorBatch> batch(new TensorBatch(capacity)); // NOSONAR: constructor is private.
        Cache::Instance().add(*batch);
        return batch;
    }
    else
    {
        auto batch = std::static_pointer_cast<TensorBatch>(item->acquireClaimedObject());
        batch->clear(); // make sure it's in pristine state
        return batch;
    }
}

std::shared_ptr<TensorBatch> TensorBatch::WrapExternalBufferVector(std::vector<py::object>           buffers,
                                                                   std::optional<nvcv::TensorLayout> layout)
{
    TensorList list;
    list.reserve(buffers.size());
    for (auto &obj : buffers)
    {
        std::shared_ptr<ExternalBuffer> buffer = cast_py_object_as<ExternalBuffer>(obj);
        if (!buffer)
        {
            throw TensorBatchError("Input buffer doesn't provide cuda_array_interface or DLPack interfaces.");
        }
        auto tensor = Tensor::Wrap(*buffer, layout);
        list.push_back(tensor);
    }
    auto batch = Create(static_cast<int>(buffers.size()));
    batch->pushBackMany(list);
    return batch;
}

TensorBatch::TensorBatch(int capacity)
    : m_key(capacity)
    , m_data(capacity)
    , m_size_inbytes(doComputeSizeInBytes(nvcv::TensorBatch::CalcRequirements(capacity)))
{
#ifdef NB_FREE_THREADED
    m_data->size = m_size_inbytes;
#endif
}

int64_t TensorBatch::doComputeSizeInBytes(const NVCVTensorBatchRequirements &reqs) const
{
    int64_t size_inbytes;
    util::CheckThrow(nvcvMemRequirementsCalcTotalSizeBytes(&(reqs.mem.cudaMem), &size_inbytes));
    return size_inbytes;
}

int64_t TensorBatch::GetSizeInBytes() const
{
    // m_size_inbytes == -1 indicates failure case and value has not been computed yet
    NVCV_ASSERT(m_size_inbytes != -1
                && "TensorBatch has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
    return m_size_inbytes;
}

RetainedMemory TensorBatch::retainedMemory() const
{
    std::lock_guard lock(m_batchMutex);
    if (m_retainedMemoryDirty)
    {
        m_retainedMemory = Container::retainedMemory();
        for (const std::shared_ptr<Tensor> &tensor : m_data->list)
        {
            if (tensor)
            {
                RetainedMemory child = tensor->retainedMemory();
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

const nvcv::TensorBatch &TensorBatch::impl() const
{
    return m_data->impl;
}

nvcv::TensorBatch &TensorBatch::impl()
{
    return m_data->impl;
}

int32_t TensorBatch::rank() const
{
    std::lock_guard lock(m_batchMutex);
    return m_data->impl.rank();
}

int32_t TensorBatch::capacity() const
{
    std::lock_guard lock(m_batchMutex);
    return m_data->impl.capacity();
}

int32_t TensorBatch::numTensors() const
{
    std::lock_guard lock(m_batchMutex);
    NVCV_ASSERT(m_data->impl.numTensors() == static_cast<int32_t>(m_data->list.size()));
    return m_data->impl.numTensors();
}

std::optional<nvcv::DataType> TensorBatch::dtype() const
{
    std::lock_guard lock(m_batchMutex);
    auto            dtype = m_data->impl.dtype();
    if (dtype != nvcv::DataType())
    {
        return {dtype};
    }
    else
    {
        return std::nullopt;
    }
}

std::optional<nvcv::TensorLayout> TensorBatch::layout() const
{
    std::lock_guard lock(m_batchMutex);
    auto            layout = m_data->impl.layout();
    if (layout != nvcv::TENSOR_NONE)
    {
        return {layout};
    }
    else
    {
        return std::nullopt;
    }
}

void TensorBatch::pushBack(Tensor &tensor)
{
    std::lock_guard lock(m_batchMutex);
    m_data.writable();
    m_retainedMemoryDirty = true;
    m_data->impl.pushBack(tensor.impl());
    m_data->list.push_back(SharedContainerFrom(tensor));
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

void TensorBatch::pushBackMany(std::vector<std::shared_ptr<Tensor>> &tensorList)
{
    std::lock_guard lock(m_batchMutex);
    m_data.writable();
    m_retainedMemoryDirty = true;
    std::vector<nvcv::Tensor> nvcvTensors;
    nvcvTensors.reserve(tensorList.size());
    for (const auto &tensor : tensorList)
    {
        if (tensor)
            nvcvTensors.push_back(tensor->impl());
        else
            nvcvTensors.push_back(nvcv::Tensor());
    }

    m_data->list.reserve(m_data->list.size() + tensorList.size());

    m_data->impl.pushBack(nvcvTensors.begin(), nvcvTensors.end());
    m_data->list.insert(m_data->list.end(), tensorList.begin(), tensorList.end());
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

void TensorBatch::popBack(int tensorCount)
{
    std::lock_guard lock(m_batchMutex);
    m_data.writable();
    m_retainedMemoryDirty = true;
    m_data->impl.popTensors(tensorCount);
    m_data->list.erase(m_data->list.end() - tensorCount, m_data->list.end());
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

void TensorBatch::clear()
{
    std::lock_guard lock(m_batchMutex);
    m_data.writable();
    m_retainedMemoryDirty = true;
    m_data->impl.clear();
    m_data->list.clear();
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

std::shared_ptr<Tensor> TensorBatch::at(int64_t idx) const
{
    std::lock_guard lock(m_batchMutex);
    if (idx < 0)
    {
        throw TensorBatchError("Invalid index: " + std::to_string(idx));
    }
    else if (idx >= static_cast<int64_t>(m_data->list.size()))
    {
        throw TensorBatchError("Cannot get tensor at index " + std::to_string(idx) + ". Batch has only "
                               + std::to_string(m_data->list.size()) + " elements.");
    }
    return m_data->list[idx];
}

void TensorBatch::set_at(int64_t idx, std::shared_ptr<Tensor> tensor)
{
    std::lock_guard lock(m_batchMutex);
    m_data.writable();
    m_retainedMemoryDirty = true;
    if (idx < 0)
    {
        throw TensorBatchError("Invalid index: " + std::to_string(idx));
    }
    else if (idx >= static_cast<int64_t>(m_data->list.size()))
    {
        throw TensorBatchError("Cannot set tensor at index " + std::to_string(idx) + ". Batch has only "
                               + std::to_string(m_data->list.size()) + " elements.");
    }
    m_data->impl.setTensor(static_cast<int32_t>(idx), tensor->impl());
    m_data->list[idx] = tensor;
#ifdef NB_FREE_THREADED
    ++m_version;
#endif
}

auto TensorBatch::begin() const -> TensorList::const_iterator
{
    return m_data->list.begin();
}

auto TensorBatch::end() const -> TensorList::const_iterator
{
    return m_data->list.end();
}

#ifdef NB_FREE_THREADED
void TensorBatch::visitSubmissionResources(LockMode mode, const SubmissionVisitor &visit)
{
    std::lock_guard lock(m_batchMutex);
    auto            data = m_data.snapshot();
    visit(*data, mode == LOCK_MODE_NONE ? mode : LOCK_MODE_READWRITE, true);
    visit(data->submissionChildren(), mode, false);
}

uint64_t TensorBatch::iterationVersion() const
{
    std::lock_guard lock(m_batchMutex);
    return m_version;
}

std::shared_ptr<Tensor> TensorBatch::iterationAt(size_t &index, uint64_t version) const
{
    std::lock_guard lock(m_batchMutex);
    if (version != m_version)
        throw std::runtime_error("Batch structurally modified during iteration");
    if (index == m_data->list.size())
        throw py::stop_iteration();
    return m_data->list[index++];
}

#endif

void TensorBatch::Export(py::module_ &m)
{
    using namespace py::literals;
#ifdef NB_FREE_THREADED
    using Iterator = BatchIterator<TensorBatch, Tensor>;
    py::class_<Iterator>(m, "_TensorBatchIterator")
        .def(
            "__iter__", [](Iterator &it) -> Iterator & { return it; }, py::rv_policy::reference_internal)
        .def("__next__", &Iterator::next, py::lock_self());
#endif

    py::class_<TensorBatch, Container>(
        m, "TensorBatch",
        "Container for a batch of tensors.\n"
        "The capacity of the container must be specified upfront in the batch initialization.\n"
        "The tensors in the batch may differ in shapes but they must have "
        "a uniform dimensionality, data type and layout.",
        py::is_weak_referenceable())
        .def(py::new_(&TensorBatch::Create), "capacity"_a,
             "Create a new TensorBatch object with the specified capacity.")
        .def_prop_ro("layout", &TensorBatch::layout,
                     "Layout of the tensors in the tensor batch."
                     " None if the batch is empty.")
        .def_prop_ro("dtype", &TensorBatch::dtype,
                     "Data type of tensors in the tensor batch."
                     " None if the batch is empty.")
        .def_prop_ro("capacity", &TensorBatch::capacity, "Capacity of the tensor batch.")
        .def_prop_ro("ndim", &TensorBatch::rank,
                     "Return the number of dimensions of the tensors or -1 for an empty batch")
        .def("__len__", &TensorBatch::numTensors, "Return the number of tensors.")
        .def(
            "__iter__",
            [](const TensorBatch &batch)
            {
#ifdef NB_FREE_THREADED
                return Iterator(SharedContainerFrom(batch));
#else
                return py::make_iterator(py::type<TensorBatch>(), "Iterator", batch.begin(), batch.end());
#endif
            },
            py::keep_alive<0, 1>(), "Return an iterator over the tensors in the TensorBatch.")
        .def("__setitem__", &TensorBatch::set_at, "Set tensor at a given index.")
        .def("__getitem__", &TensorBatch::at, "Get a tensor at a given index.")
        .def("pushback", &TensorBatch::pushBack, "Add a new image to the end of the TensorBatch.")
        .def("pushback", &TensorBatch::pushBackMany, "Add multiple images to the end of the TensorBatch.")
        .def("popback", &TensorBatch::popBack, "count"_a = 1,
             "Remove one or more images from the end of the TensorBatch.")
        .def("clear", &TensorBatch::clear, "Remove all images from the TensorBatch.");

    m.def("as_tensors", ::cvcudapy::NvtxTrace("cvcuda.as_tensors", &TensorBatch::WrapExternalBufferVector),
          "buffers"_a = std::vector<py::object>{}, "layout"_a = std::nullopt, py::keep_alive<0, 1>(),
          "Wrap a list of external buffers as a batch of tensors, and tie the buffers lifetime to it");
}

} // namespace nvcvpy::priv
