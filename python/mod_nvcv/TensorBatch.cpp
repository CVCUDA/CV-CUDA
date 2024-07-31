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

#include "TensorBatch.hpp"

#include "CastUtils.hpp"
#include "DataType.hpp"
#include "ExternalBuffer.hpp"
#include "Tensor.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>

namespace nvcvpy::priv {

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
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{capacity});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<TensorBatch> batch(new TensorBatch(capacity));
        Cache::Instance().add(*batch);
        return batch;
    }
    else
    {
        // Get the first one
        auto batch = std::static_pointer_cast<TensorBatch>(vcont[0]);
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
            throw std::runtime_error("Input buffer doesn't provide cuda_array_interface or DLPack interfaces.");
        }
        auto tensor = Tensor::Wrap(*buffer, layout);
        list.push_back(tensor);
    }
    auto batch = Create(buffers.size());
    batch->pushBackMany(list);
    return batch;
}

TensorBatch::TensorBatch(int capacity)
    : m_key(capacity)
    , m_impl(capacity)
    , m_size_inbytes(doComputeSizeInBytes(nvcv::TensorBatch::CalcRequirements(capacity)))
{
    m_list.reserve(capacity);
}

int64_t TensorBatch::doComputeSizeInBytes(const NVCVTensorBatchRequirements &reqs)
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

const nvcv::TensorBatch &TensorBatch::impl() const
{
    return m_impl;
}

nvcv::TensorBatch &TensorBatch::impl()
{
    return m_impl;
}

int32_t TensorBatch::rank() const
{
    return m_impl.rank();
}

int32_t TensorBatch::capacity() const
{
    return m_impl.capacity();
}

int32_t TensorBatch::numTensors() const
{
    NVCV_ASSERT(m_impl.numTensors() == static_cast<int32_t>(m_list.size()));
    return m_impl.numTensors();
}

std::optional<nvcv::DataType> TensorBatch::dtype() const
{
    auto dtype = m_impl.dtype();
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
    auto layout = m_impl.layout();
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
    m_impl.pushBack(tensor.impl());
    m_list.push_back(tensor.shared_from_this());
}

void TensorBatch::pushBackMany(std::vector<std::shared_ptr<Tensor>> &tensorList)
{
    std::vector<nvcv::Tensor> nvcvTensors;
    nvcvTensors.reserve(tensorList.size());
    for (auto &tensor : tensorList)
    {
        m_list.push_back(tensor);
        if (tensor)
            nvcvTensors.push_back(tensor->impl());
        else
            nvcvTensors.push_back(nvcv::Tensor());
    }
    m_impl.pushBack(nvcvTensors.begin(), nvcvTensors.end());
}

void TensorBatch::popBack(int tensorCount)
{
    m_impl.popTensors(tensorCount);
    m_list.erase(m_list.end() - tensorCount, m_list.end());
}

void TensorBatch::clear()
{
    m_impl.clear();
    m_list.clear();
}

std::shared_ptr<Tensor> TensorBatch::at(int64_t idx) const
{
    if (idx < 0)
    {
        throw std::runtime_error("Invalid index: " + std::to_string(idx));
    }
    else if (idx >= static_cast<int64_t>(m_list.size()))
    {
        throw std::runtime_error("Cannot get tensor at index " + std::to_string(idx) + ". Batch has only "
                                 + std::to_string(m_list.size()) + " elements.");
    }
    return m_list[idx];
}

void TensorBatch::set_at(int64_t idx, std::shared_ptr<Tensor> tensor)
{
    if (idx < 0)
    {
        throw std::runtime_error("Invalid index: " + std::to_string(idx));
    }
    else if (idx >= static_cast<int64_t>(m_list.size()))
    {
        throw std::runtime_error("Cannot set tensor at index " + std::to_string(idx) + ". Batch has only "
                                 + std::to_string(m_list.size()) + " elements.");
    }
    m_impl.setTensor(static_cast<int32_t>(idx), tensor->impl());
    m_list[idx] = tensor;
}

auto TensorBatch::begin() const -> TensorList::const_iterator
{
    return m_list.begin();
}

auto TensorBatch::end() const -> TensorList::const_iterator
{
    return m_list.end();
}

std::shared_ptr<TensorBatch> TensorBatch::shared_from_this()
{
    return std::static_pointer_cast<TensorBatch>(Container::shared_from_this());
}

std::shared_ptr<const TensorBatch> TensorBatch::shared_from_this() const
{
    return std::static_pointer_cast<const TensorBatch>(Container::shared_from_this());
}

void TensorBatch::Export(py::module &m)
{
    using namespace py::literals;

    py::class_<TensorBatch, std::shared_ptr<TensorBatch>, Container>(
        m, "TensorBatch",
        "Container for a batch of tensors.\n"
        "The capacity of the container must be specified upfront in the batch initialization.\n"
        "The tensors in the batch may differ in shapes but they must have "
        "a uniform dimensionality, data type and layout.")
        .def(py::init(&TensorBatch::Create), "capacity"_a,
             "Create a new TensorBatch object with the specified capacity.")
        .def_property_readonly("layout", &TensorBatch::layout,
                               "Layout of the tensors in the tensor batch."
                               " None if the batch is empty.")
        .def_property_readonly("dtype", &TensorBatch::dtype,
                               "Data type of tensors in the tensor batch."
                               " None if the batch is empty.")
        .def_property_readonly("capacity", &TensorBatch::capacity, "Capacity of the tensor batch.")
        .def_property_readonly("ndim", &TensorBatch::rank,
                               "Return the number of dimensions of the tensors or -1 for an empty batch")
        .def("__len__", &TensorBatch::numTensors, "Return the number of tensors.")
        .def(
            "__iter__", [](const TensorBatch &batch) { return py::make_iterator(batch); },
            "Return an iterator over the tensors in the TensorBatch.")
        .def("__setitem__", &TensorBatch::set_at, "Set tensor at a given index.")
        .def("__getitem__", &TensorBatch::at, "Get a tensor at a given index.")
        .def("pushback", &TensorBatch::pushBack, "Add a new image to the end of the TensorBatch.")
        .def("pushback", &TensorBatch::pushBackMany, "Add multiple images to the end of the TensorBatch.")
        .def("popback", &TensorBatch::popBack, "count"_a = 1,
             "Remove one or more images from the end of the TensorBatch.")
        .def("clear", &TensorBatch::clear, "Remove all images from the TensorBatch.");

    m.def("as_tensors", &TensorBatch::WrapExternalBufferVector, "buffers"_a = std::vector<py::object>{},
          "layout"_a = std::nullopt, py::keep_alive<0, 1>(),
          "Wrap a list of external buffers as a batch of tensors, and tie the buffers lifetime to it");
}

} // namespace nvcvpy::priv
