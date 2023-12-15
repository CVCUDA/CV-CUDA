/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Array.hpp"

#include "DataType.hpp"
#include "ExternalBuffer.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/Hash.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace nvcvpy::priv {

std::shared_ptr<Array> Array::CreateFromReqs(const nvcv::Array::Requirements &reqs)
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{reqs});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Array> array(new Array(reqs));
        array->impl().resize(reqs.capacity);
        Cache::Instance().add(*array);
        return array;
    }
    else
    {
        // Get the first one
        auto array = std::static_pointer_cast<Array>(vcont[0]);
        NVCV_ASSERT(array->dtype() == reqs.dtype);
        return array;
    }
}

std::shared_ptr<Array> Array::Create(int64_t length, nvcv::DataType dtype)
{
    nvcv::Array::Requirements reqs = nvcv::Array::CalcRequirements(length, dtype);
    return CreateFromReqs(reqs);
}

std::shared_ptr<Array> Array::Create(const Shape &shape, nvcv::DataType dtype)
{
    return Create(LengthIf1D(shape), dtype);
}

namespace {

NVCVArrayData FillNVCVArrayData(const DLTensor &tensor, NVCVArrayBufferType bufType)
{
    NVCVArrayData arrayData = {};

    // dtype ------------
    arrayData.dtype = py::cast<nvcv::DataType>(ToDType(ToNVCVDataType(tensor.dtype)));

    // rank ------------
    {
        // TODO: Add 0D support
        int rank = tensor.ndim == 0 ? 1 : tensor.ndim;
        if (rank != 1)
        {
            throw std::invalid_argument(util::FormatString("The tensor rank must be 1 not %d", rank));
        }
    }

    // shape ------------
    arrayData.capacity = arrayData.length = tensor.shape[0];

    // buffer type ------------
    if (IsCudaAccessible(tensor.device.device_type))
    {
        arrayData.bufferType = NVCV_ARRAY_BUFFER_HOST;
    }
    else
    {
        throw std::runtime_error("Only CUDA-accessible arrays are supported for now");
    }

    NVCVArrayBufferStrided &dataStrided = arrayData.buffer.strided;

    // stride ------------
    int elemStrideBytes = (tensor.dtype.bits * tensor.dtype.lanes + 7) / 8;
    for (int d = 0; d < tensor.ndim; ++d)
    {
        dataStrided.stride = tensor.strides[d] * elemStrideBytes;
    }

    // Memory buffer ------------
    dataStrided.basePtr = reinterpret_cast<NVCVByte *>(tensor.data) + tensor.byte_offset;

    return arrayData;
}

NVCVArrayData FillNVCVArrayDataCUDA(const DLTensor &tensor)
{
    return FillNVCVArrayData(tensor, NVCV_ARRAY_BUFFER_HOST);
}

} // namespace

std::shared_ptr<Array> Array::Wrap(ExternalBuffer &buffer)
{
    const DLTensor &dlTensor = buffer.dlTensor();

    nvcv::ArrayDataCuda data{FillNVCVArrayDataCUDA(dlTensor)};

    // This is the key of a tensor wrapper.
    // All tensor wrappers have the same key.
    Array::Key key;
    // We take this opportunity to remove from cache all wrappers that aren't
    // being used. They aren't reusable anyway.
    Cache::Instance().removeAllNotInUseMatching(key);

    auto array = std::shared_ptr<Array>(new Array(data, py::cast(buffer.shared_from_this())));

    // Need to add wrappers to cache so that they don't get destroyed by
    // the cuda stream when they're last used, and python script isn't
    // holding a reference to them. If we don't do it, things might break.
    Cache::Instance().add(*array);
    return array;
}

std::shared_ptr<Array> Array::ResizeArray(Array &array, int64_t length)
{
    Array::Key key;
    Cache::Instance().removeAllNotInUseMatching(key);

    auto array_impl = array.impl();
    array_impl.resize(length);

    auto new_array = std::shared_ptr<Array>(new Array(std::move(array_impl)));

    // Need to add wrappers to cache so that they don't get destroyed by
    // the cuda stream when they're last used, and python script isn't
    // holding a reference to them. If we don't do it, things might break.
    Cache::Instance().add(*new_array);
    return new_array;
}

std::shared_ptr<Array> Array::ResizeArray(Array &array, Shape shape)
{
    return ResizeArray(array, LengthIf1D(shape));
}

std::shared_ptr<Array> Array::Resize(int64_t length)
{
    return ResizeArray(*this, length);
}

std::shared_ptr<Array> Array::Resize(Shape shape)
{
    return ResizeArray(*this, shape);
}

Array::Array(const nvcv::Array::Requirements &reqs)
    : m_impl{reqs}
    , m_key{reqs}
{
}

Array::Array(const nvcv::ArrayData &data, py::object wrappedObject)
    : m_impl{nvcv::ArrayWrapData(data)}
    , m_key{}
    , m_wrappedObject(wrappedObject)
{
}

Array::Array(nvcv::Array &&array)
    : m_impl{std::move(array)}
    , m_key{}
{
}

std::shared_ptr<Array> Array::shared_from_this()
{
    return std::static_pointer_cast<Array>(Container::shared_from_this());
}

std::shared_ptr<const Array> Array::shared_from_this() const
{
    return std::static_pointer_cast<const Array>(Container::shared_from_this());
}

nvcv::Array &Array::impl()
{
    return m_impl;
}

const nvcv::Array &Array::impl() const
{
    return m_impl;
}

Shape Array::shape() const
{
    return CreateShape(m_impl.length());
}

nvcv::DataType Array::dtype() const
{
    return m_impl.dtype();
}

int Array::rank() const
{
    return m_impl.rank();
}

int64_t Array::length() const
{
    return m_impl.length();
}

Array::Key::Key(const nvcv::Array::Requirements &reqs)
    : Key(reqs.capacity, static_cast<nvcv::DataType>(reqs.dtype))
{
}

Array::Key::Key(int64_t length, nvcv::DataType dtype)
    : m_length(std::move(length))
    , m_dtype(dtype)
    , m_wrapper(false)
{
}

size_t Array::Key::doGetHash() const
{
    if (m_wrapper)
    {
        return 0; // all wrappers are equal wrt. the cache
    }
    else
    {
        using util::ComputeHash;
        return ComputeHash(m_length, m_dtype);
    }
}

bool Array::Key::doIsCompatible(const IKey &that_) const
{
    const Key &that = static_cast<const Key &>(that_);

    // Wrapper key's all compare equal, are they can't be used
    // and whenever we query the cache for wrappers, we really
    // want to get them all (as long as they aren't being used).
    if (m_wrapper && that.m_wrapper)
    {
        return true;
    }
    else if (m_wrapper || that.m_wrapper) // xor
    {
        return false;
    }
    else
    {
        return std::tie(m_length, m_dtype) == std::tie(that.m_length, that.m_dtype);
    }
}

auto Array::key() const -> const Key &
{
    return m_key;
}

static py::object ToPython(const nvcv::ArrayData &arrayData, py::object owner)
{
    py::object out;

    auto data = arrayData.cast<nvcv::ArrayData>();
    if (!data)
    {
        throw std::runtime_error("Only tensors with pitch-linear data can be exported");
    }

    DLPackTensor dlTensor(*data);
    return ExternalBuffer::Create(std::move(dlTensor), owner);
}

py::object Array::cuda() const
{
    nvcv::ArrayData arrayData = m_impl.exportData();

    // Note: we can't cache the returned ExternalBuffer because it is holding
    // a reference to us. Doing so would lead to mem leaks.
    return ToPython(arrayData, py::cast(this->shared_from_this()));
}

std::ostream &operator<<(std::ostream &out, const Array &array)
{
    return out << "<nvcv.Array length=" << array.length()
               << " dtype=" << py::str(py::cast(array.dtype())).cast<std::string>() << '>';
}

void Array::Export(py::module &m)
{
    using namespace py::literals;

    using CreateFromLengthPtr = std::shared_ptr<Array> (*)(int64_t, nvcv::DataType);
    using CreateFromShapePtr  = std::shared_ptr<Array> (*)(const Shape &, nvcv::DataType);

    using ResizeLengthPtr = std::shared_ptr<Array> (Array::*)(int64_t DataType);
    using ResizeShapePtr  = std::shared_ptr<Array> (Array::*)(Shape);

    using ResizeArrayLengthPtr = std::shared_ptr<Array> (*)(Array &, int64_t DataType);
    using ResizeArrayShapePtr  = std::shared_ptr<Array> (*)(Array &, Shape);

    py::class_<Array, std::shared_ptr<Array>, Container>(m, "Array")
        .def(py::init(static_cast<CreateFromLengthPtr>(&Array::Create)), "length"_a, "dtype"_a,
             "Create a Array object with the given length and data type.")
        .def(py::init(static_cast<CreateFromShapePtr>(&Array::Create)), "shape"_a, "dtype"_a,
             "Create a Array object with the given shape and data type.")
        .def_property_readonly("shape", &Array::shape, "The shape of the Array.")
        .def_property_readonly("dtype", &Array::dtype, "The data type of the Array.")
        // numpy and others use ndim, let's be consistent with them in python.
        // It's not a requirement to be consistent between NVCV Python and C/C++.
        // Each language use whatever is appropriate (and expected) in their environment.
        .def_property_readonly("ndim", &Array::rank, "The number of dimensions of the Array.")
        .def("cuda", &Array::cuda, "Reference to the Array on the CUDA device.")
        .def("resize", static_cast<ResizeLengthPtr>(&Array::Resize), "length"_a,
             "Produces an array pointing to the same data but with a new length.")
        .def("resize", static_cast<ResizeShapePtr>(&Array::Resize), "shape"_a,
             "Produces an array pointing to the same data but with a new shape.")
        .def("__repr__", &util::ToString<Array>, "Return the string representation of the Array object.");

    m.def("as_array", &Array::Wrap, "buffer"_a, "Wrap an existing buffer into a Array object with the given layout.");
    m.def("resize", static_cast<ResizeArrayLengthPtr>(&Array::ResizeArray), "array"_a, "length"_a,
          "Produces an array pointing to the same data but with a new length.");
    m.def("resize", static_cast<ResizeArrayShapePtr>(&Array::ResizeArray), "array"_a, "shape"_a,
          "Produces an array pointing to the same data but with a new shape.");
}

} // namespace nvcvpy::priv
