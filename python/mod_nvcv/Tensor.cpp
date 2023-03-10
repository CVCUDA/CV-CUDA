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

#include "Tensor.hpp"

#include "DataType.hpp"
#include "ExternalBuffer.hpp"
#include "Image.hpp"
#include "ImageFormat.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/Hash.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <nvcv/TensorShapeInfo.hpp>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace nvcv {

static size_t ComputeHash(const nvcv::TensorShape &shape)
{
    using nvcvpy::util::ComputeHash;
    return ComputeHash(shape.shape(), shape.layout());
}

} // namespace nvcv

namespace nvcvpy::priv {

std::shared_ptr<Tensor> Tensor::CreateForImageBatch(int numImages, const Size2D &size, nvcv::ImageFormat fmt)
{
    nvcv::Tensor::Requirements reqs
        = nvcv::Tensor::CalcRequirements(numImages, nvcv::Size2D{std::get<0>(size), std::get<1>(size)}, fmt);
    return CreateFromReqs(reqs);
}

std::shared_ptr<Tensor> Tensor::Create(Shape shape, nvcv::DataType dtype, std::optional<nvcv::TensorLayout> layout)
{
    if (!layout)
    {
        layout = nvcv::TENSOR_NONE;
    }

    nvcv::Tensor::Requirements reqs = nvcv::Tensor::CalcRequirements(CreateNVCVTensorShape(shape, *layout), dtype);
    return CreateFromReqs(reqs);
}

std::shared_ptr<Tensor> Tensor::CreateFromReqs(const nvcv::Tensor::Requirements &reqs)
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{reqs});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Tensor> tensor(new Tensor(reqs));
        Cache::Instance().add(*tensor);
        return tensor;
    }
    else
    {
        // Get the first one
        auto tensor = std::static_pointer_cast<Tensor>(vcont[0]);
        NVCV_ASSERT(tensor->dtype() == reqs.dtype);
        return tensor;
    }
}

namespace {

NVCVTensorData FillNVCVTensorData(const DLTensor &tensor, std::optional<nvcv::TensorLayout> layout,
                                  NVCVTensorBufferType bufType)
{
    NVCVTensorData tensorData = {};

    // dtype ------------
    tensorData.dtype = py::cast<nvcv::DataType>(ToDType(ToNVCVDataType(tensor.dtype)));

    // layout ------------
    if (layout)
    {
        tensorData.layout = *layout;
    }

    // rank ------------
    {
        int rank = tensor.ndim == 0 ? 1 : tensor.ndim;
        if (rank < 1 || rank > NVCV_TENSOR_MAX_RANK)
        {
            throw std::invalid_argument(util::FormatString("Number of dimensions must be between 1 and %d, not %d",
                                                           NVCV_TENSOR_MAX_RANK, rank));
        }
        tensorData.rank = rank;
    }

    // shape ------------
    std::copy_n(tensor.shape, tensor.ndim, tensorData.shape);

    // buffer type ------------
    if (IsCudaAccessible(tensor.device.device_type))
    {
        tensorData.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    }
    else
    {
        throw std::runtime_error("Only CUDA-accessible tensors are supported for now");
    }

    NVCVTensorBufferStrided &dataStrided = tensorData.buffer.strided;

    // stride ------------
    int elemStrideBytes = (tensor.dtype.bits * tensor.dtype.lanes + 7) / 8;
    for (int d = 0; d < tensor.ndim; ++d)
    {
        dataStrided.strides[d] = tensor.strides[d] * elemStrideBytes;
    }

    // Memory buffer ------------
    dataStrided.basePtr = reinterpret_cast<NVCVByte *>(tensor.data) + tensor.byte_offset;

    return tensorData;
}

NVCVTensorData FillNVCVTensorDataCUDA(const DLTensor &tensor, std::optional<nvcv::TensorLayout> layout)
{
    return FillNVCVTensorData(tensor, std::move(layout), NVCV_TENSOR_BUFFER_STRIDED_CUDA);
}

} // namespace

std::shared_ptr<Tensor> Tensor::Wrap(ExternalBuffer &buffer, std::optional<nvcv::TensorLayout> layout)
{
    const DLTensor &dlTensor = buffer.dlTensor();

    nvcv::TensorDataStridedCuda data{FillNVCVTensorDataCUDA(dlTensor, std::move(layout))};

    // This is the key of a tensor wrapper.
    // All tensor wrappers have the same key.
    Tensor::Key key;
    // We take this opportunity to remove from cache all wrappers that aren't
    // being used. They aren't reusable anyway.
    Cache::Instance().removeAllNotInUseMatching(key);

    auto tensor = std::shared_ptr<Tensor>(new Tensor(data, py::cast(buffer.shared_from_this())));

    // Need to add wrappers to cache so that they don't get destroyed by
    // the cuda stream when they're last used, and python script isn't
    // holding a reference to them. If we don't do it, things might break.
    Cache::Instance().add(*tensor);
    return tensor;
}

std::shared_ptr<Tensor> Tensor::WrapImage(Image &img)
{
    Tensor::Key key;
    Cache::Instance().removeAllNotInUseMatching(key);

    auto tensor = std::shared_ptr<Tensor>(new Tensor(img));

    Cache::Instance().add(*tensor);
    return tensor;
}

Tensor::Tensor(const nvcv::Tensor::Requirements &reqs)
    : m_impl{std::make_unique<nvcv::Tensor>(reqs)}
    , m_key{reqs}
{
}

Tensor::Tensor(const nvcv::ITensorData &data, py::object wrappedObject)
    : m_impl{std::make_unique<nvcv::TensorWrapData>(data)}
    , m_key{}
    , m_wrappedObject(wrappedObject)
{
}

Tensor::Tensor(Image &img)
    : m_impl{std::make_unique<nvcv::TensorWrapImage>(img.impl())}
    , m_key{}
    , m_wrappedObject(py::cast(img))
{
}

std::shared_ptr<Tensor> Tensor::shared_from_this()
{
    return std::static_pointer_cast<Tensor>(Container::shared_from_this());
}

std::shared_ptr<const Tensor> Tensor::shared_from_this() const
{
    return std::static_pointer_cast<const Tensor>(Container::shared_from_this());
}

nvcv::ITensor &Tensor::impl()
{
    return *m_impl;
}

const nvcv::ITensor &Tensor::impl() const
{
    return *m_impl;
}

Shape Tensor::shape() const
{
    return CreateShape(m_impl->shape());
}

std::optional<nvcv::TensorLayout> Tensor::layout() const
{
    const nvcv::TensorLayout &layout = m_impl->layout();
    if (layout != nvcv::TENSOR_NONE)
    {
        return layout;
    }
    else
    {
        return std::nullopt;
    }
}

nvcv::DataType Tensor::dtype() const
{
    return m_impl->dtype();
}

int Tensor::rank() const
{
    return m_impl->rank();
}

Tensor::Key::Key(const nvcv::Tensor::Requirements &reqs)
    : Key(nvcv::TensorShape(reqs.shape, reqs.rank, reqs.layout), static_cast<nvcv::DataType>(reqs.dtype))
{
}

Tensor::Key::Key(const nvcv::TensorShape &shape, nvcv::DataType dtype)
    : m_shape(std::move(shape))
    , m_dtype(dtype)
    , m_wrapper(false)
{
}

size_t Tensor::Key::doGetHash() const
{
    if (m_wrapper)
    {
        return 0; // all wrappers are equal wrt. the cache
    }
    else
    {
        using util::ComputeHash;
        return ComputeHash(m_shape, m_dtype);
    }
}

bool Tensor::Key::doIsEqual(const IKey &that_) const
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
        return std::tie(m_shape, m_dtype) == std::tie(that.m_shape, that.m_dtype);
    }
}

auto Tensor::key() const -> const Key &
{
    return m_key;
}

static py::object ToPython(const nvcv::ITensorData &imgData, py::object owner)
{
    py::object out;

    auto *stridedData = dynamic_cast<const nvcv::ITensorDataStrided *>(&imgData);
    if (!stridedData)
    {
        throw std::runtime_error("Only tensors with pitch-linear data can be exported");
    }

    DLPackTensor dlTensor(*stridedData);
    return ExternalBuffer::Create(std::move(dlTensor), owner);
}

py::object Tensor::cuda() const
{
    const nvcv::ITensorData *tensorData = m_impl->exportData();
    if (!tensorData)
    {
        throw std::runtime_error("Tensor data can't be exported");
    }

    // Note: we can't cache the returned ExternalBuffer because it is holding
    // a reference to us. Doing so would lead to mem leaks.
    return ToPython(*tensorData, py::cast(this->shared_from_this()));
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor)
{
    return out << "<nvcv.Tensor shape=" << tensor.impl().shape()
               << " dtype=" << py::str(py::cast(tensor.dtype())).cast<std::string>() << '>';
}

static std::string TensorLayoutToString(const nvcv::TensorLayout &layout)
{
    std::ostringstream ss;
    ss << layout;
    std::string s = ss.str();

    auto p = s.rfind('_');
    if (p != s.npos)
    {
        return s.substr(p + 1);
    }
    else
    {
        return s;
    }
}

void Tensor::Export(py::module &m)
{
    using namespace py::literals;

    py::class_<nvcv::TensorLayout>(m, "TensorLayout")
        .def(py::init<const char *>())
#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) .def_readonly_static(#LAYOUT, &nvcv::TENSOR_##LAYOUT)
#include <nvcv/TensorLayoutDef.inc>
#undef NVCV_DETAIL_DEF_TLAYOUT
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", &TensorLayoutToString);

    py::implicitly_convertible<py::str, nvcv::TensorLayout>();

    py::class_<Tensor, std::shared_ptr<Tensor>, Container>(m, "Tensor")
        .def(py::init(&Tensor::CreateForImageBatch), "nimages"_a, "imgsize"_a, "format"_a)
        .def(py::init(&Tensor::Create), "shape"_a, "dtype"_a, "layout"_a = std::nullopt)
        .def_property_readonly("layout", &Tensor::layout)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("dtype", &Tensor::dtype)
        // numpy and others use ndim, let's be consistent with them in python.
        // It's not a requirement to be consistent between NVCV Python and C/C++.
        // Each language use whatever is appropriate (and expected) in their environment.
        .def_property_readonly("ndim", &Tensor::rank)
        .def("cuda", &Tensor::cuda)
        .def("__repr__", &util::ToString<Tensor>);

    m.def("as_tensor", &Tensor::Wrap, "buffer"_a, "layout"_a = std::nullopt);
    m.def("as_tensor", &Tensor::WrapImage, "image"_a);
}

} // namespace nvcvpy::priv
