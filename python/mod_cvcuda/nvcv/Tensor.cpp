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

#include "Tensor.hpp"

#include "../NvtxRange.hpp"
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

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace nvcv {

static size_t ComputeHash(const nvcv::TensorShape &shape)
{
    using nvcvpy::util::ComputeHash;
    return ComputeHash(shape.shape(), shape.layout());
}

} // namespace nvcv

namespace nvcvpy::priv {

namespace {

class TensorError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace

std::shared_ptr<Tensor> Tensor::CreateForImageBatch(int numImages, const Size2D &size, nvcv::ImageFormat fmt,
                                                    int rowalign)
{
    nvcv::Tensor::Requirements reqs
        = nvcv::Tensor::CalcRequirements(numImages, nvcv::Size2D{std::get<0>(size), std::get<1>(size)}, fmt,
                                         rowalign == 0 ? nvcv::MemAlignment{} : nvcv::MemAlignment{}.rowAddr(rowalign));
    return CreateFromReqs(reqs);
}

std::shared_ptr<Tensor> Tensor::Create(Shape shape, nvcv::DataType dtype, std::optional<nvcv::TensorLayout> layout,
                                       int rowalign)
{
    if (!layout)
    {
        layout = nvcv::TENSOR_NONE;
    }

    nvcv::Tensor::Requirements reqs
        = nvcv::Tensor::CalcRequirements(CreateNVCVTensorShape(shape, *layout), dtype,
                                         rowalign == 0 ? nvcv::MemAlignment{} : nvcv::MemAlignment{}.rowAddr(rowalign));
    return CreateFromReqs(reqs);
}

std::shared_ptr<Tensor> Tensor::CreateFromReqs(const nvcv::Tensor::Requirements &reqs)
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{reqs});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Tensor> tensor(new Tensor(reqs)); // NOSONAR: constructor is private.
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
    tensorData.dtype = static_cast<NVCVDataType>(py::cast<nvcv::DataType>(ToDType(ToNVCVDataType(tensor.dtype))));

    // layout ------------
    if (layout)
    {
        tensorData.layout = static_cast<const NVCVTensorLayout &>(*layout);
    }

    // rank ------------
    {
        // REVISIT: Add 0D support
        int rank = tensor.ndim == 0 ? 1 : tensor.ndim;
        if (rank < 1 || rank > NVCV_TENSOR_MAX_RANK)
        {
            throw std::invalid_argument(util::ConcatString("Number of dimensions must be between 1 and ",
                                                           NVCV_TENSOR_MAX_RANK, ", not ", rank));
        }
        tensorData.rank = rank;
    }

    // shape ------------
    std::copy_n(tensor.shape, tensor.ndim, tensorData.shape);

    // buffer type ------------
    if (IsCudaAccessible(tensor.device.device_type))
    {
        tensorData.bufferType = bufType;
    }
    else
    {
        throw TensorError("Only CUDA-accessible tensors are supported for now");
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

    auto tensor = std::shared_ptr<Tensor>( // NOSONAR: constructor is private.
        new Tensor(data, py::cast(buffer.shared_from_this())));

    // Seed the tensor's Resource state with the producer's CUDA stream so the
    // first cvcuda op that reads this tensor inserts the necessary
    // cross-stream wait.  Skipped when the producer advertised
    // `stream: None` / `stream: -1`, which means they've already synchronized.
    if (!buffer.producerIsSynced() && buffer.producerStream() != nullptr)
    {
        int device = buffer.producerDevice();
        if (device < 0)
        {
            util::CheckThrow(cudaGetDevice(&device));
        }
        tensor->seedLastStream(buffer.producerStream(), device);
    }

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

    auto tensor = std::shared_ptr<Tensor>(new Tensor(img)); // NOSONAR: constructor is private.

    Cache::Instance().add(*tensor);
    return tensor;
}

std::shared_ptr<Tensor> Tensor::ReshapeTensor(Tensor &tensor, Shape shape, std::optional<nvcv::TensorLayout> layout)
{
    Tensor::Key key;
    Cache::Instance().removeAllNotInUseMatching(key);

    nvcv::Tensor tensor_impl      = tensor.impl();
    auto         new_tensor_shape = CreateNVCVTensorShape(shape, layout ? *layout : tensor_impl.layout());
    nvcv::Tensor new_tensor_impl  = tensor_impl.reshape(new_tensor_shape);
    auto         new_tensor
        = std::shared_ptr<Tensor>(new Tensor(std::move(new_tensor_impl))); // NOSONAR: constructor is private.

    // Need to add wrappers to cache so that they don't get destroyed by
    // the cuda stream when they're last used, and python script isn't
    // holding a reference to them. If we don't do it, things might break.
    Cache::Instance().add(*new_tensor);
    return new_tensor;
}

std::shared_ptr<Tensor> Tensor::Reshape(Shape shape, std::optional<nvcv::TensorLayout> layout)
{
    return ReshapeTensor(*this, std::move(shape), std::move(layout));
}

Tensor::Tensor(const nvcv::Tensor::Requirements &reqs)
    : m_impl{reqs}
    , m_key{reqs}
    , m_size_inbytes{doComputeSizeInBytes(reqs)}
{
}

Tensor::Tensor(const nvcv::TensorData &data, py::object wrappedObject)
    : m_impl{nvcv::TensorWrapData(data)}
    , m_size_inbytes{doComputeSizeInBytes(nvcv::Tensor::Requirements())}
    , m_wrappedObject(wrappedObject)
{
}

Tensor::Tensor(Image &img)
    : m_impl{nvcv::TensorWrapImage(img.impl())}
    , m_size_inbytes{doComputeSizeInBytes(nvcv::Tensor::Requirements())}
    , m_wrappedObject(py::cast(img))
{
}

Tensor::Tensor(nvcv::Tensor &&tensor)
    : m_impl{std::move(tensor)}
    , m_size_inbytes{doComputeSizeInBytes(nvcv::Tensor::Requirements())}
{
}

int64_t Tensor::doComputeSizeInBytes(const nvcv::Tensor::Requirements &reqs) const
{
    int64_t size_inbytes;
    util::CheckThrow(nvcvMemRequirementsCalcTotalSizeBytes(&(reqs.mem.cudaMem), &size_inbytes));
    return size_inbytes;
}

int64_t Tensor::GetSizeInBytes() const
{
    // m_size_inbytes == -1 indicates failure case and value has not been computed yet
    NVCV_ASSERT(m_size_inbytes != -1
                && "Tensor has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
    return m_size_inbytes;
}

nvcv::Tensor &Tensor::impl()
{
    return m_impl;
}

const nvcv::Tensor &Tensor::impl() const
{
    return m_impl;
}

Shape Tensor::shape() const
{
    return CreateShape(m_impl.shape());
}

std::optional<nvcv::TensorLayout> Tensor::layout() const
{
    const nvcv::TensorLayout &layout = m_impl.layout();
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
    return m_impl.dtype();
}

int Tensor::rank() const
{
    return m_impl.rank();
}

Tensor::Key::Key(const nvcv::Tensor::Requirements &reqs)
    : Key(nvcv::TensorShape(reqs.shape, reqs.rank, nvcv::TensorLayout{reqs.layout}),
          static_cast<nvcv::DataType>(reqs.dtype))
{
}

Tensor::Key::Key(const nvcv::TensorShape &shape, nvcv::DataType dtype)
    : m_shape(shape)
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

bool Tensor::Key::doIsCompatible(const IKey &that_) const
{
    const auto &that = static_cast<const Key &>(that_);

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

static py::object ToPython(const nvcv::TensorData &tensorData, py::object owner, cudaStream_t exportStream,
                           bool setExportStream)
{
    py::object out;

    auto stridedData = tensorData.cast<nvcv::TensorDataStrided>();
    if (!stridedData)
    {
        throw TensorError("Only tensors with pitch-linear data can be exported");
    }

    DLPackTensor dlTensor(*stridedData);
    return ExternalBuffer::Create(std::move(dlTensor), owner, exportStream, setExportStream);
}

py::object Tensor::cuda() const
{
    nvcv::TensorData tensorData = m_impl.exportData();

    // Advertise the stream the tensor's data was last written on via the
    // CAI `stream` field so downstream consumers (cupy/torch) can sync.
    // If the tensor has never been used by a cvcuda op, getLastStreamHandle()
    // returns 0 -- in which case we fall back to the default "stream: 1"
    // (legacy default) by not populating the export stream.
    cudaStream_t lastStream = this->getLastStreamHandle();
    bool         setStream  = lastStream != nullptr;

    // Note: we can't cache the returned ExternalBuffer because it is holding
    // a reference to us. Doing so would lead to mem leaks.
    return ToPython(tensorData, py::cast(SharedContainerFrom(*this)), lastStream, setStream);
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor)
{
    return out << "<nvcv.Tensor shape=" << tensor.shape()
               << " dtype=" << py::str(py::cast(tensor.dtype())).cast<std::string>() << '>';
}

static std::string TensorLayoutToString(const nvcv::TensorLayout &layout)
{
    std::ostringstream ss;
    ss << layout;
    std::string s = ss.str();

    auto p = s.rfind('_');
    if (p != std::string::npos)
    {
        return s.substr(p + 1);
    }
    else
    {
        return s;
    }
}

static std::string_view TensorLayoutLabels(const nvcv::TensorLayout &layout)
{
    return std::string_view(layout.m_layout.data, layout.rank());
}

// Named-layout Python objects, keyed by their labels and shared for the
// process lifetime (never destroyed: py::object statics must not outlive the
// interpreter). Tensor.layout returns these instead of constructing a new
// Python object per access, so repeated reads yield the identical object and
// dict lookups keyed by the class constants hit CPython's identity shortcut.
static std::unordered_map<std::string, py::object> &InternedTensorLayouts()
{
    static auto *interned = new std::unordered_map<std::string, py::object>; // NOSONAR: deliberately immortal
    return *interned;
}

static py::object PyTensorLayout(const std::optional<nvcv::TensorLayout> &layout)
{
    if (!layout)
    {
        return py::none();
    }
    auto &interned = InternedTensorLayouts();
    if (auto it = interned.find(std::string(TensorLayoutLabels(*layout))); it != interned.end())
    {
        return it->second;
    }
    return py::cast(*layout);
}

// Hash over the raw dimension labels so it stays consistent with __eq__
// (which compares label data): equal layouts always hash equal, whether built
// from a named constant or a string. std::hash<std::string_view> is required
// to match std::hash<std::string> for equal characters.
static Py_hash_t TensorLayoutHashValue(const nvcv::TensorLayout &l)
{
    auto h = static_cast<Py_hash_t>(std::hash<std::string_view>{}(TensorLayoutLabels(l)));
    return h == -1 ? -2 : h; // CPython reserves -1 for errors
}

// Installed directly as tp_hash: dict/set operations then skip the Python
// method-dispatch of a def("__hash__"), which costs several times the hash
// itself on this hot path.
static Py_hash_t TensorLayoutTpHash(PyObject *self)
{
    try
    {
        return TensorLayoutHashValue(py::cast<nvcv::TensorLayout &>(py::handle(self)));
    }
    catch (const std::exception &e) // NOSONAR: tp_hash is a C slot; no C++ exception may reach CPython
    {
        PyErr_SetString(PyExc_TypeError, e.what());
        return -1;
    }
}

void ExportTensorLayout(py::module &m)
{
    auto cls = py::class_<nvcv::TensorLayout>(m, "TensorLayout");
    cls.def(py::init<const char *>())
        .def(
            "__eq__", [](const nvcv::TensorLayout &a, const nvcv::TensorLayout &b) { return a == b; },
            py::is_operator(), "Check if two TensorLayout objects are equal.")
        .def(
            "__ne__", [](const nvcv::TensorLayout &a, const nvcv::TensorLayout &b) { return a != b; },
            py::is_operator(), "Check if two TensorLayout objects are not equal.")
        // Kept alongside the tp_hash slot so TensorLayout.__hash__ resolves to
        // the same value; without a def, pybind11 nulls __hash__ once __eq__
        // is defined, leaving TensorLayout unhashable.
        .def(
            "__hash__", [](const nvcv::TensorLayout &l) { return TensorLayoutHashValue(l); },
            "Return a value-based hash so TensorLayout can be used as a dict key or set member.")
        .def("__repr__", &TensorLayoutToString, "Return the string representation of the TensorLayout object.");

    // The class constants are the interned instances themselves (not
    // per-access getters), so `TensorLayout.NHWC is tensor.layout` holds.
    auto intern = [&cls](const char *name, const nvcv::TensorLayout &layout)
    {
        py::object obj = py::cast(layout);
        InternedTensorLayouts().try_emplace(std::string(TensorLayoutLabels(layout)), obj);
        cls.attr(name) = obj;
    };
    cls.attr("NONE") = py::cast(nvcv::TENSOR_NONE);
#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) intern(#LAYOUT, nvcv::TENSOR_##LAYOUT);
#include <nvcv/TensorLayoutDef.inc> // NOSONAR: this include expands TensorLayout constants inside the binding chain.
#undef NVCV_DETAIL_DEF_TLAYOUT

    // Override last: pybind's def("__hash__") above set tp_hash to CPython's
    // slot dispatcher; replace it with the direct C implementation.
    auto *tp    = reinterpret_cast<PyTypeObject *>(cls.ptr());
    tp->tp_hash = &TensorLayoutTpHash;
    PyType_Modified(tp);

    py::implicitly_convertible<py::str, nvcv::TensorLayout>();
}

void Tensor::Export(py::module &m)
{
    using namespace py::literals;

    py::class_<Tensor, std::shared_ptr<Tensor>, Container>(m, "Tensor", "Tensor")
        .def(py::init(&Tensor::CreateForImageBatch), "nimages"_a, "imgsize"_a, "format"_a, "rowalign"_a = 0,
             "Create a Tensor object for an ImageBatch.")
        .def(py::init(&Tensor::Create), "shape"_a, "dtype"_a, "layout"_a = std::nullopt, "rowalign"_a = 0,
             "Create a Tensor object with the given shape, data type and layout.")
        .def_property_readonly(
            "layout", [](const Tensor &self) { return PyTensorLayout(self.layout()); },
            "The TensorLayout of the Tensor.")
        .def_property_readonly("shape", &Tensor::shape, "The shape of the Tensor.")
        .def_property_readonly("dtype", &Tensor::dtype, "The data type of the Tensor.")
        // numpy and others use ndim, let's be consistent with them in python.
        // It's not a requirement to be consistent between NVCV Python and C/C++.
        // Each language use whatever is appropriate (and expected) in their environment.
        .def_property_readonly("ndim", &Tensor::rank, "The number of dimensions of the Tensor.")
        .def("cuda", ::cvcudapy::NvtxTrace("cvcuda.Tensor.cuda", &Tensor::cuda),
             "Reference to the Tensor on the CUDA device.")
        .def("reshape", &Tensor::Reshape, "shape"_a, "layout"_a = std::nullopt,
             "Produces a tensor pointing to the same data but with a new shape and layout.")
        .def("__repr__", &util::ToString<Tensor>, "Return the string representation of the Tensor object.");

    m.def("as_tensor", ::cvcudapy::NvtxTrace("cvcuda.as_tensor", &Tensor::Wrap), "buffer"_a, "layout"_a = std::nullopt,
          "Wrap an existing buffer into a Tensor object with the given layout.");
    m.def("as_tensor", ::cvcudapy::NvtxTrace("cvcuda.as_tensor", &Tensor::WrapImage), "image"_a,
          "Wrap an existing image into a Tensor object.");
    m.def("reshape", ::cvcudapy::NvtxTrace("cvcuda.reshape", &Tensor::ReshapeTensor), "tensor"_a, "shape"_a,
          "layout"_a = std::nullopt, "Produces a tensor pointing to the same data but with a new shape and layout.");
}

} // namespace nvcvpy::priv
