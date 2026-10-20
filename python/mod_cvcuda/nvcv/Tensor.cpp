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
#include "BufferAllocator.hpp"
#include "CastUtils.hpp"
#include "DataType.hpp"
#include "ExternalBuffer.hpp"
#include "Image.hpp"
#include "ImageFormat.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <common/CheckError.hpp>
#include <common/Hash.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <nvcv/TensorShapeInfo.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

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
        = nvcv::Tensor::CalcRequirements(numImages, nvcv::Size2D{size.width, size.height}, fmt,
                                         rowalign == 0 ? nvcv::MemAlignment{} : nvcv::MemAlignment{}.rowAddr(rowalign));
    return CreateFromReqs(reqs);
}

std::shared_ptr<Tensor> Tensor::Create(Shape shape, nvcv::DataType dtype, std::optional<nvcv::TensorLayout> layout,
                                       int rowalign, Stream *consumer)
{
    if (!layout)
    {
        layout = nvcv::TENSOR_NONE;
    }

    nvcv::Tensor::Requirements reqs
        = nvcv::Tensor::CalcRequirements(CreateNVCVTensorShape(shape, *layout), dtype,
                                         rowalign == 0 ? nvcv::MemAlignment{} : nvcv::MemAlignment{}.rowAddr(rowalign));
    return CreateFromReqs(reqs, consumer);
}

std::shared_ptr<Tensor> Tensor::CreateFromReqs(const nvcv::Tensor::Requirements &reqs, Stream *consumer)
{
    const Key              requested{reqs};
    std::optional<int64_t> requiredBytes;
    auto                   getRequiredBytes = [&requiredBytes, &reqs]
    {
        if (!requiredBytes.has_value())
            requiredBytes = doComputeSizeInBytes(reqs);
        return *requiredBytes;
    };
    const auto exact = [](const CacheItem &item, const IKey &requestedKey)
    {
        const auto &tensor = static_cast<const Tensor &>(*item.cachedResource());
        const auto &key    = static_cast<const Key &>(requestedKey);
        return tensor.m_key->matchesDescriptor(key)
            && reinterpret_cast<uintptr_t>(tensor.m_buffer->data.get()) % key.alignment() == 0;
    };
    auto findExact = [&]() -> std::shared_ptr<Tensor>
    {
        if (auto item = Cache::Instance().fetchOne(requested, exact, consumer, {}, getRequiredBytes))
            return std::static_pointer_cast<Tensor>(item->acquireClaimedObject());
        return {};
    };
    if (auto tensor = findExact())
        return tensor;

    auto fits = [&reqs, &getRequiredBytes](Resource &resource, bool pending)
    {
        const auto &tensor = static_cast<const Tensor &>(resource);
        return tensor.allowsCacheReuse(pending) && !pending && tensor.m_size_inbytes >= getRequiredBytes()
            && reinterpret_cast<uintptr_t>(tensor.m_buffer->data.get()) % reqs.alignBytes == 0;
    };
    auto findFit = [&]() -> std::shared_ptr<Tensor>
    {
        auto smaller = [](const CacheItem &a, const CacheItem &b)
        {
            return a.GetSizeInBytes() < b.GetSizeInBytes();
        };
        if (auto item = Cache::Instance().fetchOne(requested, {}, nullptr, fits, getRequiredBytes, smaller))
        {
            auto tensor    = std::static_pointer_cast<Tensor>(item->acquireClaimedObject());
            auto impl      = nvcv::Tensor(reqs, CreateBufferAllocator(tensor->m_buffer));
            tensor->m_impl = std::move(impl);
            tensor->m_key->setDescriptor(reqs);
            return tensor;
        }
        return {};
    };
    if (auto tensor = findFit())
        return tensor;

    // An idle reshape wrapper can be the last native alias of an allocation.
    Cache::Instance().removeAllNotInUseMatching(WrapperKey{});
    if (auto tensor = findExact())
        return tensor;
    if (auto tensor = findFit())
        return tensor;

    std::shared_ptr<Tensor> tensor(new Tensor(reqs)); // NOSONAR: constructor is private.
    Cache::Instance().add(*tensor);
    return tensor;
}

namespace {

NVCVTensorData FillNVCVTensorData(const DLTensor &tensor, std::optional<nvcv::TensorLayout> layout,
                                  NVCVTensorBufferType bufType)
{
    NVCVTensorData tensorData = {};

    // dtype ------------
    std::optional<nvcv::DataType> dtype = NormalizeDataType(ToNVCVDataType(tensor.dtype));
    if (!dtype)
    {
        throw TensorError("Buffer data type is not supported");
    }
    tensorData.dtype = static_cast<NVCVDataType>(*dtype);

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

// Accept a string layout in the first overload pass. Falling back to the
// registered str -> TensorLayout conversion probes the buffer argument again,
// including another potentially expensive __cuda_array_interface__ parse.
using LayoutArg = std::variant<nvcv::TensorLayout, py::str>;

std::optional<nvcv::TensorLayout> ToTensorLayout(std::optional<LayoutArg> layout)
{
    if (!layout)
    {
        return std::nullopt;
    }
    if (const auto *name = std::get_if<py::str>(&*layout))
    {
        std::string nameString = py::cast<std::string>(*name);
        return nvcv::TensorLayout(nameString.c_str());
    }
    return std::get<nvcv::TensorLayout>(*layout);
}

} // namespace

std::shared_ptr<Tensor> Tensor::Wrap(ExternalBuffer &buffer, std::optional<nvcv::TensorLayout> layout)
{
    const DLTensor &dlTensor = buffer.dlTensor();

    nvcv::TensorDataStridedCuda data{FillNVCVTensorDataCUDA(dlTensor, std::move(layout))};

    // This is the key of a tensor wrapper.
    // All tensor wrappers have the same key.
    Tensor::WrapperKey key;
    // We take this opportunity to remove from cache all wrappers that aren't
    // being used. They aren't reusable anyway.
    Cache::Instance().removeAllNotInUseMatching(key);

    auto tensor = std::shared_ptr<Tensor>( // NOSONAR: constructor is private.
        new Tensor(data, py::cast(buffer.shared_from_this())));
#ifdef NB_FREE_THREADED
    tensor->setReadOnly(buffer.readOnly());
    tensor->setSubmissionStates(buffer.submissionStates());
#endif

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
    Tensor::WrapperKey key{img.key()};
    Cache::Instance().removeAllNotInUseMatching(key);

    auto tensor = std::shared_ptr<Tensor>(new Tensor(img)); // NOSONAR: constructor is private.

    Cache::Instance().add(*tensor);
    return tensor;
}

std::shared_ptr<Tensor> Tensor::ReshapeTensor(Tensor &tensor, Shape shape, std::optional<nvcv::TensorLayout> layout)
{
    Tensor::WrapperKey key{tensor.key()};
    Cache::Instance().removeAllNotInUseMatching(key);

    nvcv::Tensor tensor_impl      = tensor.impl();
    auto         new_tensor_shape = CreateNVCVTensorShape(shape, layout ? *layout : tensor_impl.layout());
    nvcv::Tensor new_tensor_impl  = tensor_impl.reshape(new_tensor_shape);
    auto         new_tensor       = std::shared_ptr<Tensor>( // NOSONAR: constructor is private.
        new Tensor(std::move(new_tensor_impl), tensor.key()));
#ifdef NB_FREE_THREADED
    new_tensor->shareSubmissionStates(tensor);
#endif

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
    : Container(false)
    , m_buffer{AllocateBuffer(doComputeSizeInBytes(reqs))}
    , m_impl{reqs, CreateBufferAllocator(m_buffer)}
    , m_key{std::make_unique<Key>(reqs)}
    , m_wrapperKey{*m_key}
    , m_size_inbytes{doComputeSizeInBytes(reqs)}
{
#ifdef NB_FREE_THREADED
    setSubmissionStates({AllocationSubmissionState(m_impl.exportData<nvcv::TensorDataStridedCuda>()->basePtr())});
#endif
}

Tensor::Tensor(const nvcv::TensorData &data, py::object wrappedObject)
    : Container(false)
    , m_impl{nvcv::TensorWrapData(data)}
    , m_size_inbytes{0}
    , m_wrappedObject(wrappedObject)
{
}

Tensor::Tensor(Image &img)
    : Container(false)
    , m_impl{nvcv::TensorWrapImage(img.impl())}
    , m_wrapperKey{img.key()}
    , m_size_inbytes{0}
    , m_wrappedObject(py::cast(&img, py::rv_policy::reference))
{
#ifdef NB_FREE_THREADED
    shareSubmissionStates(img);
#endif
}

Tensor::Tensor(nvcv::Tensor &&tensor, const IKey &deviceKey)
    : Container(false)
    , m_impl{std::move(tensor)}
    , m_wrapperKey{deviceKey}
    , m_size_inbytes{0}
{
}

int64_t Tensor::doComputeSizeInBytes(const nvcv::Tensor::Requirements &reqs)
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

RetainedMemory Tensor::retainedMemory() const
{
    return m_wrappedObject ? RetainedMemory{0, 1} : Container::retainedMemory();
}

bool Tensor::allowsCacheReuse([[maybe_unused]] bool pendingRelease) const
{
    return m_impl.refCount() == 1;
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
    : m_layout(reqs.layout)
    , m_dtype(reqs.dtype)
{
    setDescriptor(reqs);
}

bool Tensor::Key::matchesDescriptor(const Key &requested) const noexcept
{
    return m_rank == requested.m_rank
        && std::equal(m_shape.begin(), m_shape.begin() + m_rank, requested.m_shape.begin())
        && std::equal(m_strides.begin(), m_strides.begin() + m_rank, requested.m_strides.begin());
}

void Tensor::Key::setDescriptor(const nvcv::Tensor::Requirements &reqs) noexcept
{
    m_rank       = reqs.rank;
    m_alignBytes = reqs.alignBytes;
    std::copy_n(reqs.shape, m_rank, m_shape.begin());
    std::copy_n(reqs.strides, m_rank, m_strides.begin());
}

size_t Tensor::Key::doGetHash() const
{
    using util::ComputeHash;
    return ComputeHash(m_layout, m_dtype);
}

bool Tensor::Key::doIsCompatible(const IKey &that_) const
{
    const auto &that = static_cast<const Key &>(that_);
    return std::tie(m_layout, m_dtype) == std::tie(that.m_layout, that.m_dtype);
}

auto Tensor::key() const -> const IKey &
{
    return m_key ? static_cast<const IKey &>(*m_key) : static_cast<const IKey &>(m_wrapperKey);
}

static py::object ToPython(const nvcv::TensorData &tensorData, py::object owner,
                           Resource::ExportStreamState exportStream)
{
    auto stridedData = tensorData.cast<nvcv::TensorDataStrided>();
    if (!stridedData)
    {
        throw TensorError("Only tensors with pitch-linear data can be exported");
    }

    DLPackTensor dlTensor(*stridedData);
    return ExternalBuffer::Create(std::move(dlTensor), std::move(owner), std::move(exportStream.handle),
                                  std::move(exportStream.owner));
}

py::object Tensor::cuda() const
{
    nvcv::TensorData tensorData = m_impl.exportData();

    // Advertise the stream the tensor's data was last written on via the
    // CAI `stream` field so downstream consumers (cupy/torch) can sync.
    ExportStreamState exportStream = this->exportStreamState();

    // Note: we can't cache the returned ExternalBuffer because it is holding
    // a reference to us. Doing so would lead to mem leaks.
    return ToPython(tensorData, py::cast(SharedContainerFrom(*this)), std::move(exportStream));
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor)
{
    return out << "<nvcv.Tensor shape=" << py::cast<std::string>(py::str(tensor.shape()))
               << " dtype=" << py::cast<std::string>(py::str(py::cast(tensor.dtype()))) << '>';
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

void ExportTensorLayout(py::module_ &m)
{
    auto cls = py::class_<nvcv::TensorLayout>(m, "TensorLayout", py::is_weak_referenceable());
    cls.def(py::init<const char *>())
        .def(
            "__eq__", [](const nvcv::TensorLayout &a, const nvcv::TensorLayout &b) { return a == b; },
            py::is_operator(), "Check if two TensorLayout objects are equal.")
        .def(
            "__ne__", [](const nvcv::TensorLayout &a, const nvcv::TensorLayout &b) { return a != b; },
            py::is_operator(), "Check if two TensorLayout objects are not equal.")
        // Kept alongside the tp_hash slot so TensorLayout.__hash__ resolves to
        // the same value; without a def, nanobind nulls __hash__ once __eq__
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

    // Override last: def("__hash__") above set tp_hash to CPython's
    // slot dispatcher; replace it with the direct C implementation.
    auto *tp    = reinterpret_cast<PyTypeObject *>(cls.ptr());
    tp->tp_hash = &TensorLayoutTpHash;
    PyType_Modified(tp);

    py::implicitly_convertible<py::str, nvcv::TensorLayout>();
}

void Tensor::Export(py::module_ &m)
{
    using namespace py::literals;

    py::class_<Tensor, Container>(m, "Tensor", "Tensor", py::is_weak_referenceable())
        .def(py::new_(&Tensor::CreateForImageBatch), "nimages"_a, "imgsize"_a, "format"_a, "rowalign"_a = 0,
             "Create a Tensor object for an ImageBatch.")
        .def(py::new_(+[](Shape shape, nvcv::DataType dtype, std::optional<LayoutArg> layout, int rowalign)
                      { return Tensor::Create(std::move(shape), dtype, ToTensorLayout(std::move(layout)), rowalign); }),
             "shape"_a, "dtype"_a, "layout"_a = std::nullopt, "rowalign"_a = 0,
             "Create a Tensor object with the given shape, data type and layout.")
        .def_prop_ro(
            "layout", [](const Tensor &self) { return PyTensorLayout(self.layout()); },
            "The TensorLayout of the Tensor.")
        .def_prop_ro("shape", &Tensor::shape, "The shape of the Tensor.")
        .def_prop_ro("dtype", &Tensor::dtype, "The data type of the Tensor.")
        // numpy and others use ndim, let's be consistent with them in python.
        // It's not a requirement to be consistent between NVCV Python and C/C++.
        // Each language use whatever is appropriate (and expected) in their environment.
        .def_prop_ro("ndim", &Tensor::rank, "The number of dimensions of the Tensor.")
        .def("cuda", ::cvcudapy::NvtxTrace("cvcuda.Tensor.cuda", &Tensor::cuda),
             "Reference to the Tensor on the CUDA device.")
        .def(
            "reshape",
            [](Tensor &self, Shape shape, std::optional<LayoutArg> layout)
            { return self.Reshape(std::move(shape), ToTensorLayout(std::move(layout))); },
            "shape"_a, "layout"_a = std::nullopt,
            "Produces a tensor pointing to the same data but with a new shape and layout.")
        .def("__repr__", &util::ToString<Tensor>, "Return the string representation of the Tensor object.");

    m.def("as_tensor",
          ::cvcudapy::NvtxTrace(
              "cvcuda.as_tensor", +[](ExternalBuffer &buffer, std::optional<LayoutArg> layout)
                                  { return Tensor::Wrap(buffer, ToTensorLayout(std::move(layout))); }),
          "buffer"_a, "layout"_a = std::nullopt, "Wrap an existing buffer into a Tensor object with the given layout.");
    m.def("as_tensor", ::cvcudapy::NvtxTrace("cvcuda.as_tensor", &Tensor::WrapImage), "image"_a,
          "Wrap an existing image into a Tensor object.");
    m.def("reshape",
          ::cvcudapy::NvtxTrace(
              "cvcuda.reshape",
              +[](Tensor &self, Shape shape, std::optional<LayoutArg> layout)
              { return Tensor::ReshapeTensor(self, std::move(shape), ToTensorLayout(std::move(layout))); }),
          "tensor"_a, "shape"_a, "layout"_a = std::nullopt,
          "Produces a tensor pointing to the same data but with a new shape and layout.");
}

} // namespace nvcvpy::priv
