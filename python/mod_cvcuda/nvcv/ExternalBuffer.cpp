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

#include "ExternalBuffer.hpp"

#include "DataType.hpp"
#include "Resource.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional> // for std::multiplies
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace nvcvpy::priv {

using namespace py::literals;

namespace {

class ExternalBufferError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

py::tuple CheckedTuple(Py_ssize_t size)
{
    PyObject *tuple // NOSONAR: the pointer is handed to nanobind's steal, which requires a non-const PyObject*.
        = PyTuple_New(size);
    if (tuple == nullptr)
    {
        py::raise_python_error();
    }
    return py::steal<py::tuple>(tuple);
}

} // namespace

static std::byte *AsBytePointer(std::uintptr_t address) noexcept
{
    return reinterpret_cast<std::byte *>(address);
}

static int CheckValidCUDABuffer(const std::byte *ptr)
{
    if (ptr == nullptr)
    {
        throw ExternalBufferError("NULL CUDA buffer not accepted");
    }

    cudaPointerAttributes attrs = {};
    cudaError_t           err   = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError(); // reset the cuda error (if any)
    if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered)
    {
        throw ExternalBufferError("Buffer is not CUDA-accessible");
    }
    return attrs.device;
}

// Query the CUDA device a pointer lives on.  Returns -1 if the attributes
// can't be obtained (e.g. null pointer); callers then fall back to the
// current device.
static int GetCudaDeviceForPtr(const std::byte *ptr)
{
    if (ptr == nullptr)
    {
        return -1;
    }

    cudaPointerAttributes attrs = {};
    cudaError_t           err   = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError();
    if (err != cudaSuccess)
    {
        return -1;
    }
    return attrs.device;
}

// Parse the CAI `stream` field per the v3 spec:
//   absent / field missing : legacy default stream implied (producer was
//                            on stream 0)  --> handle = cudaStreamLegacy,
//                            isSynced = false
//   None / -1              : producer has synchronized; no wait needed
//                            --> handle = 0, isSynced = true
//   0                      : disallowed by spec but cupy emits this for
//                            its null stream; treat as legacy default
//                            --> handle = cudaStreamLegacy, isSynced = false
//   1                      : legacy default stream
//                            --> handle = cudaStreamLegacy
//   2                      : per-thread default stream
//                            --> handle = cudaStreamPerThread
//   other positive int     : raw stream handle
static void ParseCAIStreamField(const py::dict &iface, cudaStream_t &outHandle, bool &outIsSynced)
{
    outHandle   = reinterpret_cast<cudaStream_t>(1); // cudaStreamLegacy
    outIsSynced = false;

    if (!iface.contains("stream"))
    {
        return; // legacy default
    }

    py::object sf = iface["stream"];
    if (sf.is_none())
    {
        outHandle   = nullptr;
        outIsSynced = true;
        return;
    }

    long value = 0;
    try
    {
        value = py::cast<long>(sf);
    }
    catch (py::cast_error &)
    {
        // Unparseable stream field; conservative default.
        return;
    }

    if (value == -1)
    {
        outHandle   = nullptr;
        outIsSynced = true;
        return;
    }
    if (value == 0 || value == 1)
    {
        outHandle = reinterpret_cast<cudaStream_t>(1); // cudaStreamLegacy
        return;
    }
    if (value == 2)
    {
        outHandle = reinterpret_cast<cudaStream_t>(2); // cudaStreamPerThread
        return;
    }
    outHandle = reinterpret_cast<cudaStream_t>(value);
}

// Parse the DLPack v1 `stream` kwarg to `__dlpack__`:
//   None / 0 / -1        : no synchronization requested; producer should
//                          not insert a cross-stream wait
//                          --> outConsumerStream = 0, outNeedSync = false
//   1                    : legacy default stream
//                          --> outConsumerStream = cudaStreamLegacy,
//                              outNeedSync = true
//   2                    : per-thread default stream
//                          --> outConsumerStream = cudaStreamPerThread,
//                              outNeedSync = true
//   other int            : consumer's CUDA stream handle
//                          --> outConsumerStream = handle, outNeedSync = true
//
// See https://dmlc.github.io/dlpack/latest/python_spec.html.  Note that
// None/0/-1 all map to "no sync" — the consumer is asserting they'll
// handle synchronization themselves (or the data is already retired).
static void ParseDLPackStreamArg(py::handle streamArg, cudaStream_t &outConsumerStream, bool &outNeedSync)
{
    outConsumerStream = nullptr;
    outNeedSync       = false;

    if (streamArg.is_none())
    {
        return; // no sync
    }

    long value = 0;
    try
    {
        value = py::cast<long>(streamArg);
    }
    catch (py::cast_error &)
    {
        // Unparseable stream arg; conservative: no sync, consumer is on
        // their own.  This also covers edge cases like callers passing a
        // cupy.cuda.Stream object directly (which nanobind may not cast).
        return;
    }

    if (value == 0 || value == -1)
    {
        return; // no sync
    }
    if (value == 1)
    {
        outConsumerStream = reinterpret_cast<cudaStream_t>(1); // cudaStreamLegacy
        outNeedSync       = true;
        return;
    }
    if (value == 2)
    {
        outConsumerStream = reinterpret_cast<cudaStream_t>(2); // cudaStreamPerThread
        outNeedSync       = true;
        return;
    }
    outConsumerStream = reinterpret_cast<cudaStream_t>(value);
    outNeedSync       = true;
}

namespace {
bool PythonRuntimeAvailable() noexcept
{
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsInitialized() && !Py_IsFinalizing();
#else
    return Py_IsInitialized() && !_Py_IsFinalizing();
#endif
}

class ScopedCudaDevice
{
public:
    explicit ScopedCudaDevice(int device)
    {
        if (device < 0)
        {
            return;
        }

        util::CheckThrow(cudaGetDevice(&m_previous));
        if (device != m_previous)
        {
            util::CheckThrow(cudaSetDevice(device));
            m_restore = true;
        }
    }

    ScopedCudaDevice(const ScopedCudaDevice &)            = delete;
    ScopedCudaDevice &operator=(const ScopedCudaDevice &) = delete;

    ~ScopedCudaDevice() noexcept
    {
        if (m_restore)
        {
            cudaError_t status = cudaSetDevice(m_previous);
            util::CheckLog(status);
            if (status != cudaSuccess)
            {
                (void)cudaGetLastError();
            }
        }
    }

private:
    int  m_previous = -1;
    bool m_restore  = false;
};

struct VersionedManagerCtx
{
    DLManagedTensorVersioned              tensor;
    std::shared_ptr<const ExternalBuffer> extBuffer;
};

struct ManagerCtx
{
    DLManagedTensor                       tensor;
    std::shared_ptr<const ExternalBuffer> extBuffer;
};

void DeleteShapeAndStrides(DLManagedTensor *self) noexcept
{
    std::unique_ptr<int64_t[]> shape(self->dl_tensor.shape);
    std::unique_ptr<int64_t[]> strides(self->dl_tensor.strides);
    self->dl_tensor.shape   = nullptr;
    self->dl_tensor.strides = nullptr;
}

DLPackTensor CreateCAITensor()
{
    DLManagedTensor dlManagedTensor = {};
    dlManagedTensor.deleter         = DeleteShapeAndStrides;
    return DLPackTensor{std::move(dlManagedTensor)};
}

template<class Context, class ManagedTensor>
void DeleteManagerContext(ManagedTensor *tensor) noexcept
{
    auto *raw = static_cast<Context *>(tensor->manager_ctx);
    if (raw == nullptr || !PythonRuntimeAvailable())
    {
        // Contexts retain Python objects through ExternalBuffer. If Python is
        // finalizing, leaking is safer than attaching to a torn-down runtime.
        return;
    }

    try
    {
        if (util::HasPythonThreadState())
        {
            delete raw; // NOSONAR: the DLPack callback owns this context; keep direct destruction on the hot path.
            return;
        }
        py::gil_scoped_acquire   gil;
        std::unique_ptr<Context> ctx(raw);
    }
    catch (...)
    {
        // DLPack deleters are C callbacks and must not throw. If acquiring the
        // GIL failed, leave the context alive rather than decref off-GIL.
        std::fputs("[cvcuda] DLPack cleanup could not acquire the GIL; leaking its context\n", stderr);
    }
}

void DeleteVersionedManager(DLManagedTensorVersioned *tensor) noexcept
{
    DeleteManagerContext<VersionedManagerCtx>(tensor);
}

void DeleteManager(DLManagedTensor *tensor) noexcept
{
    DeleteManagerContext<ManagerCtx>(tensor);
}

void ReleaseVersionedCapsule(PyObject *ptr)
{
    if (!PyCapsule_IsValid(ptr, "dltensor_versioned"))
    {
        return;
    }

    auto *vt = static_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(ptr, "dltensor_versioned"));
    if (vt != nullptr && vt->deleter != nullptr)
    {
        vt->deleter(vt);
    }
}

void ReleaseLegacyCapsule(PyObject *ptr)
{
    if (!PyCapsule_IsValid(ptr, "dltensor"))
    {
        return;
    }

    auto *dlTensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(ptr, "dltensor"));
    if (dlTensor != nullptr && dlTensor->deleter != nullptr)
    {
        dlTensor->deleter(dlTensor);
    }
}

bool UseVersionedDLPack(py::handle maxVersion)
{
    if (maxVersion.is_none())
    {
        return false;
    }

    if (!PyTuple_Check(maxVersion.ptr()))
        throw py::cast_error();
    // The call owns this immutable tuple; inspecting it needs no extra owner.
    return PyTuple_GET_SIZE(maxVersion.ptr()) >= 1
        && py::cast<int>(py::handle(PyTuple_GET_ITEM(maxVersion.ptr(), 0))) >= 1;
}

py::capsule CreateVersionedDLPackCapsule(const DLTensor &dlTensor, std::shared_ptr<const ExternalBuffer> extBuffer)
{
    auto ctx = std::make_unique<VersionedManagerCtx>();

    ctx->tensor.version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
#ifdef NB_FREE_THREADED
    ctx->tensor.flags = extBuffer->readOnly() ? DLPACK_FLAG_BITMASK_READ_ONLY : 0;
#else
    ctx->tensor.flags = 0;
#endif
    ctx->tensor.manager_ctx = ctx.get();
    ctx->tensor.deleter     = DeleteVersionedManager;
    ctx->tensor.dl_tensor   = dlTensor;
    ctx->extBuffer          = std::move(extBuffer);

    py::capsule cap
        = py::steal<py::capsule>(PyCapsule_New(&ctx->tensor, "dltensor_versioned", ReleaseVersionedCapsule));
    ctx.release();
    return cap;
}

py::capsule CreateLegacyDLPackCapsule(const DLTensor &dlTensor, std::shared_ptr<const ExternalBuffer> extBuffer)
{
    auto ctx = std::make_unique<ManagerCtx>();

    ctx->tensor.manager_ctx = ctx.get();
    ctx->tensor.deleter     = DeleteManager;
    ctx->tensor.dl_tensor   = dlTensor;
    ctx->extBuffer          = std::move(extBuffer);

    py::capsule cap = py::steal<py::capsule>(PyCapsule_New(&ctx->tensor, "dltensor", ReleaseLegacyCapsule));
    ctx.release();
    return cap;
}

// Insert a cross-stream synchronization into the consumer's stream before
// returning a DLPack capsule that exposes writer-stream-pending data.
//
// Each wait captures its recording, so later exports can re-record the same
// event without changing earlier waits. The buffer retains it until destruction.
#ifndef NB_FREE_THREADED
void InsertDLPackStreamSync(cudaStream_t writerStream, cudaStream_t consumerStream, int device, cudaEvent_t &event)
{
    if (writerStream == consumerStream)
    {
        return;
    }

    ScopedCudaDevice deviceGuard(device);
    if (!event)
        util::CheckThrow(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    util::CheckThrow(cudaEventRecord(event, writerStream));
    util::CheckThrow(cudaStreamWaitEvent(consumerStream, event, 0));
}
#endif

std::string ToFormatString(const DLDataType &dtype)
{
    // Scalar CAI spellings need no intermediate NumPy dtype. Keep the existing
    // conversion for vector and uncommon types, including their error behavior.
    char kind = '\0';
    if (dtype.lanes == 1)
    {
        switch (dtype.code)
        {
        case kDLInt:
        case kDLUInt:
            if (dtype.bits == 8 || dtype.bits == 16 || dtype.bits == 32 || dtype.bits == 64)
                kind = dtype.code == kDLInt ? 'i' : 'u';
            break;
        case kDLFloat:
            if (dtype.bits == 16 || dtype.bits == 32 || dtype.bits == 64)
                kind = 'f';
            break;
        case kDLComplex:
            if (dtype.bits == 64 || dtype.bits == 128)
                kind = 'c';
            break;
        default:
            break;
        }
    }
    if (kind != '\0')
    {
        const uint16_t nativeOrder = 1;
        char           order       = '|';
        if (dtype.bits != 8)
            order = *reinterpret_cast<const std::byte *>(&nativeOrder) != std::byte{0} ? '<' : '>';
        return std::string{order, kind} + std::to_string(dtype.bits / 8);
    }
    py::numpy_dtype dt = ToDType(ToNVCVDataType(dtype));
    return py::cast<std::string>(dt.attr("str"));
}

std::unique_ptr<int64_t[]> ParseCAIShape(const py::tuple &shape)
{
    auto shapeData = std::make_unique<int64_t[]>(shape.size());

    for (size_t i = 0; i < shape.size(); ++i)
    {
        shapeData[i] = py::cast<long>(shape[i]);
    }

    return shapeData;
}

std::unique_ptr<int64_t[]> ParseCAIStrides(const py::dict &iface, int ndim, const int64_t *shape, int64_t itemSize)
{
    auto stridesData = std::make_unique<int64_t[]>(ndim);

    if (iface.contains("strides") && !iface["strides"].is_none())
    {
        py::tuple strides = py::cast<py::tuple>(iface["strides"]);
        for (int i = 0; i < ndim; ++i)
        {
            int64_t strideBytes = py::cast<long>(strides[i]);
            if (strideBytes % itemSize != 0)
            {
                throw ExternalBufferError("Stride must be a multiple of the element size in bytes");
            }
            stridesData[i] = strideBytes / itemSize;
        }
        return stridesData;
    }

    if (ndim <= 0)
    {
        return stridesData;
    }

    // If strides isn't defined, according to cuda array interface, we must
    // set them up for packed, row-major strides.
    stridesData[ndim - 1] = 1;
    for (int i = ndim - 1; i > 0; --i)
    {
        stridesData[i - 1] = stridesData[i] * shape[i];
    }
    return stridesData;
}

bool HasRequiredCAIFields(const py::dict &iface)
{
    return iface.contains("shape") && iface.contains("typestr") && iface.contains("data") && iface.contains("version");
}

struct CAIType
{
    DLDataType dtype;
    int        itemSize;
};

// Parse a CAI typestr without allocating a numpy dtype. Supported kinds and
// widths come from the same policy used by ToNVCVDataType(py::dtype).
std::optional<CAIType> ParseCAITypestr(std::string_view typestr)
{
    if (typestr.empty())
    {
        return std::nullopt;
    }

    // The byte-order character is optional: numpy accepts a bare "u1", and so
    // did the lookup this replaces.
    size_t kindPos   = 0;
    bool   bigEndian = false;
    switch (typestr[0])
    {
    case '>':
        bigEndian = true;
        [[fallthrough]];
    case '|':
    case '=':
    case '<':
        kindPos = 1;
        break;
    default:
        break;
    }

    if (typestr.size() < kindPos + 2)
    {
        return std::nullopt;
    }

    int         itemSize = 0;
    const char *first    = typestr.data() + kindPos + 1;
    const char *last     = typestr.data() + typestr.size();
    auto [end, error]    = std::from_chars(first, last, itemSize);
    if (error != std::errc{} || end != last)
    {
        return std::nullopt;
    }

    // Byte order only matters for multi-byte values. A big-endian buffer wider
    // than one byte would need swapping, which is not done anywhere here, so
    // reject it rather than read it as native.
    if (bigEndian && itemSize > 1)
    {
        return std::nullopt;
    }

    std::optional<nvcv::DataType> dataType = ToNVCVDataType(typestr[kindPos], itemSize);
    if (!dataType)
    {
        return std::nullopt;
    }

    return CAIType{ToDLDataType(dataType->dataKind(), itemSize * 8, 1), itemSize};
}

py::capsule RequestDLPackCapsule(const py::object &object)
{
    try
    {
        return py::cast<py::capsule>(object.attr("__dlpack__")("stream"_a = 1, "max_version"_a = py::make_tuple(1, 0)));
    }
    catch (const py::error_already_set &error)
    {
        // A v0 producer rejects the new keyword before entering its exporter.
        // CPython and nanobind format that dispatch error differently. Do not
        // retry an exporter that ran and failed for its own reason.
        const std::string_view message = error.what();
        if (const bool unsupportedKeyword
            = message.find("max_version") != std::string_view::npos
           && (message.find("unexpected keyword argument") != std::string_view::npos
               || (message.find("incompatible function arguments") != std::string_view::npos
                   && message.find("kwargs:") != std::string_view::npos));
            !error.matches(PyExc_TypeError) || !unsupportedKeyword)
        {
            throw;
        }
        return py::cast<py::capsule>(object.attr("__dlpack__")("stream"_a = 1));
    }
}

} // namespace

py::object ExternalBuffer::Create(DLPackTensor &&dlPackTensor, py::object wrappedObj,
                                  std::optional<cudaStream_t>              exportStream,
                                  std::shared_ptr<const StreamHandleOwner> exportStreamOwner)
{
    auto buf = std::make_shared<ExternalBuffer>(std::move(dlPackTensor));
#ifdef NB_FREE_THREADED
    Resource *owner = nullptr;
    if (py::try_cast(wrappedObj, owner) && owner)
    {
        buf->m_readOnly         = owner->readOnly();
        buf->m_submissionStates = owner->submissionStates();
    }
#endif

    if (exportStream.has_value())
    {
        buf->setExportStream(*exportStream, exportStreamOwner);
    }

    // Keep the allocation owner in the native object, not only in the Python
    // wrapper. DLPack consumers retain this shared_ptr after that wrapper dies.
    buf->m_wrappedObj = std::move(wrappedObj);

    py::object o = py::cast(buf);

    // If we expose cuda array interface,
#ifndef NB_FREE_THREADED
    if (auto iface = buf->cudaArrayInterface())
    {
        o.attr("__cuda_array_interface__") = *iface;
    }
#endif

    return o;
}

void ExternalBuffer::ReleaseDLPackCapsuleOffGil(py::capsule capsule)
{
    if (PyCapsule_IsValid(capsule.ptr(), "dltensor_versioned"))
    {
        auto *tensor
            = static_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(capsule.ptr(), "dltensor_versioned"));
        if (tensor == nullptr)
        {
            throw py::error_already_set();
        }
        if (PyCapsule_SetName(capsule.ptr(), "used_dltensor_versioned") != 0)
        {
            throw py::error_already_set();
        }
        if (tensor->deleter != nullptr)
        {
            std::thread            worker([tensor] { tensor->deleter(tensor); });
            py::gil_scoped_release release;
            worker.join();
        }
        return;
    }

    if (PyCapsule_IsValid(capsule.ptr(), "dltensor"))
    {
        auto *tensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
        if (tensor == nullptr)
        {
            throw py::error_already_set();
        }
        if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0)
        {
            throw py::error_already_set();
        }
        if (tensor->deleter != nullptr)
        {
            std::thread            worker([tensor] { tensor->deleter(tensor); });
            py::gil_scoped_release release;
            worker.join();
        }
        return;
    }

    throw std::invalid_argument("Expected an unused DLPack capsule");
}

#ifndef NB_FREE_THREADED
ExternalBuffer::~ExternalBuffer()
{
    if (!m_exportEvent)
        return;
    int previous;
    if (cudaGetDevice(&previous) != cudaSuccess)
        return;
    const int device = m_dlTensor->device.device_id;
    if (previous != device && cudaSetDevice(device) != cudaSuccess)
        return;
    util::CheckLog(cudaEventDestroy(m_exportEvent));
    if (previous != device)
        util::CheckLog(cudaSetDevice(previous));
}

const StreamHandleOwner *ExternalBuffer::exportStreamOwner() const
{
    return m_exportStreamOwner && m_exportStream == m_exportStreamOwner->handle() ? m_exportStreamOwner.get() : nullptr;
}
#endif

void ExternalBuffer::setExportStream(cudaStream_t handle, const std::shared_ptr<const StreamHandleOwner> &owner,
                                     py::handle wrapper)
{
#ifdef NB_FREE_THREADED
    py::ft_lock_guard lock(m_interfaceMutex);
#endif
    if (m_exportStream.has_value() && *m_exportStream == handle && m_exportStreamOwner == owner)
    {
        return;
    }

    m_exportStream      = handle;
    m_exportStreamOwner = owner;
    m_cacheCudaArrayInterface.reset();

    // Exported ExternalBuffer instances may already expose a dynamic CAI
    // attribute. Refresh that snapshot without adding the attribute to imported
    // DLPack-only buffers, preserving their existing protocol surface.
#ifndef NB_FREE_THREADED
    if (wrapper && py::hasattr(wrapper, "__cuda_array_interface__"))
    {
        if (std::optional<py::dict> iface = cudaArrayInterface())
        {
            wrapper.attr("__cuda_array_interface__") = std::move(*iface);
        }
    }
#endif
}

ExternalBuffer::ExternalBuffer(DLPackTensor &&dlTensor)
{
    if (!IsCudaAccessible(dlTensor->device.device_type))
    {
        throw ExternalBufferError("Only CUDA memory buffers can be wrapped");
    }

    if (dlTensor->data != nullptr)
    {
        CheckValidCUDABuffer(static_cast<const std::byte *>(dlTensor->data));
    }

    m_dlTensor = std::move(dlTensor);
#ifdef NB_FREE_THREADED
    m_submissionStates = {AllocationSubmissionState(m_dlTensor->data)};
#endif
}

Shape ExternalBuffer::shape() const
{
    Shape shape(static_cast<size_t>(m_dlTensor->ndim));
    for (size_t i = 0; i < shape.size(); ++i)
    {
        PyTuple_SET_ITEM(shape.ptr(), i, py::cast(m_dlTensor->shape[i]).release().ptr());
    }

    return shape;
}

py::tuple ExternalBuffer::strides() const
{
    py::tuple strides = CheckedTuple(m_dlTensor->ndim);

    for (size_t i = 0; i < strides.size(); ++i)
    {
        PyTuple_SET_ITEM(strides.ptr(), i, py::cast(m_dlTensor->strides[i]).release().ptr());
    }

    return strides;
}

py::object ExternalBuffer::dtype() const
{
    return ToDType(ToNVCVDataType(m_dlTensor->dtype));
}

bool ExternalBuffer::load(PyObject *o)
{
    if (!o)
    {
        return false;
    }

    py::object tmp = py::reinterpret_borrow<py::object>(o);

    // Fetch a CAI property once. A null default distinguishes a missing
    // attribute from a present None, which must be rejected by the CAI loader.
    if (py::object iface = py::getattr(tmp, "__cuda_array_interface__", py::handle()); iface.is_valid())
    {
        return loadCudaArrayInterface(tmp, iface);
    }

    // No equivalent double-fetch below: `__dlpack__` is a method, so looking it
    // up binds it without producing anything. The cost there is in calling it,
    // which happens once.
    if (hasattr(tmp, "__dlpack__"))
    {
        return loadDLPack(tmp);
    }

    return false;
}

bool ExternalBuffer::loadCudaArrayInterface(const py::object &producer, const py::object &interface)
{
    if (!py::isinstance<py::dict>(interface))
    {
        return false;
    }

    py::dict iface = py::cast<py::dict>(interface);

    if (!HasRequiredCAIFields(iface))
    {
        return false;
    }

    if (int version = py::cast<int>(iface["version"]); version < 2)
    {
        return false;
    }

    DLPackTensor dlTensor = CreateCAITensor();
    dlTensor->byte_offset = 0;

    // REVISIT: infer the device type from the memory buffer
    dlTensor->device.device_type = kDLCUDA;
    // REVISIT: infer the device from the memory buffer
    dlTensor->device.device_id = 0;

    py::tuple tdata = py::cast<py::tuple>(iface["data"]);
#ifdef NB_FREE_THREADED
    m_readOnly = py::cast<bool>(tdata[1]);
#endif
    auto *ptr      = AsBytePointer(py::cast<std::uintptr_t>(tdata[0]));
    int   device   = CheckValidCUDABuffer(ptr);
    dlTensor->data = ptr;

    std::optional<CAIType> parsedType = ParseCAITypestr(py::cast<std::string>(iface["typestr"]));
    if (!parsedType)
    {
        return false;
    }
    dlTensor->dtype = parsedType->dtype;

    py::tuple shape = py::cast<py::tuple>(iface["shape"]);
    dlTensor->ndim  = static_cast<int32_t>(shape.size());
    if (dlTensor->ndim < 1)
    {
        return false;
    }

    auto shapeData    = ParseCAIShape(shape);
    auto stridesData  = ParseCAIStrides(iface, dlTensor->ndim, shapeData.get(), parsedType->itemSize);
    dlTensor->shape   = shapeData.release();
    dlTensor->strides = stridesData.release();

    // Parse CAI v3 `stream` to record producer-stream info so the first cvcuda
    // op that reads this buffer can insert the appropriate cross-stream wait.
    ParseCAIStreamField(iface, m_producerStream, m_producerIsSynced);
    m_producerDevice = device;
    if (m_producerDevice >= 0)
    {
        dlTensor->device.device_id = m_producerDevice;
    }

    m_wrappedObj              = producer;
    m_cacheCudaArrayInterface = std::move(iface);
    m_dlTensor                = std::move(dlTensor);
#ifdef NB_FREE_THREADED
    m_submissionStates = {AllocationSubmissionState(m_dlTensor->data, device)};
#endif
    return true;
}

bool ExternalBuffer::loadDLPack(const py::object &object)
{
    if (hasattr(object, "__dlpack_device__"))
    {
        py::tuple dlpackDevice = py::cast<py::tuple>(object.attr("__dlpack_device__")());
        auto      devType      = static_cast<DLDeviceType>(py::cast<int>(dlpackDevice[0]));
        if (!IsCudaAccessible(devType))
        {
            throw ExternalBufferError("Only CUDA-accessible memory buffers can be wrapped");
        }
    }

    py::capsule cap = RequestDLPackCapsule(object);
    loadDLPackCapsule(cap);

    // DLPack does not thread a producer-stream field through the capsule the
    // way CAI does. We asked for stream=1, so seed that as the producer stream
    // for the first cvcuda op that consumes the buffer.
    if (m_dlTensor->data != nullptr)
    {
        m_producerStream   = reinterpret_cast<cudaStream_t>(1);
        m_producerIsSynced = false;
        m_producerDevice   = GetCudaDeviceForPtr(static_cast<const std::byte *>(m_dlTensor->data));
    }
#ifdef NB_FREE_THREADED
    m_submissionStates = {AllocationSubmissionState(m_dlTensor->data, m_producerDevice)};
#endif

    return true;
}

#ifdef NB_FREE_THREADED
void ExternalBuffer::seedSubmissionState() const
{
    if (!m_producerIsSynced && m_producerStream)
        for (const auto &state : submissionStates()) state->seed(m_producerStream, m_producerDevice);
}

SubmissionStates ExternalBuffer::submissionStates() const
{
    py::ft_lock_guard lock(m_interfaceMutex);
    return m_submissionStates;
}

void ExternalBuffer::shareSubmissionStates(const SubmissionStates &states)
{
    py::ft_lock_guard lock(m_interfaceMutex);
    for (const auto &state : states)
    {
        auto position = std::lower_bound(m_submissionStates.begin(), m_submissionStates.end(), state,
                                         [](const auto &a, const auto &b)
                                         { return std::less<SubmissionState *>{}(a.get(), b.get()); });
        if (position == m_submissionStates.end() || *position != state)
            m_submissionStates.insert(position, state);
    }
}
#endif

void ExternalBuffer::loadDLPackCapsule(py::capsule &cap)
{
    if (PyCapsule_IsValid(cap.ptr(), "dltensor_versioned"))
    {
        auto *vt = static_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(cap.ptr(), "dltensor_versioned"));
        if (vt != nullptr)
        {
#ifdef NB_FREE_THREADED
            m_readOnly = (vt->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
#endif
            m_dlTensor = DLPackTensor{vt->dl_tensor};
            m_dlManagedVersioned.reset(vt);
            PyCapsule_SetName(cap.ptr(), "used_dltensor_versioned");
            return;
        }
    }
    else if (PyCapsule_IsValid(cap.ptr(), "dltensor"))
    {
        if (auto *tensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(cap.ptr(), "dltensor")))
        {
            // Mirror the versioned branch: copy the descriptor, but keep the
            // capsule's own DLManagedTensor so its deleter is later called with
            // the pointer the producer handed us. A null deleter means the
            // producer handed over nothing to release and keeps no allocation
            // alive for us, so that case must stay a plain copy.
            m_dlTensor = DLPackTensor{tensor->dl_tensor};
            PyCapsule_SetName(cap.ptr(), "used_dltensor");
            if (tensor->deleter != nullptr)
            {
                m_dlManagedLegacy.reset(tensor);
            }
            return;
        }
    }

    m_dlTensor = {};
}

std::optional<py::dict> ExternalBuffer::cudaArrayInterface() const
{
#ifdef NB_FREE_THREADED
    std::shared_ptr<const StreamHandleOwner> previousOwner;
    auto                                     states = submissionStates();
    SubmissionLocks                          locks;
    locks.add(states, LOCK_MODE_READWRITE);
    locks.lock();
    py::ft_lock_guard interfaceLock(m_interfaceMutex);
#endif
    if (!m_cacheCudaArrayInterface)
    {
        if (!IsCudaAccessible(m_dlTensor->device.device_type))
        {
            return std::nullopt;
        }

        nvcv::DataType dataType = ToNVCVDataType(m_dlTensor->dtype);

        NVCV_ASSERT(m_dlTensor->dtype.bits % 8 == 0);
        NVCV_ASSERT(dataType.strideBytes() * 8 == m_dlTensor->dtype.bits * m_dlTensor->dtype.lanes);
        int elemStrideBytes = dataType.strideBytes();

        py::object strides;

        if (m_dlTensor->strides == nullptr)
        {
            strides = py::none();
        }
        else
        {
            py::tuple vStrides = CheckedTuple(m_dlTensor->ndim);
            for (size_t i = 0; i < vStrides.size(); ++i)
            {
                PyTuple_SET_ITEM(vStrides.ptr(), i, py::cast(m_dlTensor->strides[i] * elemStrideBytes).release().ptr());
            }
            strides = vStrides;
        }

        std::string format = ToFormatString(m_dlTensor->dtype);

        // CAI v3 `stream` field — advertise which stream the consumer must
        // synchronize with before reading.  If an export stream was set
        // (typically by Tensor::cuda() from the owning tensor's last-writer
        // stream), emit its integer handle.  Otherwise, default to `1` (legacy
        // default stream), which is the safe conservative value per the CAI
        // spec and matches behavior for locally-constructed buffers whose
        // writer stream is unknown.
        long streamValue = 1; // cudaStreamLegacy
        if (m_exportStream.has_value())
        {
            // A null handle is CUDA's legacy default stream. Only the -1
            // sentinel means the producer has already synchronized.
            streamValue = *m_exportStream == nullptr ? 1L : reinterpret_cast<intptr_t>(*m_exportStream);
        }

        py::dict iface;
        iface["shape"]            = this->shape();
        iface["strides"]          = strides;
        iface["typestr"]          = format;
        iface["data"]             = py::make_tuple(reinterpret_cast<long>(m_dlTensor->data), false /* read/write */);
        iface["stream"]           = streamValue;
        iface["version"]          = 3;
        m_cacheCudaArrayInterface = std::move(iface);
    }

#ifdef NB_FREE_THREADED
    py::dict result = py::steal<py::dict>(PyDict_Copy(m_cacheCudaArrayInterface->ptr()));
    if (!result.is_valid())
        py::raise_python_error();
    result["data"] = py::make_tuple(reinterpret_cast<uintptr_t>(m_dlTensor->data), m_readOnly);
    if (!states.empty())
    {
        auto access         = ExportSubmissionAccess(states);
        result["stream"]    = access.stream == nullptr ? 1L : reinterpret_cast<intptr_t>(access.stream);
        previousOwner       = std::move(m_exportStreamOwner);
        m_exportStreamOwner = std::move(access.owner);
        m_exportStream      = access.stream;
    }
    return result;
#else
    return *m_cacheCudaArrayInterface;
#endif
}

py::capsule ExternalBuffer::dlpack(py::handle stream, py::handle maxVersion) const
{
#ifdef NB_FREE_THREADED
    // Integer conversions can call Python and re-enter this buffer's export.
    cudaStream_t consumerStream = nullptr;
    bool         needSync       = false;
    ParseDLPackStreamArg(stream, consumerStream, needSync);
    const bool versioned = UseVersionedDLPack(maxVersion);

    SubmissionLocks locks;
    {
        py::ft_lock_guard lock(m_interfaceMutex);
        // Published buffers only add states. This call keeps the buffer,
        // and hence every captured allocation state, alive until unlock.
        locks.add(m_submissionStates, LOCK_MODE_READWRITE);
    }
    locks.lock();
    if (needSync)
    {
        locks.exportTo({consumerStream, m_dlTensor->device.device_id, {}});
    }
#else
    // Honor the DLPack v1 `stream` contract: when the consumer passes a
    // CUDA stream, synchronize the writer stream into it before returning
    // the capsule so any work the consumer queues on that stream waits for
    // our pending writes.  `m_exportStream` is set by Tensor::cuda() /
    // Image::cuda() / Array::cuda() when a cvcuda op last wrote the buffer;
    // if it's unset the buffer wasn't produced by cvcuda (pass-through
    // import), and we have no writer stream to sync.
    if (m_exportStream.has_value())
    {
        cudaStream_t consumerStream = nullptr;
        bool         needSync       = false;
        ParseDLPackStreamArg(stream, consumerStream, needSync);
        if (needSync)
        {
            cudaStream_t writerStream = *m_exportStream;
            if (writerStream != reinterpret_cast<cudaStream_t>(-1))
            {
                InsertDLPackStreamSync(writerStream, consumerStream, m_dlTensor->device.device_id, m_exportEvent);
            }
        }
    }
    const bool versioned = UseVersionedDLPack(maxVersion);
#endif
    return versioned ? CreateVersionedDLPackCapsule(*m_dlTensor, this->shared_from_this())
                     : CreateLegacyDLPackCapsule(*m_dlTensor, this->shared_from_this());
}

py::tuple ExternalBuffer::dlpackDevice() const
{
    return py::make_tuple(py::int_(static_cast<int>(m_dlTensor->device.device_type)),
                          py::int_(m_dlTensor->device.device_id));
}

const DLTensor &ExternalBuffer::dlTensor() const
{
    return *m_dlTensor;
}

void ExternalBuffer::Export(py::module_ &m)
{
    py::class_<ExternalBuffer>(m, "ExternalBuffer", py::dynamic_attr(), py::is_weak_referenceable())
#ifdef NB_FREE_THREADED
        .def_prop_ro("__cuda_array_interface__", &ExternalBuffer::cudaArrayInterface)
#endif
        .def_prop_ro("shape", &ExternalBuffer::shape, "Get the shape of the buffer as an array")
        .def_prop_ro("strides", &ExternalBuffer::strides, "Get the strides of the buffer")
        .def_prop_ro("dtype", &ExternalBuffer::dtype, "Get the data type of the buffer")
        .def("__dlpack__", &ExternalBuffer::dlpack,
             "stream"_a
#ifdef NB_FREE_THREADED
                 .none()
#endif
             = 1,
             "max_version"_a = py::none(), "Export the buffer as a DLPack tensor")
        .def("__dlpack_device__", &ExternalBuffer::dlpackDevice, "Get the device associated with the buffer");
}

} // namespace nvcvpy::priv

namespace nanobind::detail {

namespace priv = nvcvpy::priv;

bool type_caster<priv::ExternalBuffer>::from_python(handle src, uint32_t flags, cleanup_list *cleanup)
{
    m_value = nullptr;
    m_owner.reset();
    if (type_caster_base<priv::ExternalBuffer> boundCaster; boundCaster.from_python(src, flags, cleanup))
    {
        m_value = boundCaster.operator priv::ExternalBuffer *();
        return true;
    }

    m_owner = std::make_shared<priv::ExternalBuffer>();
    if (m_owner->load(src.ptr()))
    {
        m_value = m_owner.get();
        return true;
    }

    m_owner.reset();
    return false;
}

} // namespace nanobind::detail
