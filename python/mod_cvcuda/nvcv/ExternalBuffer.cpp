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

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <functional> // for std::multiplies
#include <memory>
#include <stdexcept>

namespace nvcvpy::priv {

using namespace py::literals;

namespace {

class ExternalBufferError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace

static std::byte *AsBytePointer(std::uintptr_t address) noexcept
{
    return reinterpret_cast<std::byte *>(address);
}

static void CheckValidCUDABuffer(const std::byte *ptr)
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
        value = sf.cast<long>();
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
static void ParseDLPackStreamArg(const py::object &streamArg, cudaStream_t &outConsumerStream, bool &outNeedSync)
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
        value = streamArg.cast<long>();
    }
    catch (py::cast_error &)
    {
        // Unparseable stream arg; conservative: no sync, consumer is on
        // their own.  This also covers edge cases like callers passing a
        // cupy.cuda.Stream object directly (which pybind11 may not cast).
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

// RAII guard that destroys a `cudaEvent_t` on scope exit.  Used below so
// an exception thrown from `cudaEventRecord` or `cudaStreamWaitEvent`
// cannot leak the event that was just created.  Declared local to this
// translation unit since no other file here needs it.
namespace {
struct ScopedCudaEvent
{
    cudaEvent_t handle = nullptr;

    ScopedCudaEvent()                                   = default;
    ScopedCudaEvent(const ScopedCudaEvent &)            = delete;
    ScopedCudaEvent &operator=(const ScopedCudaEvent &) = delete;
    ScopedCudaEvent(ScopedCudaEvent &&)                 = delete;
    ScopedCudaEvent &operator=(ScopedCudaEvent &&)      = delete;

    ~ScopedCudaEvent()
    {
        if (handle != nullptr)
        {
            util::CheckLog(cudaEventDestroy(handle));
        }
    }
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

void DeleteVersionedManager(DLManagedTensorVersioned *tensor)
{
    std::unique_ptr<VersionedManagerCtx> ctx(static_cast<VersionedManagerCtx *>(tensor->manager_ctx));
    (void)ctx;
}

void DeleteManager(DLManagedTensor *tensor)
{
    std::unique_ptr<ManagerCtx> ctx(static_cast<ManagerCtx *>(tensor->manager_ctx));
    (void)ctx;
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

bool UseVersionedDLPack(py::object maxVersion)
{
    if (maxVersion.is_none())
    {
        return false;
    }

    py::tuple ver = maxVersion.cast<py::tuple>();
    return ver.size() >= 1 && ver[0].cast<int>() >= 1;
}

py::capsule CreateVersionedDLPackCapsule(const DLTensor &dlTensor, std::shared_ptr<const ExternalBuffer> extBuffer)
{
    auto ctx = std::make_unique<VersionedManagerCtx>();

    ctx->tensor.version     = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    ctx->tensor.flags       = 0;
    ctx->tensor.manager_ctx = ctx.get();
    ctx->tensor.deleter     = DeleteVersionedManager;
    ctx->tensor.dl_tensor   = dlTensor;
    ctx->extBuffer          = std::move(extBuffer);

    py::capsule cap(&ctx->tensor, "dltensor_versioned", ReleaseVersionedCapsule);
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

    py::capsule cap(&ctx->tensor, "dltensor", ReleaseLegacyCapsule);
    ctx.release();
    return cap;
}

// Insert a cross-stream synchronization into the consumer's stream before
// returning a DLPack capsule that exposes writer-stream-pending data.
//
// Records a fresh event on the producer's writer stream and queues a wait
// on the consumer's stream.  Per CUDA docs the event may be destroyed
// immediately after `cudaStreamWaitEvent` returns (the wait captures a
// snapshot of the recording), so we destroy it on scope exit via
// `ScopedCudaEvent`.
//
// Semantics:
//   * Writer stream unknown (buffer wasn't produced by cvcuda, i.e.
//     m_hasExportStream is false or handle is null) — nothing to sync,
//     return.
//   * Writer == consumer — sync is trivial (stream is sequential),
//     return.
//   * Writer == opted-out sentinel (nullptr) from `setExportStream(
//     (cudaStream_t)-1)` — producer has asserted no sync is needed,
//     return.
//   * Otherwise — insert `cudaEventRecord`+`cudaStreamWaitEvent`.
void InsertDLPackStreamSync(cudaStream_t writerStream, cudaStream_t consumerStream)
{
    if (writerStream == nullptr || writerStream == consumerStream)
    {
        return;
    }

    ScopedCudaEvent evt;
    util::CheckThrow(cudaEventCreateWithFlags(&evt.handle, cudaEventDisableTiming));
    util::CheckThrow(cudaEventRecord(evt.handle, writerStream));
    util::CheckThrow(cudaStreamWaitEvent(consumerStream, evt.handle, 0));
    // evt destroyed by ScopedCudaEvent destructor on return / on any
    // exception propagating out of the CheckThrow calls above.
}

std::string ToFormatString(const DLDataType &dtype)
{
    py::dtype dt = ToDType(ToNVCVDataType(dtype));
    return dt.attr("str").cast<std::string>();
}

std::unique_ptr<int64_t[]> ParseCAIShape(const py::tuple &shape)
{
    auto shapeData = std::make_unique<int64_t[]>(shape.size());

    for (size_t i = 0; i < shape.size(); ++i)
    {
        shapeData[i] = shape[i].cast<long>();
    }

    return shapeData;
}

std::unique_ptr<int64_t[]> ParseCAIStrides(const py::dict &iface, int ndim, const int64_t *shape, int64_t itemSize)
{
    auto stridesData = std::make_unique<int64_t[]>(ndim);

    if (iface.contains("strides") && !iface["strides"].is_none())
    {
        py::tuple strides = iface["strides"].cast<py::tuple>();
        for (int i = 0; i < ndim; ++i)
        {
            int64_t strideBytes = strides[i].cast<long>();
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

py::capsule RequestDLPackCapsule(const py::object &object)
{
    try
    {
        return object.attr("__dlpack__")("stream"_a = 1, "max_version"_a = py::make_tuple(1, 0)).cast<py::capsule>();
    }
    catch (py::error_already_set &)
    {
        // Producer doesn't accept max_version (v0 producer) -- retry without it.
        PyErr_Clear();
        return object.attr("__dlpack__")("stream"_a = 1).cast<py::capsule>();
    }
}

} // namespace

py::object ExternalBuffer::Create(DLPackTensor &&dlPackTensor, py::object wrappedObj, cudaStream_t exportStream,
                                  bool setExportStream)
{
    auto buf = std::make_shared<ExternalBuffer>(std::move(dlPackTensor));

    if (setExportStream)
    {
        buf->setExportStream(exportStream);
    }

    // We must make the returned object keep wrappedObj alive.
    // Using py::return_value_policy::reference_internal in py::cast doesn't work
    // because buf is a shared_ptr.
    py::object o = py::cast(buf);
    py::detail::keep_alive_impl(o, wrappedObj);

    // If we expose cuda array interface,
    if (auto iface = buf->cudaArrayInterface())
    {
        o.attr("__cuda_array_interface__") = *iface;
    }

    return o;
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
}

Shape ExternalBuffer::shape() const
{
    Shape shape(m_dlTensor->ndim);
    for (size_t i = 0; i < shape.size(); ++i)
    {
        shape[i] = m_dlTensor->shape[i];
    }

    return shape;
}

py::tuple ExternalBuffer::strides() const
{
    py::tuple strides(m_dlTensor->ndim);

    for (size_t i = 0; i < strides.size(); ++i)
    {
        strides[i] = m_dlTensor->strides[i];
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
    if (hasattr(tmp, "__cuda_array_interface__"))
    {
        return loadCudaArrayInterface(tmp);
    }

    if (hasattr(tmp, "__dlpack__"))
    {
        return loadDLPack(tmp);
    }

    return false;
}

bool ExternalBuffer::loadCudaArrayInterface(const py::object &object)
{
    py::dict iface = object.attr("__cuda_array_interface__").cast<py::dict>();

    if (!HasRequiredCAIFields(iface))
    {
        return false;
    }

    if (int version = iface["version"].cast<int>(); version < 2)
    {
        return false;
    }

    DLPackTensor dlTensor = CreateCAITensor();
    dlTensor->byte_offset = 0;

    // REVISIT: infer the device type from the memory buffer
    dlTensor->device.device_type = kDLCUDA;
    // REVISIT: infer the device from the memory buffer
    dlTensor->device.device_id = 0;

    py::tuple tdata = iface["data"].cast<py::tuple>();
    auto     *ptr   = AsBytePointer(tdata[0].cast<std::uintptr_t>());
    CheckValidCUDABuffer(ptr);
    dlTensor->data = ptr;

    py::dtype dt = util::ToDType(iface["typestr"].cast<std::string>());
    if (std::optional<nvcv::DataType> dtype = ToNVCVDataType(dt))
    {
        dlTensor->dtype = ToDLDataType(*dtype);
    }

    py::tuple shape = iface["shape"].cast<py::tuple>();
    dlTensor->ndim  = static_cast<int32_t>(shape.size());
    if (dlTensor->ndim < 1)
    {
        return false;
    }

    auto shapeData    = ParseCAIShape(shape);
    auto stridesData  = ParseCAIStrides(iface, dlTensor->ndim, shapeData.get(), dt.itemsize());
    dlTensor->shape   = shapeData.release();
    dlTensor->strides = stridesData.release();

    // Parse CAI v3 `stream` to record producer-stream info so the first cvcuda
    // op that reads this buffer can insert the appropriate cross-stream wait.
    ParseCAIStreamField(iface, m_producerStream, m_producerIsSynced);
    m_producerDevice = GetCudaDeviceForPtr(static_cast<const std::byte *>(dlTensor->data));
    if (m_producerDevice >= 0)
    {
        dlTensor->device.device_id = m_producerDevice;
    }

    m_wrappedObj              = object;
    m_cacheCudaArrayInterface = std::move(iface);
    m_dlTensor                = std::move(dlTensor);
    return true;
}

bool ExternalBuffer::loadDLPack(const py::object &object)
{
    if (hasattr(object, "__dlpack_device__"))
    {
        py::tuple dlpackDevice = object.attr("__dlpack_device__")().cast<py::tuple>();
        auto      devType      = static_cast<DLDeviceType>(dlpackDevice[0].cast<int>());
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

    return true;
}

void ExternalBuffer::loadDLPackCapsule(py::capsule &cap)
{
    if (PyCapsule_IsValid(cap.ptr(), "dltensor_versioned"))
    {
        auto *vt = static_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(cap.ptr(), "dltensor_versioned"));
        if (vt != nullptr)
        {
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
            m_dlTensor = DLPackTensor{std::move(*tensor)};
            PyCapsule_SetName(cap.ptr(), "used_dltensor");
            return;
        }
    }

    m_dlTensor = {};
}

std::optional<py::dict> ExternalBuffer::cudaArrayInterface() const
{
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
            py::tuple vStrides(m_dlTensor->ndim);
            for (size_t i = 0; i < vStrides.size(); ++i)
            {
                vStrides[i] = m_dlTensor->strides[i] * elemStrideBytes;
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
        if (m_hasExportStream)
        {
            // m_exportStream == nullptr means "no sync needed" (-1 per spec),
            // typically used when the producer has already synchronized.
            streamValue = m_exportStream == nullptr ? -1L : reinterpret_cast<intptr_t>(m_exportStream);
        }

        // clang-format off
        m_cacheCudaArrayInterface = py::dict
        {
            "shape"_a = this->shape(),
            "strides"_a = strides,
            "typestr"_a = format,
            "data"_a = py::make_tuple(reinterpret_cast<long>(m_dlTensor->data), false /* read/write */),
            "stream"_a = streamValue,
            "version"_a = 3
        };
    }

    return *m_cacheCudaArrayInterface;
}

py::capsule ExternalBuffer::dlpack(py::object stream, py::object maxVersion) const
{
    // Honor the DLPack v1 `stream` contract: when the consumer passes a
    // CUDA stream, synchronize the writer stream into it before returning
    // the capsule so any work the consumer queues on that stream waits for
    // our pending writes.  `m_hasExportStream` is set by Tensor::cuda() /
    // Image::cuda() / Array::cuda() when a cvcuda op last wrote the buffer;
    // if it's unset the buffer wasn't produced by cvcuda (pass-through
    // import), and we have no writer stream to sync.
    if (m_hasExportStream)
    {
        cudaStream_t consumerStream = nullptr;
        bool         needSync       = false;
        ParseDLPackStreamArg(stream, consumerStream, needSync);
        if (needSync)
        {
            // m_exportStream == -1 sentinel means the producer asserted
            // the data is already synchronized (CAI v3 `stream: -1`
            // semantics); honor that by skipping the wait here too.
            cudaStream_t writerStream = m_exportStream;
            if (writerStream == reinterpret_cast<cudaStream_t>(-1))
            {
                writerStream = nullptr;
            }
            InsertDLPackStreamSync(writerStream, consumerStream);
        }
    }

    if (UseVersionedDLPack(maxVersion))
    {
        return CreateVersionedDLPackCapsule(*m_dlTensor, this->shared_from_this());
    }

    return CreateLegacyDLPackCapsule(*m_dlTensor, this->shared_from_this());
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

void ExternalBuffer::Export(py::module &m)
{
    py::class_<ExternalBuffer, std::shared_ptr<ExternalBuffer>>(m, "ExternalBuffer", py::dynamic_attr())
        .def_property_readonly("shape", &ExternalBuffer::shape, "Get the shape of the buffer as an array")
        .def_property_readonly("strides", &ExternalBuffer::strides, "Get the strides of the buffer")
        .def_property_readonly("dtype", &ExternalBuffer::dtype, "Get the data type of the buffer")
        .def("__dlpack__", &ExternalBuffer::dlpack, "stream"_a=1, "max_version"_a=py::none(),
             "Export the buffer as a DLPack tensor")
        .def("__dlpack_device__", &ExternalBuffer::dlpackDevice, "Get the device associated with the buffer");
}

} // namespace nvcvpy::priv

namespace pybind11::detail {

namespace priv = nvcvpy::priv;

// Python -> C++
bool type_caster<priv::ExternalBuffer>::load(handle src, bool)
{
    const PyTypeObject *srctype = Py_TYPE(src.ptr());
    const type_info *cuda_buffer_type = get_type_info(typeid(priv::ExternalBuffer));

    // src's type is ExternalBuffer?
    if(srctype == cuda_buffer_type->type)
    {
        // We know it's managed by a shared pointer (holder), let's use it
        value_and_holder vh = reinterpret_cast<instance *>(src.ptr())->get_value_and_holder();
        value = vh.template holder<std::shared_ptr<priv::ExternalBuffer>>();
        NVCV_ASSERT(value != nullptr);
        return true;
    }
    // If not, it could be an object that implements that __cuda_array_interface, let's try to
    // create a ExternalBuffer out of it.
    else
    {
        value = std::make_shared<priv::ExternalBuffer>();
        return value->load(src.ptr());
    }
}

} // namespace pybind11::detail
