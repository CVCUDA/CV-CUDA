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

#include "Stream.hpp"

#include "../NvtxRange.hpp"
#include "Cache.hpp"
#include "Definitions.hpp"
#include "StreamStack.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <pybind11/operators.h>

#include <new>
#include <stdexcept>
#include <string>

namespace nvcvpy::priv {

namespace {

class StreamError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace

// Static members initialization
std::unordered_map<int, cudaStream_t> Stream::m_auxStreams;
std::atomic<int>                      Stream::m_instanceCount = 0;
std::shared_mutex                     Stream::m_auxStreamMutex;
std::mutex                            Stream::m_gcMutex;

// Here we define the representation of external cuda streams.
// It defines pybind11's type casters from the python object
// to the corresponding ExternalStream<E>.

// Defines each external stream represetation we support.
enum ExternalStreamType
{
    VOIDP,
    INT,
    TORCH,
    CUPY,
};

template<ExternalStreamType E>
class ExternalStream : public IExternalStream
{
public:
    ExternalStream() = default;

    explicit ExternalStream(cudaStream_t cudaStream)
        : m_cudaStream(cudaStream)
    {
    }

    void setCudaStream(cudaStream_t cudaStream, py::object obj)
    {
        m_cudaStream = cudaStream;
        m_wrappedObj = std::move(obj);
    }

    cudaStream_t handle() const override
    {
        return m_cudaStream;
    }

    py::object wrappedObject() const override
    {
        return m_wrappedObj;
    }

private:
    cudaStream_t m_cudaStream;
    py::object   m_wrappedObj;
};

} // namespace nvcvpy::priv

namespace PYBIND11_NAMESPACE { namespace detail {

using namespace std::literals;
namespace util = nvcvpy::util;
namespace priv = nvcvpy::priv;

template<>
struct type_caster<priv::ExternalStream<priv::VOIDP>>
{
    PYBIND11_TYPE_CASTER(priv::ExternalStream<priv::VOIDP>, const_name("ctypes.c_void_p"));

    bool load(handle src, bool)
    {
        if (std::string strType = util::GetFullyQualifiedName(src); strType != "ctypes.c_void_p")
        {
            return false;
        }

        buffer_info info = ::pybind11::cast<buffer>(src).request();

        NVCV_ASSERT(info.itemsize == sizeof(void *));

        void *data = *reinterpret_cast<void **>(info.ptr);

        value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(src));
        return true;
    }
};

template<>
struct type_caster<priv::ExternalStream<priv::INT>>
{
    PYBIND11_TYPE_CASTER(priv::ExternalStream<priv::INT>, const_name("int"));

    bool load(handle src, bool)
    {
        try
        {
            // REVISIT: don't know how to test if a python object
            // is convertible to a type without exceptions.
            intptr_t data = src.cast<intptr_t>();
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(src));
            return true;
        }
        catch (...) // NOSONAR: pybind type casters probe conversions by catching failures.
        {
            return false;
        }
    }
};

template<>
struct type_caster<priv::ExternalStream<priv::TORCH>>
{
    PYBIND11_TYPE_CASTER(priv::ExternalStream<priv::TORCH>, const_name("torch.cuda.Stream"));

    bool load(handle src, bool)
    {
        if (std::string strType = util::GetFullyQualifiedName(src);
            strType != "torch.cuda.streams.Stream" && strType != "torch.cuda.streams.ExternalStream")
        {
            return false;
        }

        try
        {
            // REVISIT: don't know how to test if a python object
            // is convertible to a type without exceptions.
            intptr_t data = src.attr("cuda_stream").cast<intptr_t>();
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(src));
            return true;
        }
        catch (...) // NOSONAR: pybind type casters probe conversions by catching failures.
        {
            return false;
        }
    }
};

template<>
struct type_caster<priv::ExternalStream<priv::CUPY>>
{
    PYBIND11_TYPE_CASTER(priv::ExternalStream<priv::CUPY>, const_name("cupy.cuda.Stream"));

    bool load(handle src, bool)
    {
        if (std::string strType = util::GetFullyQualifiedName(src);
            strType != "cupy.cuda.stream.Stream" && strType != "cupy.cuda.stream.ExternalStream")
        {
            return false;
        }

        try
        {
            intptr_t data = src.attr("ptr").cast<intptr_t>();
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(src));
            return true;
        }
        catch (...) // NOSONAR: pybind type casters probe conversions by catching failures.
        {
            return false;
        }
    }
};

}} // namespace PYBIND11_NAMESPACE::detail

namespace nvcvpy::priv {

// In terms of caching, all streams are the same.
// Any stream in the cache can be fetched and used.
size_t Stream::Key::doGetHash() const
{
    return 0;
}

bool Stream::Key::doIsCompatible(const IKey &that) const
{
    return true;
}

std::shared_ptr<Stream> Stream::Create()
{
    ::cvcudapy::NvtxRange                   nvtxRange("cvcuda.Stream.create");
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Stream::Key{});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Stream> stream(new Stream()); // NOSONAR: constructor is private.
        Cache::Instance().add(*stream);
        return stream;
    }
    else
    {
        // Get the first one
        return std::static_pointer_cast<Stream>(vcont[0]);
    }
}

Stream::Stream()
    : m_owns(true)
    , m_size_inbytes(doComputeSizeInBytes())
{
    try
    {
        util::CheckThrow(cudaStreamCreateWithFlags(&m_handle, cudaStreamNonBlocking));
        incrementInstanceCount();
        GetAuxStream();
    }
    catch (...) // NOSONAR: constructor cleanup must run for any exception.
    {
        destroy();
        throw;
    }
}

Stream::Stream(IExternalStream &extStream)
    : m_handle(extStream.handle())
    , m_wrappedObj(extStream.wrappedObject())
{
    if (unsigned int flags; cudaStreamGetFlags(m_handle, &flags) != cudaSuccess)
    {
        throw StreamError("Invalid cuda stream");
    }

    try
    {
        incrementInstanceCount();
        GetAuxStream();
    }
    catch (...) // NOSONAR: constructor cleanup must run for any exception.
    {
        destroy();
        throw;
    }
}

void Stream::incrementInstanceCount()
{
    m_instanceCount.fetch_add(1, std::memory_order_relaxed);
}

int Stream::decrementInstanceCount()
{
    return m_instanceCount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

// Returns by value intentionally: returning a reference into m_auxStreams
// would be unsafe because a later insertion can trigger a rehash, invalidating
// the reference after the mutex is released.
cudaStream_t Stream::GetAuxStream()
{
    // Tolerate hosts with no CUDA device. The wrap-stream-0 ctor calls this
    // at module init for cache warm-up; any real consumer of the returned
    // handle (cudaStreamWaitEvent / cudaStreamAddCallback) requires a device
    // and would have failed regardless. Returning nullptr here lets
    // `import cvcuda` succeed on CPU-only build/CI hosts.
    int         dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err == cudaErrorNoDevice || err == cudaErrorStubLibrary)
    {
        (void)cudaGetLastError(); // clear sticky error
        return nullptr;
    }
    util::CheckThrow(err);

    // Shared lock: concurrent readers when the entry already exists.
    {
        std::shared_lock lock(m_auxStreamMutex);
        auto             it = m_auxStreams.find(dev);
        if (it != m_auxStreams.end())
            return it->second;
    }

    // Exclusive lock: serializes the one-time insertion of a new entry.
    std::unique_lock lock(m_auxStreamMutex);
    auto [it, inserted] = m_auxStreams.try_emplace(dev, nullptr);
    if (inserted)
    {
        cudaStream_t s = nullptr;
        util::CheckThrow(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        it->second = s;
    }
    return it->second;
}

void Stream::SyncAuxStream()
{
    ::cvcudapy::NvtxRange nvtxRange("cvcuda.Stream.syncAuxStream");
    // Sync aux streams on all devices.
    std::shared_lock      lock(m_auxStreamMutex);
    int                   savedDev = 0;
    util::CheckThrow(cudaGetDevice(&savedDev));

    struct DeviceGuard
    {
        explicit DeviceGuard(int dev_)
            : dev(dev_)
        {
        }

        DeviceGuard(const DeviceGuard &)            = delete;
        DeviceGuard(DeviceGuard &&)                 = delete;
        DeviceGuard &operator=(const DeviceGuard &) = delete;
        DeviceGuard &operator=(DeviceGuard &&)      = delete;

        ~DeviceGuard()
        {
            util::CheckLog(cudaSetDevice(dev));
        }

        int dev;
    };

    DeviceGuard guard{savedDev};

    for (const auto &[dev, auxStream] : m_auxStreams)
    {
        util::CheckThrow(cudaSetDevice(dev));
        util::CheckThrow(cudaStreamSynchronize(auxStream));
    }
}

Stream::~Stream()
{
    destroy();
}

void Stream::destroy()
{
    if (m_owns && m_handle)
    {
        int savedDev = 0;
        util::CheckLog(cudaGetDevice(&savedDev));
        util::CheckLog(cudaSetDevice(m_key.deviceId()));
        util::CheckLog(cudaStreamSynchronize(m_handle));
        util::CheckLog(cudaStreamDestroy(m_handle));
        util::CheckLog(cudaSetDevice(savedDev));
        m_handle = nullptr;
    }
    {
        std::unique_lock lock(m_auxStreamMutex);
        if (decrementInstanceCount() == 0)
        {
            int savedDev = 0;
            util::CheckLog(cudaGetDevice(&savedDev));
            for (const auto &[dev, auxStream] : m_auxStreams)
            {
                util::CheckLog(cudaSetDevice(dev));
                util::CheckLog(cudaStreamSynchronize(auxStream));
                util::CheckLog(cudaStreamDestroy(auxStream));
            }
            m_auxStreams.clear();
            util::CheckLog(cudaSetDevice(savedDev));
        }
    }
    {
        std::lock_guard lock(m_eventMutex);
        int             savedDev = 0;
        util::CheckLog(cudaGetDevice(&savedDev));
        for (const auto &[dev, evt] : m_events)
        {
            util::CheckLog(cudaSetDevice(dev));
            util::CheckLog(cudaEventDestroy(evt));
        }
        m_events.clear();
        util::CheckLog(cudaSetDevice(savedDev));
    }
}

int64_t Stream::doComputeSizeInBytes() const
{
    // We only cache the stream's handles, which are 8 byte on CPU memory, hence 0 bytes gpu memory.
    return 0;
}

int64_t Stream::GetSizeInBytes() const
{
    // m_size_inbytes == -1 indicates failure case and value has not been computed yet
    NVCV_ASSERT(m_size_inbytes != -1
                && "Stream has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
    return m_size_inbytes;
}

std::shared_ptr<Stream> Stream::sharedStream()
{
    return std::dynamic_pointer_cast<Stream>(Object::shared_from_this());
}

std::shared_ptr<const Stream> Stream::sharedStream() const
{
    return std::dynamic_pointer_cast<const Stream>(Object::shared_from_this());
}

cudaStream_t Stream::handle() const
{
    return m_handle;
}

int Stream::deviceId() const
{
    return m_key.deviceId();
}

intptr_t Stream::pyhandle() const
{
    return reinterpret_cast<intptr_t>(m_handle);
}

void Stream::sync()
{
    ::cvcudapy::NvtxRange  nvtxRange("cvcuda.Stream.sync");
    py::gil_scoped_release release;
    util::CheckThrow(cudaStreamSynchronize(m_handle));
}

void Stream::wait_stream(std::shared_ptr<Stream> other)
{
    ::cvcudapy::NvtxRange nvtxRange("cvcuda.Stream.wait_stream");
    if (!other)
        throw std::invalid_argument("other is null");

    if (other->handle() == m_handle)
        return;

    py::gil_scoped_release release;
    cudaEvent_t            evt = other->getEvent();
    util::CheckThrow(cudaEventRecord(evt, other->handle()));
    util::CheckThrow(cudaStreamWaitEvent(m_handle, evt, 0));
}

Stream &Stream::Current()
{
    auto defStream = StreamStack::Instance().top();
    if (!defStream)
    {
        // Empty stream stack means Stream::Export skipped wrapping the legacy
        // default stream — only happens on no-device hosts. Throw a Python-
        // visible error rather than NVCV_ASSERT (which would abort()), so
        // that introspection tools (pybind11_stubgen, sphinx) can read the
        // class without crashing.
        throw StreamError("No default cvcuda.Stream available (no CUDA device visible).");
    }
    return *defStream;
}

void Stream::activate()
{
    ::cvcudapy::NvtxRange nvtxRange("cvcuda.Stream.__enter__");
    if (m_owns)
    {
        int curDev = 0;
        util::CheckThrow(cudaGetDevice(&curDev));
        if (m_key.deviceId() != curDev)
        {
            throw StreamError("Cannot activate cvcuda.Stream (created on device " + std::to_string(m_key.deviceId())
                              + ") while current CUDA device is " + std::to_string(curDev));
        }
    }
    StreamStack::Instance().push(*this);
}

void Stream::deactivate(py::object, py::object, py::object) const
{
    ::cvcudapy::NvtxRange nvtxRange("cvcuda.Stream.__exit__");
    StreamStack::Instance().pop();
}

// Stores the data held by a cuda host callback function in a cuda stream.
// It's used for:
// - Extend the lifetime of the objects it contains until they aren't needed
//   by any future cuda kernels in the stream.
struct Stream::HostFunctionClosure
{
    // Also hold the stream reference so that it isn't destroyed before the processing is done.
    std::shared_ptr<const Stream> stream;
    LockResources                 resources;
};

cudaEvent_t Stream::getEvent()
{
    int dev;
    util::CheckThrow(cudaGetDevice(&dev));

    {
        std::shared_lock lock(m_eventMutex);
        auto             it = m_events.find(dev);
        if (it != m_events.end())
            return it->second;
    }

    std::unique_lock lock(m_eventMutex);
    auto [it, inserted] = m_events.try_emplace(dev, nullptr);
    if (inserted)
    {
        cudaEvent_t evt = nullptr;
        util::CheckThrow(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
        it->second = evt;
    }
    return it->second;
}

void Stream::holdResources(LockResources usedResources)
{
    if (!usedResources.empty())
    {
        // Looks like a good place to clear the gc bag, as every time we create
        // a new closure that eventually gets added to the bag, we empty it.
        // The bag shouldn't grow unlimited.
        // Calling it before allocating a new closure just avoid having two
        // closures not inside a cuda stream that are simultaneously alive, but
        // in practice it doesn't seem to matter much.
        ClearGCBag();

        auto closure = std::make_unique<HostFunctionClosure>();

        closure->stream    = this->sharedStream();
        closure->resources = std::move(usedResources);

        auto fn = [](cudaStream_t, cudaError_t, auto *userData)
        {
            std::unique_ptr<HostFunctionClosure> pclosure(static_cast<HostFunctionClosure *>(userData));
            NVCV_ASSERT(pclosure != nullptr);
            AddToGCBag(std::move(pclosure));
        };

        // If we naively execute the callback in the main stream (m_handle), the GPU will wait until the callback
        // is executed (on host). For correctness, GPU doesn't need to wait - it's the CPU that needs
        // to wait for the work already scheduled to complete.
        //
        // Naive timeline:
        //
        // stream        GPU_kernel1 | Callback | GPU_kernel2
        // GPU activity  xxxxxxxxxxx              xxxxxxxxxxx
        // CPU activity                xxxxxxxx
        //
        // Optimized timeline
        //
        //
        //                event -----v
        // stream        GPU_kernel1 | GPU_kernel2
        // aux_stream     waitEvent >| Callback
        //
        // GPU activity  xxxxxxxxxxx   xxxxxxxxxxx
        // CPU activity                xxxxxxxx

        cudaEvent_t evt = getEvent();
        util::CheckThrow(cudaEventRecord(evt, m_handle));           // add async record the event in the main stream
        util::CheckThrow(cudaStreamWaitEvent(GetAuxStream(), evt)); // add async wait for the event in the aux stream

        // cudaStreamAddCallback pushes a task to the given stream, which at some point (asynchonously) calls
        // the given callback (fn), passing to it the closure we created, among other stream states.
        // When fn is executed, the refcnt of all objects that the closure holds will eventually be decremented, which
        // will trigger their deletion if refcnt==0. This effectively extends the objects' lifetime until
        // all tasks that refer to them are finished.

        // The callback will be executed in the singleton aux stream there may be contention with other callbacks and waitEvents from
        // other streams. However the callback is used to release resources from the cache and should not be a performance bottleneck.
        // This avoids opening a new aux stream for each stream object.

        // NOTE: cudaStreamAddCallback is slated for deprecation, without a proper replacement (for now).
        // The other option we could use is cudaLaunchHostFunc, but it doesn't guarantee that the callback
        // will be called. We need this guarantee to make sure the object's refcount is eventually decremented,
        // and the closure is freed, avoiding memory leaks.
        // cudaLaunchHostFunc won't call the callback if the current cuda context is in error state, for instance.
        // Ref: CUDA SDK docs for both functions.
        util::CheckThrow(
            cudaStreamAddCallback(GetAuxStream(), fn, closure.get(), 0)); // add async callback in the aux stream
        closure.release();
    }
}

Stream::GCBag &Stream::GetGCBag()
{
    // By defining the gcBag inside this function instead of the global scope,
    // we guarantee that it'll be destroyed *before* the global python context
    // is destroyed. This is due to this function being called the first time
    // (via AddToGCBag or ClearGCBag) only after the python script (and python
    // ctx) has already started.
    //
    // Multi-GPU safety: this is a single process-wide bag holding closures from
    // all devices. This is safe because access is serialized by m_gcMutex, and
    // each HostFunctionClosure holds a shared_ptr<const Stream> whose destroy()
    // properly saves/restores the CUDA device before freeing device resources.
    static GCBag gcBag;
    return gcBag;
}

void Stream::AddToGCBag(std::unique_ptr<HostFunctionClosure> closure)
{
    std::unique_lock lk(m_gcMutex);
    GetGCBag().push_back(std::move(closure));
}

void Stream::ClearGCBag()
{
    GCBag objectsToBeDestroyed;

    GCBag &gcBag = GetGCBag();

    std::unique_lock lk(m_gcMutex);
    // Do as little as possible while mutex is locked to avoid
    // deadlocks.

    // In the case here, instead of simply empting up the gc bag,
    // which might trigger object destruction while the mutex is locked,
    // we move its contents to a temporary local bag.

    // take of benefit of ADL if available
    using std::swap;
    swap(objectsToBeDestroyed, gcBag);

    // Now the original bag is left empty, but no objects were
    // destroyed yet.
    NVCV_ASSERT(gcBag.empty()); // post-condition (can't be guaranteed after unlock)

    lk.unlock();

    // Let the local object bag go out of scope, the objects in it
    // will be finally destroyed with the mutex unlocked.
}

void Stream::SynchronizeAndClearGCBag()
{
    bool hasAuxStreams;
    {
        std::shared_lock lock(m_auxStreamMutex);
        hasAuxStreams = !m_auxStreams.empty();
    }

    if (hasAuxStreams)
    {
        SyncAuxStream();
    }
    ClearGCBag();
}

std::ostream &operator<<(std::ostream &out, const Stream &stream)
{
    return out << "<nvcv.cuda.Stream id=" << stream.id() << " handle=" << stream.handle() << '>';
}

template<ExternalStreamType E>
static void ExportExternalStream(py::module &m)
{
    m.def("as_stream",
          [](ExternalStream<E> extStream)
          {
              ::cvcudapy::NvtxRange nvtxRange("cvcuda.as_stream");
              return std::make_shared<Stream>(extStream);
          });
}

static void LogCleanupWarning(const std::exception &e)
{
    std::cerr << "Warning CVCUDA cleanup may be incomplete due to: " << e.what() << std::endl;
}

void Stream::CleanupAtExit(const std::shared_ptr<Stream> &globalStream)
{
    // No globalStream means no device at module-init time, no streams were ever
    // created, nothing to sync. Cleanup is a no-op.
    if (!globalStream)
    {
        return;
    }

    try
    {
        for (std::shared_ptr<Stream> stream : Cache::Instance().fetchAll<Stream>())
        {
            stream->sync();
        }
        globalStream->sync();
        SyncAuxStream();

        // There should only be 1 stream in the stack, namely the global stream.
        if (auto s = StreamStack::Instance().top(); s != globalStream)
        {
            std::cerr << "Stream stack leak detected" << std::endl;
        }

        // Make sure stream stack is empty.
        while (auto s = StreamStack::Instance().top())
        {
            StreamStack::Instance().pop();
        }

        // Make sure the gc bag is also cleaned up *after* all streams are done,
        // when all remaining items that need to be GC'd are in the bag.
        ClearGCBag();
    }
    catch (const py::error_already_set &e)
    {
        LogCleanupWarning(e);
    }
    catch (const StreamError &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::invalid_argument &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::domain_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::length_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::out_of_range &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::range_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::overflow_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::underflow_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::bad_alloc &e)
    {
        LogCleanupWarning(e);
    }
}

void Stream::Export(py::module &m)
{
    py::class_<Stream, std::shared_ptr<Stream>, CacheItem> stream(m, "Stream",
                                                                  R"pbdoc(
        A CUDA stream that cvcuda operators can be submitted to.

        Streams created via ``cvcuda.Stream()`` are non-blocking
        (``cudaStreamNonBlocking``) so they can run concurrently with CUDA's
        legacy default stream and with other non-blocking streams.

        Cross-library stream safety is handled automatically via the CUDA
        Array Interface (CAI) v3 ``stream`` field:

        * On **input**, ``cvcuda.as_tensor`` / ``cvcuda.as_image`` read
          ``__cuda_array_interface__["stream"]`` from the producer (cupy,
          torch, jax, etc.) and arrange for the first cvcuda operator to
          insert the proper ``cudaStreamWaitEvent`` before reading the data.
          No user-side ``synchronize()`` call is required.
        * On **output**, tensors returned by ``cvcuda.Tensor.cuda()`` /
          ``cvcuda.Image.cuda()`` populate ``__cuda_array_interface__
          ["stream"]`` with the cvcuda stream the data was last written on,
          so downstream consumers can sync themselves.

        Advanced usage:

        * **Opt out of the implicit wait**: export your buffer with
          ``stream: -1`` in its CAI dict to tell cvcuda you've already
          synchronized; the first op then runs without an extra event.
        * **Legacy default stream**: if your producer uses v2 CAI (no
          ``stream`` field) or advertises ``stream: 1``, cvcuda waits on the
          legacy default stream.  This is the safe conservative default but
          forfeits concurrency with stream-0 work.
        * **Use the default stream**: ``cvcuda.Stream.default`` wraps stream
          0 and has the usual legacy-default blocking semantics; use it when
          you want implicit sync with everything else on stream 0 and don't
          need the concurrency of a dedicated stream.
        )pbdoc");

    stream.def(py::init(&Stream::Create),
               "Create a new CUDA stream.  Streams created this way are non-blocking; "
               "cross-library synchronization is handled automatically via the CAI "
               "``stream`` field (see the class docstring).");

    py::module_ internal = m.attr(INTERNAL_SUBMODULE_NAME);
    internal.def("syncAuxStream", &SyncAuxStream);

    // Wrap the CUDA legacy default stream as `cvcuda.Stream.default` and seed
    // the per-thread stream stack so `cvcuda.Stream.current` has something to
    // return.
    //
    // Skipped on hosts with no CUDA device (CPU-only build/CI nodes,
    // CUDA_VISIBLE_DEVICES=""): wrapping stream 0 unavoidably touches the
    // CUDA runtime (e.g. the `Stream` ctor's cudaStreamGetFlags + aux-stream
    // bookkeeping), which fails without a driver-bound device. We also skip
    // registering the `current` static property in that case — there is no
    // sensible value, and exposing a getter that throws breaks introspection
    // tools (pybind11_stubgen, sphinx-autodoc) that walk class members.
    int deviceCount = 0;
    if (cudaError_t devCntErr = cudaGetDeviceCount(&deviceCount);
        devCntErr == cudaErrorNoDevice || devCntErr == cudaErrorStubLibrary)
    {
        (void)cudaGetLastError();
        deviceCount = 0;
    }
    std::shared_ptr<Stream> globalStream;
    if (deviceCount > 0)
    {
        stream.def_property_readonly_static(
            "current", [](py::object) { return Current().sharedStream(); },
            "Get the current CUDA stream for this thread.");

        static priv::ExternalStream<priv::VOIDP> cudaDefaultStream(static_cast<cudaStream_t>(nullptr));
        globalStream = std::make_shared<Stream>(cudaDefaultStream);
        StreamStack::Instance().push(*globalStream);
        stream.attr("default") = globalStream;
    }

    // Order from most specific to less specific
    ExportExternalStream<TORCH>(m);
    ExportExternalStream<CUPY>(m);
    ExportExternalStream<VOIDP>(m);
    ExportExternalStream<INT>(m);

    fflush(stdout);

    stream.def("__enter__", &Stream::activate, "Activate the CUDA stream as the current stream for this thread.")
        .def("__exit__", &Stream::deactivate, "Deactivate the CUDA stream as the current stream for this thread.")
        .def("sync", &Stream::sync, "Wait for all preceding CUDA calls in the current stream to complete.")
        .def("wait_stream", &Stream::wait_stream, py::arg("other"),
             "Insert a dependency on 'other' into this stream. All subsequent work enqueued on this stream "
             "will wait until all work currently enqueued on 'other' has completed. "
             "Calling wait_stream(self) is a no-op. "
             "Cross-device usage is not supported and will raise a CUDA error.")
        .def("__int__", &Stream::pyhandle, "Cast the CUDA stream object to an integer handle.")
        .def("__repr__", &util::ToString<Stream>, "Return a string representation of the CUDA stream object.")
        .def_property_readonly("handle", &Stream::pyhandle, "Get the integer handle for the CUDA stream object.")
        .def_property_readonly("id", &Stream::id, "Get the unique ID for the CUDA stream object.");

    // Make sure all streams we've created are synced when script ends.
    // Also make cleanup hold the globalStream reference during script execution.
    util::RegisterCleanup(m, [globalStream]() { CleanupAtExit(globalStream); });
}

} // namespace nvcvpy::priv
