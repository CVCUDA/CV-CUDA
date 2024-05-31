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

#include "Stream.hpp"

#include "Cache.hpp"
#include "StreamStack.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <pybind11/operators.h>

namespace nvcvpy::priv {

// Static members initialization
cudaStream_t     Stream::m_auxStream     = nullptr;
std::atomic<int> Stream::m_instanceCount = 0;
std::mutex       Stream::m_auxStreamMutex;

// Here we define the representation of external cuda streams.
// It defines pybind11's type casters from the python object
// to the corresponding ExternalStream<E>.

// Defines each external stream represetation we support.
enum ExternalStreamType
{
    VOIDP,
    INT,
    TORCH,
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

    virtual cudaStream_t handle() const override
    {
        return m_cudaStream;
    }

    virtual py::object wrappedObject() const override
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
        std::string strType = util::GetFullyQualifiedName(src);

        if (strType != "ctypes.c_void_p")
        {
            return false;
        }

        buffer_info info = ::pybind11::cast<buffer>(src).request();

        NVCV_ASSERT(info.itemsize == sizeof(void *));

        void *data = *reinterpret_cast<void **>(info.ptr);

        value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(std::move(src)));
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
            // TODO: don't know how to test if a python object
            // is convertible to a type without exceptions.
            intptr_t data = src.cast<intptr_t>();
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(std::move(src)));
            return true;
        }
        catch (...)
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
        std::string strType = util::GetFullyQualifiedName(src);

        if (strType != "torch.cuda.streams.Stream" && strType != "torch.cuda.streams.ExternalStream")
        {
            return false;
        }

        try
        {
            // TODO: don't know how to test if a python object
            // is convertible to a type without exceptions.
            intptr_t data = src.attr("cuda_stream").cast<intptr_t>();
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(std::move(src)));
            return true;
        }
        catch (...)
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
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Stream::Key{});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Stream> stream(new Stream());
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
{
    try
    {
        util::CheckThrow(cudaStreamCreateWithFlags(&m_handle, cudaStreamNonBlocking));
        incrementInstanceCount();
        GetAuxStream();
        util::CheckThrow(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
    }
    catch (...)
    {
        destroy();
        throw;
    }
}

Stream::Stream(IExternalStream &extStream)
    : m_owns(false)
    , m_handle(extStream.handle())
    , m_wrappedObj(std::move(extStream.wrappedObject()))
{
    unsigned int flags;
    if (cudaStreamGetFlags(m_handle, &flags) != cudaSuccess)
    {
        throw std::runtime_error("Invalid cuda stream");
    }

    try
    {
        incrementInstanceCount();
        GetAuxStream(); // Make sure the singleton aux stream is created
        util::CheckThrow(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
    }
    catch (...)
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

cudaStream_t &Stream::GetAuxStream()
{
    if (!m_auxStream)
    {
        std::lock_guard<std::mutex> lock(m_auxStreamMutex);
        if (!m_auxStream)
        {
            util::CheckThrow(cudaStreamCreateWithFlags(&m_auxStream, cudaStreamNonBlocking));
        }
    }
    return m_auxStream;
}

Stream::~Stream()
{
    destroy();
}

void Stream::destroy()
{
    if (m_owns)
    {
        if (m_handle)
        {
            util::CheckLog(cudaStreamSynchronize(m_handle));
            util::CheckLog(cudaStreamDestroy(m_handle));
            m_handle = nullptr;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_auxStreamMutex);
        if (m_auxStream && decrementInstanceCount() == 0)
        {
            util::CheckThrow(cudaStreamSynchronize(m_auxStream));
            util::CheckThrow(cudaStreamDestroy(m_auxStream));
            m_auxStream = nullptr;
        }
    }
    if (m_event)
    {
        util::CheckThrow(cudaEventDestroy(m_event));
        m_event = nullptr;
    }
}

std::shared_ptr<Stream> Stream::shared_from_this()
{
    return std::dynamic_pointer_cast<Stream>(Object::shared_from_this());
}

std::shared_ptr<const Stream> Stream::shared_from_this() const
{
    return std::dynamic_pointer_cast<const Stream>(Object::shared_from_this());
}

cudaStream_t Stream::handle() const
{
    return m_handle;
}

intptr_t Stream::pyhandle() const
{
    return reinterpret_cast<intptr_t>(m_handle);
}

void Stream::sync()
{
    py::gil_scoped_release release;
    util::CheckThrow(cudaStreamSynchronize(m_handle));
}

Stream &Stream::Current()
{
    auto defStream = StreamStack::Instance().top();
    NVCV_ASSERT(defStream);
    return *defStream;
}

void Stream::activate()
{
    StreamStack::Instance().push(*this);
}

void Stream::deactivate(py::object exc_type, py::object exc_value, py::object exc_tb)
{
    StreamStack::Instance().pop();
}

void Stream::holdResources(LockResources usedResources)
{
    struct HostFunctionClosure
    {
        // Also hold the stream reference so that it isn't destroyed before the processing is done.
        std::shared_ptr<const Stream> stream;
        LockResources                 resources;
    };

    if (!usedResources.empty())
    {
        auto closure = std::make_unique<HostFunctionClosure>();

        closure->stream    = this->shared_from_this();
        closure->resources = std::move(usedResources);

        auto fn = [](cudaStream_t stream, cudaError_t error, void *userData) -> void
        {
            auto *pclosure = reinterpret_cast<HostFunctionClosure *>(userData);
            delete pclosure;
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

        util::CheckThrow(cudaEventRecord(m_event, m_handle)); // add async record the event in the main stream
        util::CheckThrow(
            cudaStreamWaitEvent(GetAuxStream(), m_event)); // add async wait for the event in the aux stream
        // The callback will be executed in the singleton aux stream there may be contention with other callbacks and waitEvents from
        // other streams. However the callback is used to release resources from the cache and should not be a performance bottleneck.
        // This avoids opening a new aux stream for each stream object.
        util::CheckThrow(
            cudaStreamAddCallback(GetAuxStream(), fn, closure.get(), 0)); // add async callback in the aux stream
        closure.release();
    }
}

std::ostream &operator<<(std::ostream &out, const Stream &stream)
{
    return out << "<nvcv.cuda.Stream id=" << stream.id() << " handle=" << stream.handle() << '>';
}

template<ExternalStreamType E>
static void ExportExternalStream(py::module &m)
{
    m.def("as_stream", [](ExternalStream<E> extStream) { return std::shared_ptr<Stream>(new Stream(extStream)); });
}

void Stream::Export(py::module &m)
{
    py::class_<Stream, std::shared_ptr<Stream>> stream(m, "Stream");

    stream
        .def_property_readonly_static(
            "current", [](py::object) { return Current().shared_from_this(); },
            "Get the current CUDA stream for this thread.")
        .def(py::init(&Stream::Create), "Create a new CUDA stream.");

    // Create the global stream object by wrapping cuda stream 0.
    // It'll be destroyed when python module is deinitialized.
    static priv::ExternalStream<priv::VOIDP> cudaDefaultStream((cudaStream_t)0);
    auto                                     globalStream = std::make_shared<Stream>(cudaDefaultStream);
    StreamStack::Instance().push(*globalStream);
    stream.attr("default") = globalStream;

    // Order from most specific to less specific
    ExportExternalStream<TORCH>(m);
    ExportExternalStream<VOIDP>(m);
    ExportExternalStream<INT>(m);

    fflush(stdout);

    stream.def("__enter__", &Stream::activate, "Activate the CUDA stream as the current stream for this thread.")
        .def("__exit__", &Stream::deactivate, "Deactivate the CUDA stream as the current stream for this thread.")
        .def("sync", &Stream::sync, "Wait for all preceding CUDA calls in the current stream to complete.")
        .def("__int__", &Stream::pyhandle, "Cast the CUDA stream object to an integer handle.")
        .def("__repr__", &util::ToString<Stream>, "Return a string representation of the CUDA stream object.")
        .def_property_readonly("handle", &Stream::pyhandle, "Get the integer handle for the CUDA stream object.")
        .def_property_readonly("id", &Stream::id, "Get the unique ID for the CUDA stream object.");

    // Make sure all streams we've created are synced when script ends.
    // Also make cleanup hold the globalStream reference during script execution.
    util::RegisterCleanup(m,
                          [globalStream]()
                          {
                              try
                              {
                                  for (std::shared_ptr<Stream> stream : Cache::Instance().fetchAll<Stream>())
                                  {
                                      stream->sync();
                                  }
                                  globalStream->sync();

                                  // There should only be 1 stream in the stack, namely the
                                  // global stream.
                                  auto s = StreamStack::Instance().top();
                                  if (s != globalStream)
                                  {
                                      std::cerr << "Stream stack leak detected" << std::endl;
                                  }

                                  // Make sure stream stack is empty
                                  while (auto s = StreamStack::Instance().top())
                                  {
                                      StreamStack::Instance().pop();
                                  }
                              }
                              catch (const std::exception &e)
                              {
                                  //Do nothing here this can happen if someone closes the cuda context prior to exit.
                                  std::cerr << "Warning CVCUDA cleanup may be incomplete due to: " << e.what() << "\n";
                              }
                          });
}

} // namespace nvcvpy::priv
