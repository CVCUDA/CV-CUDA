/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Here we define the representation of external cuda streams.
// It defines pybind11's type casters from the python object
// to the corresponding ExternalStream<E>.

// Defines each external stream represetation we support.
enum ExternalStreamType
{
    VOIDP,
    INT,
    TORCH,
    NUMBA,
};

template<ExternalStreamType E>
class ExternalStream : public IExternalStream
{
public:
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

template<>
struct type_caster<priv::ExternalStream<priv::NUMBA>>
{
    PYBIND11_TYPE_CASTER(priv::ExternalStream<priv::NUMBA>, const_name("numba.cuda.Stream"));

    bool load(handle src, bool)
    {
        std::string strType = util::GetFullyQualifiedName(src);

        if (strType != "numba.cuda.cudadrv.driver.Stream")
        {
            return false;
        }

        try
        {
            // NUMBA cuda stream can be converted to ints, which is the cudaStream handle.
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

}} // namespace PYBIND11_NAMESPACE::detail

namespace nvcvpy::priv {

// In terms of caching, all streams are the same.
// Any stream in the cache can be fetched and used.
size_t Stream::Key::doGetHash() const
{
    return 0;
}

bool Stream::Key::doIsEqual(const IKey &that) const
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
    util::CheckThrow(cudaStreamCreate(&m_handle));
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
}

Stream::~Stream()
{
    if (m_owns)
    {
        util::CheckLog(cudaStreamSynchronize(m_handle));
        util::CheckLog(cudaStreamDestroy(m_handle));
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

        util::CheckThrow(cudaStreamAddCallback(m_handle, fn, closure.get(), 0));

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

    stream.def_property_readonly_static("current", [](py::object) { return Current().shared_from_this(); })
        .def(py::init(&Stream::Create));

    // Create the global stream object. It'll be destroyed when
    // python module is deinitialized.
    auto globalStream = Stream::Create();
    StreamStack::Instance().push(*globalStream);
    stream.attr("default") = globalStream;

    // Order from most specific to less specific
    ExportExternalStream<TORCH>(m);
    ExportExternalStream<NUMBA>(m);
    ExportExternalStream<VOIDP>(m);
    ExportExternalStream<INT>(m);

    stream.def("__enter__", &Stream::activate)
        .def("__exit__", &Stream::deactivate)
        .def("sync", &Stream::sync)
        .def("__int__", &Stream::pyhandle)
        .def("__repr__", &util::ToString<Stream>)
        .def_property_readonly("handle", &Stream::pyhandle)
        .def_property_readonly("id", &Stream::id);

    // Make sure all streams we've created are synced when script ends.
    // Also make cleanup hold the globalStream reference during script execution.
    util::RegisterCleanup(m,
                          [globalStream]()
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
                          });
}

} // namespace nvcvpy::priv
