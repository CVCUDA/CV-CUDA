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

#include "Resource.hpp"

#include "Stream.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>

namespace nvcvpy::priv {

Resource::Resource()
{
    static uint64_t idnext = 0;

    m_id = idnext++;

    m_event = nullptr;
}

Resource::~Resource()
{
    cudaEventDestroy(m_event);
}

uint64_t Resource::id() const
{
    return m_id;
}

cudaEvent_t Resource::event()
{
    if (m_event == nullptr)
    {
        util::CheckThrow(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
    }
    return m_event;
}

void Resource::submitSync(Stream &stream)
{
    std::unique_lock<std::mutex> lk(m_mtx);
    //Check if we have a last stream, if not set it to the current stream
    if (!m_lastStream.has_value())
    {
        m_lastStream.emplace(stream.shared_from_this()); //store a shared pointer to the stream
    }

    // if we are on the same stream we dont need to do anything
    // as streams are sequential and we can assume that the last operation on the stream is done
    if (m_lastStream.value()->handle() == stream.handle())
    {
        return;
    }

    // if we are on a different stream we need to wait for that stream to finish
    // write event on the old stream, the new stream will have to wait for it to be done
    util::CheckThrow(cudaEventRecord(event(), m_lastStream.value()->handle()));
    util::CheckThrow(cudaStreamWaitEvent(stream.handle(), event()));

    // update the last stream since we changed streams
    m_lastStream.reset();
    m_lastStream.emplace(stream.shared_from_this());
}

std::shared_ptr<Resource> Resource::shared_from_this()
{
    return std::dynamic_pointer_cast<Resource>(Object::shared_from_this());
}

std::shared_ptr<const Resource> Resource::shared_from_this() const
{
    return std::dynamic_pointer_cast<const Resource>(Object::shared_from_this());
}

void Resource::Export(py::module &m)
{
    py::class_<Resource, std::shared_ptr<Resource>>(m, "Resource", "Resource")
        .def_property_readonly("id", &Resource::id, "Unique resource instance identifier")
        .def("submitStreamSync", &Resource::submitSync, "Syncs object on new Stream");
}

} // namespace nvcvpy::priv
