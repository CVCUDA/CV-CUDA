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

#include "Resource.hpp"

#include "Stream.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>

#include <iostream>

namespace nvcvpy::priv {

Resource::Resource()
{
    static uint64_t idnext = 0;

    m_id = idnext++;

    m_readEvent = m_writeEvent = nullptr;
    try
    {
        util::CheckThrow(cudaEventCreateWithFlags(&m_readEvent, cudaEventDisableTiming));
        util::CheckThrow(cudaEventCreateWithFlags(&m_writeEvent, cudaEventDisableTiming));
    }
    catch (...)
    {
        cudaEventDestroy(m_readEvent);
        cudaEventDestroy(m_writeEvent);
        throw;
    }
}

Resource::~Resource()
{
    cudaEventDestroy(m_readEvent);
    cudaEventDestroy(m_writeEvent);
}

uint64_t Resource::id() const
{
    return m_id;
}

void Resource::submitSignal(Stream &stream, LockMode mode) const
{
    doBeforeSubmitSignal(stream, mode);

    if (mode & LOCK_READ)
    {
        util::CheckThrow(cudaEventRecord(m_readEvent, stream.handle()));
    }
    if (mode & LOCK_WRITE)
    {
        util::CheckThrow(cudaEventRecord(m_writeEvent, stream.handle()));
    }
}

void Resource::submitSync(Stream &stream, LockMode mode) const
{
    doBeforeSubmitSync(stream, mode);

    doSubmitSync(stream, mode);
}

void Resource::doSubmitSync(Stream &stream, LockMode mode) const
{
    if (mode & LOCK_READ)
    {
        util::CheckThrow(cudaStreamWaitEvent(stream.handle(), m_writeEvent));
    }
    if (mode & LOCK_WRITE)
    {
        util::CheckThrow(cudaStreamWaitEvent(stream.handle(), m_writeEvent));
        util::CheckThrow(cudaStreamWaitEvent(stream.handle(), m_readEvent));
    }
}

void Resource::sync(LockMode mode) const
{
    py::gil_scoped_release release;

    doBeforeSync(mode);

    doSync(mode);
}

void Resource::doSync(LockMode mode) const
{
    NVCV_ASSERT(PyGILState_Check() == 0);

    if (mode & LOCK_READ)
    {
        util::CheckThrow(cudaEventSynchronize(m_writeEvent));
    }
    if (mode & LOCK_WRITE)
    {
        util::CheckThrow(cudaEventSynchronize(m_writeEvent));
        util::CheckThrow(cudaEventSynchronize(m_readEvent));
    }
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
    py::class_<Resource, std::shared_ptr<Resource>>(m, "Resource")
        .def_property_readonly("id", &Resource::id, "Unique resource instance identifier")
        .def("submitSync", &Resource::submitSync)
        .def("submitSignal", &Resource::submitSignal);
}

} // namespace nvcvpy::priv
