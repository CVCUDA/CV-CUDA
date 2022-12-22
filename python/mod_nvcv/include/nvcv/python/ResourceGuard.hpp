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

#ifndef NVCV_PYTHON_RESOURCE_GUARD_HPP
#define NVCV_PYTHON_RESOURCE_GUARD_HPP

#include "CAPI.hpp"
#include "LockMode.hpp"
#include "Resource.hpp"
#include "Stream.hpp"

namespace nvcvpy {

namespace py = pybind11;

class ResourceGuard
{
public:
    ResourceGuard(Stream &stream)
        : m_pyStream(stream)
    {
    }

    ~ResourceGuard()
    {
        this->commit();
    }

    ResourceGuard &add(LockMode mode, std::initializer_list<std::reference_wrapper<const Resource>> resources)
    {
        py::object pyLockMode;
        switch (mode)
        {
        case LockMode::LOCK_NONE:
            pyLockMode = py::str("");
            break;
        case LockMode::LOCK_READ:
            pyLockMode = py::str("r");
            break;
        case LockMode::LOCK_WRITE:
            pyLockMode = py::str("w");
            break;
        case LockMode::LOCK_READWRITE:
            pyLockMode = py::str("rw");
            break;
        }

        for (const std::reference_wrapper<const Resource> &r : resources)
        {
            py::object pyRes = r.get();

            capi().Resource_SubmitSync(pyRes.ptr(), m_pyStream.ptr(), pyLockMode.ptr());
            m_resourcesPerLockMode.append(std::make_pair(pyLockMode, std::move(pyRes)));
        }
        return *this;
    }

    void commit()
    {
        capi().Stream_HoldResources(m_pyStream.ptr(), m_resourcesPerLockMode.ptr());

        py::list newList;

        auto it = m_resourcesPerLockMode.begin();
        try
        {
            // Try to signal the resources, stop on the first that fails, or
            // when all resources were signaled
            for (; it != m_resourcesPerLockMode.end(); ++it)
            {
                py::tuple t = it->cast<py::tuple>();

                // resource, stream, lockmode
                capi().Resource_SubmitSignal(t[1].ptr(), m_pyStream.ptr(), t[0].ptr());
            }
        }
        catch (...)
        {
            // Add all resources that weren't signaled to the newList.
            for (; it != m_resourcesPerLockMode.end(); ++it)
            {
                newList.append(std::move(*it));
            }
            throw;
        }

        m_resourcesPerLockMode = std::move(newList);
    }

private:
    py::object m_pyStream;
    py::object m_pyLockMode;
    py::list   m_resourcesPerLockMode;
};

} // namespace nvcvpy

#endif // NVCV_PYTHON_RESOURCE_GUARD_HPP
