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

#include "PyUtil.hpp"

#include "Assert.hpp"
#include "String.hpp"

#include <cstdlib>
#include <list>

namespace nvcvpy::util {

namespace {
// Class that holds cleanup functions that are called when python is shutting down.
class Cleanup
{
public:
    Cleanup()
    {
        NVCV_ASSERT(m_instance == nullptr && "Only one instance of Cleanup can be instantiated");
        m_instance = this;

        // Functions registered with python's atexit will be executed after
        // the script ends without fatal errors, but before python interpreter is torn down.
        // That's a good place for our cleanup.
        auto py_atexit = py::module_::import("atexit");
        py_atexit.attr("register")(py::cpp_function(&doCleanup));
    }

    ~Cleanup()
    {
        doCleanup(); // last chance for cleaning up
        m_instance = nullptr;
    }

    void addHandler(std::function<void()> fn)
    {
        m_handlers.emplace_back(std::move(fn));
    }

private:
    static Cleanup                  *m_instance;
    std::list<std::function<void()>> m_handlers;

    static void doCleanup()
    {
        NVCV_ASSERT(m_instance != nullptr);

        while (!m_instance->m_handlers.empty())
        {
            std::function<void()> handler = m_instance->m_handlers.back();
            m_instance->m_handlers.pop_back();

            handler();
        }
    }
};

Cleanup *Cleanup::m_instance = nullptr;
} // namespace

void RegisterCleanup(py::module &m, std::function<void()> fn)
{
    static Cleanup cleanup;
    cleanup.addHandler(std::move(fn));
}

std::string GetFullyQualifiedName(py::handle h)
{
    py::handle type = h.get_type();

    std::ostringstream ss;
    ss << type.attr("__module__").cast<std::string>() << '.' << type.attr("__qualname__").cast<std::string>();
    return ss.str();
}

static std::string ProcessBufferInfoFormat(const std::string &fmt)
{
    // pybind11 (as of v2.6.2) doesn't recognize formats 'l' and 'L',
    // which according to https://docs.python.org/3/library/struct.html#format-characters
    // are equal to 'i' and 'I', respectively.
    if (fmt == "l")
    {
        return "i";
    }
    else if (fmt == "L")
    {
        return "I";
    }
    else
    {
        return fmt;
    }
}

py::dtype ToDType(const py::buffer_info &info)
{
    std::string fmt = ProcessBufferInfoFormat(info.format);

    PyObject *ptr = nullptr;
    if ((py::detail::npy_api::get().PyArray_DescrConverter_(py::str(fmt).ptr(), &ptr) == 0) || !ptr)
    {
        PyErr_Clear();
        return py::dtype(info);
    }
    else
    {
        return py::dtype(fmt);
    }
}

py::dtype ToDType(const std::string &fmt)
{
    py::buffer_info buf;
    buf.format = ProcessBufferInfoFormat(fmt);

    return ToDType(buf);
}

} // namespace nvcvpy::util
