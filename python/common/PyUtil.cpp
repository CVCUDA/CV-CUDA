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

#include "PyUtil.hpp"

#include "Assert.hpp"
#include "String.hpp"

#include <cstdlib>
#include <list>
#include <stdexcept>

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
        auto py_atexit = py::module_::import_("atexit");
        py_atexit.attr("register")(py::cpp_function(&doCleanup));
    }

    ~Cleanup()
    {
        doCleanup(); // last chance for cleaning up
        m_instance = nullptr;
    }

    Cleanup(const Cleanup &)            = delete;
    Cleanup(Cleanup &&)                 = delete;
    Cleanup &operator=(const Cleanup &) = delete;
    Cleanup &operator=(Cleanup &&)      = delete;

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

void RegisterCleanup(py::module_ &, std::function<void()> fn)
{
    static Cleanup cleanup;
    cleanup.addHandler(std::move(fn));
}

std::string GetFullyQualifiedName(py::handle h)
{
    py::handle type = h.type();

    std::ostringstream ss;
    ss << py::cast<std::string>(type.attr("__module__")) << '.' << py::cast<std::string>(type.attr("__qualname__"));
    return ss.str();
}

static std::string ProcessBufferInfoFormat(const std::string &fmt)
{
    // nanobind (as of v2.6.2) doesn't recognize formats 'l' and 'L',
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
    else if (fmt == "Zf")
    {
        return "F";
    }
    else if (fmt == "Zd")
    {
        return "D";
    }
    else
    {
        return fmt;
    }
}

py::numpy_dtype ToDType(const py::buffer_info &info)
{
    std::string fmt = ProcessBufferInfoFormat(info.format);
    return py::numpy_dtype(fmt);
}

py::numpy_dtype ToDType(const std::string &fmt)
{
    return py::numpy_dtype(ProcessBufferInfoFormat(fmt));
}

py::object MakeNumpyArray(const py::buffer_info &info, py::object owner)
{
    py::numpy_dtype dtype = ToDType(info);

    py::dlpack::dtype arrayType;
    switch (dtype.kind())
    {
    case 'b':
        arrayType.code = static_cast<uint8_t>(py::dlpack::dtype_code::Bool);
        break;
    case 'i':
        arrayType.code = static_cast<uint8_t>(py::dlpack::dtype_code::Int);
        break;
    case 'u':
        arrayType.code = static_cast<uint8_t>(py::dlpack::dtype_code::UInt);
        break;
    case 'f':
        arrayType.code = static_cast<uint8_t>(py::dlpack::dtype_code::Float);
        break;
    case 'c':
        arrayType.code = static_cast<uint8_t>(py::dlpack::dtype_code::Complex);
        break;
    default:
        throw std::invalid_argument("Unsupported NumPy dtype kind");
    }
    arrayType.bits  = static_cast<uint8_t>(info.itemsize * 8);
    arrayType.lanes = 1;

    std::vector<size_t>  shape(info.shape.begin(), info.shape.end());
    std::vector<int64_t> strides;
    strides.reserve(info.strides.size());
    for (ssize_t stride : info.strides)
    {
        if (stride % info.itemsize != 0)
        {
            throw std::invalid_argument("Array stride must be a multiple of its item size");
        }
        strides.push_back(stride / info.itemsize);
    }

    py::ndarray<py::numpy> array(info.ptr, shape.size(), shape.data(), owner, strides.data(), arrayType,
                                 py::device::cpu::value);
    return array.cast();
}

} // namespace nvcvpy::util

namespace cvcudapy {

namespace {

class PyUtilError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace

float4 GetFloat4FromPyArray(const pyarray &array)
{
    if (!array.is_valid())
    {
        return nvcv::cuda::SetAll<float4>(0.f);
    }
    else if (array.ndim() == 0)
    {
        return nvcv::cuda::SetAll<float4>(*array.data());
    }
    else
    {
        if (array.ndim() != 1 || array.size() > 4)
        {
            throw PyUtilError(nvcvpy::util::ConcatString(
                "borderValue must be a scalar or 1D array with at most 4 elements, current ndim is '", array.ndim(),
                "' and size is '", array.size(), "'"));
        }

        float4 value = nvcv::cuda::SetAll<float4>(0.f);

        for (int i = 0; i < static_cast<int>(array.size()); i++)
        {
            nvcv::cuda::GetElement(value, i) = array.data()[i];
        }

        return value;
    }
}

} // namespace cvcudapy
