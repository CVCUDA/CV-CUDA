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

#ifndef NVCV_PYTHON_PYUTIL_HPP
#define NVCV_PYTHON_PYUTIL_HPP

#include <common/Binding.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>

#include <functional>
#include <sstream>

namespace nvcvpy::util {

namespace py = nanobind;

inline bool HasPythonThreadState()
{
#ifdef Py_GIL_DISABLED
    return PyThreadState_GetUnchecked() != nullptr;
#else
    return PyGILState_Check() != 0;
#endif
}

class ScopedThreadDetach
{
public:
    ScopedThreadDetach() = default;

    ~ScopedThreadDetach()
    {
        if (m_state)
            PyEval_RestoreThread(m_state);
    }

    ScopedThreadDetach(const ScopedThreadDetach &)            = delete;
    ScopedThreadDetach &operator=(const ScopedThreadDetach &) = delete;

private:
    PyThreadState *m_state = HasPythonThreadState() ? PyEval_SaveThread() : nullptr;
};

void RegisterCleanup(py::module_ &m, std::function<void()> fn);

std::string GetFullyQualifiedName(py::handle h);

// We need to support array_interface formats and some numpy typestr
// formats, in addition to formats described in PEP 3118.
// This function will handle them all. nanobind as of v2.9.1 isn't capable
// of doing that.
//
// ref: https://peps.python.org/pep-3118/#additions-to-the-struct-string-syntax
// ref: https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_interface__
// ref: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
py::numpy_dtype ToDType(const std::string &str);
py::numpy_dtype ToDType(const py::buffer_info &info);
py::object      MakeNumpyArray(const py::buffer_info &info, py::object owner);

} // namespace nvcvpy::util

namespace cvcudapy {

namespace py = nanobind;

class NB_EXPORT_SHARED pyarray
{
public:
    using array_type = py::ndarray<const float, py::c_contig, py::device::cpu>;

    pyarray() = default;

    explicit pyarray(array_type array)
        : m_array(std::move(array))
    {
    }

    bool is_valid() const
    {
        return m_array.is_valid();
    }

    size_t ndim() const
    {
        return m_array.ndim();
    }

    size_t size() const
    {
        return m_array.size();
    }

    size_t shape(size_t dimension) const
    {
        if (const size_t numDimensions = ndim(); dimension >= numDimensions)
        {
            std::ostringstream message;
            message << "invalid axis: " << dimension << " (ndim = " << numDimensions << ')';
            throw py::index_error(message.str().c_str());
        }
        return m_array.shape(dimension);
    }

    const float *data() const
    {
        return m_array.data();
    }

    const array_type &array() const
    {
        return m_array;
    }

private:
    array_type m_array;
};

float4 GetFloat4FromPyArray(const pyarray &array);

} // namespace cvcudapy

namespace nanobind::detail {

template<>
struct type_caster<cvcudapy::pyarray>
{
    using Array = cvcudapy::pyarray::array_type;

    NB_TYPE_CASTER(cvcudapy::pyarray,
                   io_name("typing.Annotated[numpy.typing.ArrayLike, numpy.float32]", "numpy.ndarray"));

    bool from_python(handle src, uint32_t flags, cleanup_list *cleanup) noexcept
    {
        if (src.is_none() && !(flags & static_cast<uint32_t>(cast_flags::accepts_none)))
        {
            return false;
        }

        if (!src.is_none())
        {
            make_caster<Array> directCaster;
            if (directCaster.from_python(src, flags, cleanup))
            {
                value = cvcudapy::pyarray{std::move(directCaster.value)};
                return true;
            }
        }

        if (!(flags & static_cast<uint32_t>(cast_flags::convert)))
        {
            return false;
        }

        try
        {
            object numpy     = module_::import_("numpy");
            object converted = numpy.attr("require")(src, numpy.attr("dtype")("float32"), make_tuple("C"));

            make_caster<Array> convertedCaster;
            if (!convertedCaster.from_python(converted, flags, cleanup))
            {
                return false;
            }
            value = cvcudapy::pyarray{std::move(convertedCaster.value)};
            return true;
        }
        catch (...)
        {
            PyErr_Clear();
            return false;
        }
    }

    static handle from_cpp(const cvcudapy::pyarray &array, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        if (!array.is_valid())
        {
            return none().release();
        }
        return make_caster<Array>::from_cpp(array.array(), policy, cleanup);
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_PYUTIL_HPP
