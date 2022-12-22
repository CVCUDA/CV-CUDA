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

#ifndef NVCV_PYTHON_PRIV_CUDA_BUFFER_HPP
#define NVCV_PYTHON_PRIV_CUDA_BUFFER_HPP

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvcvpy::priv {

namespace py = pybind11;

class CudaBuffer final : public std::enable_shared_from_this<CudaBuffer>
{
public:
    static void Export(py::module &m);

    CudaBuffer(CudaBuffer &&that) = delete;

    explicit CudaBuffer(const py::buffer_info &data, bool copy = false, py::object wrappedObj = {});

    ~CudaBuffer();

    py::dict cuda_interface() const;

    py::buffer_info request(bool writable = false) const;

    py::object shape() const;
    py::object dtype() const;

    void *data() const;

    bool load(PyObject *o);

private:
    friend py::detail::type_caster<CudaBuffer>;
    CudaBuffer();

    py::object m_wrappedObj;
    py::dict   m_cudaArrayInterface;
    bool       m_owns;
};

} // namespace nvcvpy::priv

namespace PYBIND11_NAMESPACE { namespace detail {

namespace priv = nvcvpy::priv;

template<>
struct type_caster<priv::CudaBuffer> : public type_caster_base<priv::CudaBuffer>
{
    using type = priv::CudaBuffer;
    using Base = type_caster_base<type>;

public:
    PYBIND11_TYPE_CASTER(std::shared_ptr<type>, const_name("nvcv.cuda.Buffer"));

    operator type *()
    {
        return value.get();
    }

    operator type &()
    {
        return *value;
    }

    bool load(handle src, bool);
};

}} // namespace PYBIND11_NAMESPACE::detail

#endif // NVCV_PYTHON_PRIV_CUDA_BUFFER_HPP
