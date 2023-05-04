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

#ifndef NVCV_PYTHON_PRIV_EXTERNAL_BUFFER_HPP
#define NVCV_PYTHON_PRIV_EXTERNAL_BUFFER_HPP

#include "DLPackUtils.hpp"

#include <cuda_runtime.h>
#include <nvcv/python/Shape.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvcvpy::priv {

namespace py = pybind11;

class ExternalBuffer final : public std::enable_shared_from_this<ExternalBuffer>
{
public:
    static void Export(py::module &m);

    ExternalBuffer(ExternalBuffer &&that) = delete;

    static py::object Create(DLPackTensor &&dlTensor, py::object wrappedObj);

    const DLTensor &dlTensor() const;

    Shape      shape() const;
    py::tuple  strides() const;
    py::object dtype() const;

    void *data() const;

    bool load(PyObject *o);

private:
    explicit ExternalBuffer(DLPackTensor &&dlTensor);

    friend py::detail::type_caster<ExternalBuffer>;
    ExternalBuffer() = default;

    DLPackTensor                    m_dlTensor;
    mutable std::optional<py::dict> m_cacheCudaArrayInterface;
    py::object                      m_wrappedObj;

    // Returns the __cuda_array_interface__ if the buffer is cuda-accessible,
    // or std::nullopt if it's not.
    std::optional<py::dict> cudaArrayInterface() const;

    // __dlpack__ implementation
    py::capsule dlpack(py::object stream) const;

    // __dlpack_device__ implementation
    py::tuple dlpackDevice() const;
};

} // namespace nvcvpy::priv

namespace PYBIND11_NAMESPACE { namespace detail {

namespace priv = nvcvpy::priv;

template<>
struct type_caster<priv::ExternalBuffer> : public type_caster_base<priv::ExternalBuffer>
{
    using type = priv::ExternalBuffer;
    using Base = type_caster_base<type>;

public:
    PYBIND11_TYPE_CASTER(std::shared_ptr<type>, const_name("nvcv.ExternalBuffer"));

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

#endif // NVCV_PYTHON_PRIV_EXTERNAL_BUFFER_HPP
