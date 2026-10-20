/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PYTHON_TENSORBATCH_HPP
#define NVCV_PYTHON_TENSORBATCH_HPP

#include "CAPI.hpp"
#include "Resource.hpp"

#include <common/Assert.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/python/Tensor.hpp>

#include <cassert>
#include <stdexcept>

namespace nvcvpy {

namespace py = nanobind;

class TensorBatch
    : public Resource
    , public nvcv::TensorBatch
{
public:
    TensorBatch(const TensorBatch &)     = default;
    TensorBatch(TensorBatch &&) noexcept = default;

    TensorBatch &operator=(const TensorBatch &)     = default;
    TensorBatch &operator=(TensorBatch &&) noexcept = default;

    static TensorBatch Create(int capacity)
    {
        // reinterpret_steal below takes a non-const PyObject *.
        PyObject *tensorBatch = capi().TensorBatch_Create(capacity); // NOSONAR
        CheckCAPIError();
        NVCV_ASSERT(tensorBatch != nullptr);
        py::object pytensorBatch = py::reinterpret_steal<py::object>(tensorBatch);

        return TensorBatch(pytensorBatch);
    }

    void pushBackTensor(const Tensor &tensor) // NOSONAR: mutates state through the wrapped C handle.
    {
        capi().TensorBatch_PushBack(this->ptr(), tensor.ptr());
        CheckCAPIError();
    }

    void popBackTensors(int cnt) // NOSONAR: mutates state through the wrapped C handle.
    {
        if (cnt < 0)
        {
            throw std::invalid_argument("cnt must be >= 0");
        }
        capi().TensorBatch_PopBack(this->ptr(), static_cast<uint32_t>(cnt));
        CheckCAPIError();
    }

    void clearTensors() // NOSONAR: mutates state through the wrapped C handle.
    {
        capi().TensorBatch_Clear(this->ptr());
        CheckCAPIError();
    }

    using nvcv::TensorBatch::operator[];
    using nvcv::TensorBatch::begin;
    using nvcv::TensorBatch::end;

private:
    friend struct py::detail::type_caster<TensorBatch>;

    TensorBatch() = default;

    explicit TensorBatch(py::object obj)
        : Resource(obj)
        , nvcv::TensorBatch(FromHandle(CheckCAPIError(capi().TensorBatch_GetHandle(this->ptr())), true))
    {
    }
};

} // namespace nvcvpy

namespace nanobind::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::TensorBatch>
{
    NB_TYPE_CASTER(cvpy::TensorBatch, const_name("cvcuda.TensorBatch"));

    bool from_python(handle src, uint32_t, cleanup_list *)
    {
        if (cvpy::IsFacadeInstance<cvpy::FacadeType::TENSOR_BATCH>(src.ptr()))
        {
            value = cvpy::TensorBatch(reinterpret_borrow<object>(src));
            return true;
        }
        return false;
    }

    static handle from_cpp(cvpy::TensorBatch tensor, rv_policy, cleanup_list *) noexcept
    {
        return static_cast<object &>(tensor).release();
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_TENSORBATCH_HPP
