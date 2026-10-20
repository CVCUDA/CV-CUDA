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

#ifndef NVCV_PYTHON_STREAM_HPP
#define NVCV_PYTHON_STREAM_HPP

#include "CAPI.hpp"

#include <common/Assert.hpp>
#include <cuda_runtime.h>

namespace nvcvpy {

namespace py = nanobind;

class Stream : public py::object
{
public:
    Stream() = default;

    static Stream Current()
    {
        py::object temp = py::reinterpret_borrow<py::object>(capi().Stream_GetCurrent());
        CheckCAPIError();
        NVCV_ASSERT(temp.is_none() == false);
        return Stream(temp);
    }

    cudaStream_t cudaHandle() const
    {
        cudaStream_t out = capi().Stream_GetCudaHandle(this->ptr());
        CheckCAPIError();
        return out;
    }

private:
    friend struct py::detail::type_caster<Stream>;

    py::object m_pyStream;

    explicit Stream(py::object obj)
        : py::object(obj)
    {
    }
};

} // namespace nvcvpy

namespace nanobind::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::Stream>
{
    NB_TYPE_CASTER(cvpy::Stream, const_name("cvcuda.Stream"));

    bool from_python(handle src, uint32_t, cleanup_list *)
    {
        if (cvpy::IsFacadeInstance<cvpy::FacadeType::STREAM>(src.ptr()))
        {
            value = cvpy::Stream(reinterpret_borrow<object>(src));
            return true;
        }
        return false;
    }

    static handle from_cpp(cvpy::Stream stream, rv_policy, cleanup_list *) noexcept
    {
        return stream.release();
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_STREAM_HPP
