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

#ifndef NVCV_PYTHON_DATATYPE_HPP
#define NVCV_PYTHON_DATATYPE_HPP

#include "CAPI.hpp"

#include <nvcv/DataType.hpp>

namespace pybind11::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<nvcv::DataType>
{
    PYBIND11_TYPE_CASTER(nvcv::DataType, const_name("nvcv.Type"));

    bool load(handle src, bool)
    {
        NVCVDataType p = cvpy::capi().DataType_FromPython(src.ptr());
        value          = nvcv::DataType(p);
        return true;
    }

    static handle cast(nvcv::DataType type, return_value_policy /* policy */, handle /*parent */)
    {
        return cvpy::capi().DataType_ToPython(static_cast<NVCVDataType>(type));
    }
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_DATATYPE_HPP
