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

#ifndef NVCV_PYTHON_SIZE_HPP
#define NVCV_PYTHON_SIZE_HPP

#include <nvcv/Size.hpp>

namespace pybind11::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<nvcv::Size2D>
{
    PYBIND11_TYPE_CASTER(nvcv::Size2D, const_name("nvcv.Size2D"));

    bool load(handle src, bool)
    {
        if (PyTuple_Check(src.ptr()))
        {
            tuple t = src.cast<tuple>();

            if (t.size() == 2)
            {
                value.w = t[0].cast<int>();
                value.h = t[1].cast<int>();
                return true;
            }
        }
        return false;
    }

    static handle cast(nvcv::Size2D size, return_value_policy /* policy */, handle /*parent */)
    {
        return make_tuple(size.w, size.h);
    }
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_SIZE_HPP
