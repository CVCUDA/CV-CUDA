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

#ifndef NVCV_PYTHON_SIZE_HPP
#define NVCV_PYTHON_SIZE_HPP

#include <nvcv/Size.hpp>

namespace nanobind::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<nvcv::Size2D>
{
    NB_TYPE_CASTER(nvcv::Size2D, const_name("cvcuda.Size2D"));

    bool from_python(handle src, uint32_t, cleanup_list *)
    {
        if (PyTuple_Check(src.ptr()))
        {
            nanobind::tuple t = nanobind::borrow<nanobind::tuple>(src);

            if (t.size() == 2)
            {
                value.w = nanobind::cast<int>(t[0]);
                value.h = nanobind::cast<int>(t[1]);
                return true;
            }
        }
        return false;
    }

    static handle from_cpp(nvcv::Size2D size, rv_policy, cleanup_list *)
    {
        return make_tuple(size.w, size.h).release();
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_SIZE_HPP
