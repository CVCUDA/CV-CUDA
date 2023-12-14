/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PYTHON_SHAPE_HPP
#define NVCV_PYTHON_SHAPE_HPP

#include <nvcv/TensorShape.hpp>
#include <pybind11/pytypes.h>

namespace nvcvpy {

using Shape = pybind11::tuple;

inline Shape CreateShape(const nvcv::TensorShape &tshape)
{
    const auto &shape = tshape.shape();

    Shape s(shape.rank());
    for (int i = 0; i < shape.rank(); ++i)
    {
        s[i] = shape[i];
    }
    return s;
}

inline nvcv::TensorShape CreateNVCVTensorShape(const Shape &shape, nvcv::TensorLayout layout = nvcv::TENSOR_NONE)
{
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        dims.push_back(shape[i].cast<int64_t>());
    }

    return nvcv::TensorShape(dims.data(), dims.size(), layout);
}

} // namespace nvcvpy

#endif // NVCV_PYTHON_SHAPE_HPP
