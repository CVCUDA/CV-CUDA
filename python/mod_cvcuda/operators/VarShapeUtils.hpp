/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDAPY_OPERATORS_VARSHAPEUTILS_HPP
#define CVCUDAPY_OPERATORS_VARSHAPEUTILS_HPP

#include <nvcv/ImageFormat.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>

#include <tuple>
#include <vector>

namespace cvcudapy {

inline nvcvpy::ImageBatchVarShape CreateSameShapeImageBatch(const nvcvpy::ImageBatchVarShape &input,
                                                            nvcv::ImageFormat format, int capacity)
{
    nvcvpy::ImageBatchVarShape output = nvcvpy::ImageBatchVarShape::Create(capacity);

    for (int i = 0; i < input.numImages(); ++i)
    {
        output.pushBackImage(nvcvpy::Image::Create(input[i].size(), format));
    }

    return output;
}

inline nvcvpy::ImageBatchVarShape CreateSameShapeImageBatch(const nvcvpy::ImageBatchVarShape &input,
                                                            nvcv::ImageFormat                 format)
{
    return CreateSameShapeImageBatch(input, format, input.capacity());
}

inline nvcvpy::ImageBatchVarShape CreateSameShapeImageBatch(const nvcvpy::ImageBatchVarShape &input, int capacity)
{
    nvcvpy::ImageBatchVarShape output = nvcvpy::ImageBatchVarShape::Create(capacity);

    for (int i = 0; i < input.numImages(); ++i)
    {
        output.pushBackImage(nvcvpy::Image::Create(input[i].size(), input[i].format()));
    }

    return output;
}

inline nvcvpy::ImageBatchVarShape CreateSameShapeImageBatch(const nvcvpy::ImageBatchVarShape &input)
{
    return CreateSameShapeImageBatch(input, input.capacity());
}

inline nvcvpy::ImageBatchVarShape CreateSizedImageBatch(const nvcvpy::ImageBatchVarShape        &input,
                                                        const std::vector<std::tuple<int, int>> &sizes, int capacity)
{
    nvcvpy::ImageBatchVarShape output = nvcvpy::ImageBatchVarShape::Create(capacity);

    for (int i = 0; i < input.numImages(); ++i)
    {
        auto [size0, size1] = sizes[i];
        output.pushBackImage(nvcvpy::Image::Create({size0, size1}, input[i].format()));
    }

    return output;
}

inline nvcvpy::ImageBatchVarShape CreateSizedImageBatch(const nvcvpy::ImageBatchVarShape        &input,
                                                        const std::vector<std::tuple<int, int>> &sizes)
{
    return CreateSizedImageBatch(input, sizes, input.capacity());
}

} // namespace cvcudapy

#endif // CVCUDAPY_OPERATORS_VARSHAPEUTILS_HPP
