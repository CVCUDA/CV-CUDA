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

#ifndef CVCUDAPY_OPERATORS_NATIVEVARSHAPEUTILS_HPP
#define CVCUDAPY_OPERATORS_NATIVEVARSHAPEUTILS_HPP

#include "../nvcv/Image.hpp"
#include "../nvcv/ImageBatch.hpp"

#include <nvcv/ImageFormat.hpp>

#include <memory>
#include <optional>

namespace cvcudapy {

inline std::optional<nvcv::ImageFormat> GetUniqueFormat(const nvcvpy::priv::ImageBatchVarShape &input)
{
    auto it = input.begin();
    if (it == input.end())
    {
        return std::nullopt;
    }

    nvcv::ImageFormat format = (*it)->format();
    for (++it; it != input.end(); ++it)
    {
        if ((*it)->format() != format)
        {
            return std::nullopt;
        }
    }
    return format;
}

inline std::shared_ptr<nvcvpy::priv::ImageBatchVarShape> CreateSameShapeImageBatch(
    const nvcvpy::priv::ImageBatchVarShape &input, nvcv::ImageFormat format, int capacity)
{
    auto output = nvcvpy::priv::ImageBatchVarShape::Create(capacity);
    for (const std::shared_ptr<nvcvpy::priv::Image> &image : input)
    {
        auto newImage = nvcvpy::priv::Image::Create(image->size(), format, 0);
        output->pushBack(*newImage);
    }
    return output;
}

inline std::shared_ptr<nvcvpy::priv::ImageBatchVarShape> CreateSameShapeImageBatch(
    const nvcvpy::priv::ImageBatchVarShape &input, nvcv::ImageFormat format)
{
    return CreateSameShapeImageBatch(input, format, input.capacity());
}

inline std::shared_ptr<nvcvpy::priv::ImageBatchVarShape> CreateSameShapeImageBatch(
    const nvcvpy::priv::ImageBatchVarShape &input, int capacity)
{
    auto output = nvcvpy::priv::ImageBatchVarShape::Create(capacity);
    for (const std::shared_ptr<nvcvpy::priv::Image> &image : input)
    {
        auto newImage = nvcvpy::priv::Image::Create(image->size(), image->format(), 0);
        output->pushBack(*newImage);
    }
    return output;
}

inline std::shared_ptr<nvcvpy::priv::ImageBatchVarShape> CreateSameShapeImageBatch(
    const nvcvpy::priv::ImageBatchVarShape &input)
{
    return CreateSameShapeImageBatch(input, input.capacity());
}

inline std::shared_ptr<nvcvpy::priv::ImageBatchVarShape> CreateSameShapeImageBatch(
    const nvcvpy::priv::ImageBatchVarShape &input, nvcvpy::priv::Stream &stream)
{
    std::vector<nvcvpy::priv::ImageBatchVarShape::OutputImage> images;
    images.reserve(input.numImages());
    for (const auto &image : input) images.push_back({image->size(), image->format()});
    return nvcvpy::priv::ImageBatchVarShape::CreateForOutput(input.capacity(), images, stream);
}

} // namespace cvcudapy

#endif // CVCUDAPY_OPERATORS_NATIVEVARSHAPEUTILS_HPP
