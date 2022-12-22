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

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpMedianBlur.hpp>
#include <cvcuda/Types.h>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor MedianBlurInto(Tensor &output, Tensor &input, const std::tuple<int, int> &ksize, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto median_blur = CreateOperator<cvcuda::MedianBlur>(0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*median_blur});

    nvcv::Size2D ksizeArg{std::get<0>(ksize), std::get<1>(ksize)};

    median_blur->submit(pstream->cudaHandle(), input, output, ksizeArg);

    return output;
}

Tensor MedianBlur(Tensor &input, const std::tuple<int, int> &ksize, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return MedianBlurInto(output, input, ksize, pstream);
}

ImageBatchVarShape VarShapeMedianBlurInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &ksize,
                                          std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto median_blur = CreateOperator<cvcuda::MedianBlur>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, ksize});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*median_blur});

    median_blur->submit(pstream->cudaHandle(), input, output, ksize);

    return output;
}

ImageBatchVarShape VarShapeMedianBlur(ImageBatchVarShape &input, Tensor &ksize, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return VarShapeMedianBlurInto(output, input, ksize, pstream);
}

} // namespace

void ExportOpMedianBlur(py::module &m)
{
    using namespace pybind11::literals;

    m.def("median_blur", &MedianBlur, "src"_a, "ksize"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("median_blur_into", &MedianBlurInto, "dst"_a, "src"_a, "ksize"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("median_blur", &VarShapeMedianBlur, "src"_a, "ksize"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("median_blur_into", &VarShapeMedianBlurInto, "dst"_a, "src"_a, "ksize"_a, py::kw_only(),
          "stream"_a = nullptr);
}

} // namespace cvcudapy
