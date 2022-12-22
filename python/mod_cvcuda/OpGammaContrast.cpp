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
#include <cvcuda/OpGammaContrast.hpp>
#include <cvcuda/Types.h>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
ImageBatchVarShape VarShapeGammaContrastInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &gamma,
                                             std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto gamma_contrast = CreateOperator<cvcuda::GammaContrast>(input.capacity(), input.uniqueFormat().numChannels());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, gamma});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*gamma_contrast});

    gamma_contrast->submit(pstream->cudaHandle(), input, output, gamma);

    return output;
}

ImageBatchVarShape VarShapeGammaContrast(ImageBatchVarShape &input, Tensor &gamma, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return VarShapeGammaContrastInto(output, input, gamma, pstream);
}

} // namespace

void ExportOpGammaContrast(py::module &m)
{
    using namespace pybind11::literals;

    m.def("gamma_contrast", &VarShapeGammaContrast, "src"_a, "gamma"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("gamma_contrast_into", &VarShapeGammaContrastInto, "dst"_a, "src"_a, "gamma"_a, py::kw_only(),
          "stream"_a = nullptr);
}

} // namespace cvcudapy
