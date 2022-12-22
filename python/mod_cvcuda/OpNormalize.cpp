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
#include <cvcuda/OpNormalize.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {

enum OpFlags : uint32_t
{
    SCALE_IS_STDDEV = CVCUDA_NORMALIZE_SCALE_IS_STDDEV
};

} // namespace

namespace {
Tensor NormalizeInto(Tensor &output, Tensor &input, Tensor &base, Tensor &scale, std::optional<uint32_t> flags,
                     float globalScale, float globalShift, float epsilon, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (!flags)
    {
        flags = 0;
    }

    auto normalize = CreateOperator<cvcuda::Normalize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, base, scale});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*normalize});

    normalize->submit(pstream->cudaHandle(), input, base, scale, output, globalScale, globalShift, epsilon, *flags);

    return std::move(output);
}

Tensor Normalize(Tensor &input, Tensor &base, Tensor &scale, std::optional<uint32_t> flags, float globalScale,
                 float globalShift, float epsilon, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return NormalizeInto(output, input, base, scale, flags, globalScale, globalShift, epsilon, pstream);
}

ImageBatchVarShape VarShapeNormalizeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &base,
                                         Tensor &scale, std::optional<uint32_t> flags, float globalScale,
                                         float globalShift, float epsilon, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (!flags)
    {
        flags = 0;
    }

    auto normalize = CreateOperator<cvcuda::Normalize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, base, scale});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*normalize});

    normalize->submit(pstream->cudaHandle(), input, base, scale, output, globalScale, globalShift, epsilon, *flags);

    return output;
}

ImageBatchVarShape VarShapeNormalize(ImageBatchVarShape &input, Tensor &base, Tensor &scale,
                                     std::optional<uint32_t> flags, float globalScale, float globalShift, float epsilon,
                                     std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        output.pushBack(Image::Create(input[i].size(), input[i].format()));
    }

    return VarShapeNormalizeInto(output, input, base, scale, flags, globalScale, globalShift, epsilon, pstream);
}

} // namespace

void ExportOpNormalize(py::module &m)
{
    using namespace pybind11::literals;

    py::enum_<OpFlags>(m, "NormalizeFlags").value("SCALE_IS_STDDEV", OpFlags::SCALE_IS_STDDEV);

    float defGlobalScale = 1;
    float defGlobalShift = 0;
    float defEpsilon     = 0;

    m.def("normalize", &Normalize, "src"_a, "base"_a, "scale"_a, "flags"_a = std::nullopt, py::kw_only(),
          "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon,
          "stream"_a = nullptr);

    m.def("normalize_into", &NormalizeInto, "dst"_a, "src"_a, "base"_a, "scale"_a, "flags"_a = std::nullopt,
          py::kw_only(), "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon,
          "stream"_a = nullptr);

    m.def("normalize", &VarShapeNormalize, "src"_a, "base"_a, "scale"_a, "flags"_a = std::nullopt, py::kw_only(),
          "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon,
          "stream"_a = nullptr);

    m.def("normalize_into", &VarShapeNormalizeInto, "dst"_a, "src"_a, "base"_a, "scale"_a, "flags"_a = std::nullopt,
          py::kw_only(), "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon,
          "stream"_a = nullptr);
}

} // namespace cvcudapy
