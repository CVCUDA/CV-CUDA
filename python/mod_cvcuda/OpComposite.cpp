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
#include <cvcuda/OpComposite.hpp>
#include <cvcuda/Types.h>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor CompositeInto(Tensor &output, Tensor &foreground, Tensor &background, Tensor &fgMask,
                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto composite = CreateOperator<cvcuda::Composite>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {foreground, background, fgMask});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*composite});

    composite->submit(pstream->cudaHandle(), foreground, background, fgMask, output);

    return output;
}

Tensor Composite(Tensor &foreground, Tensor &background, Tensor &fgMask, int outChannels, std::optional<Stream> pstream)
{
    nvcv::TensorShape fg_shape = foreground.shape();
    Shape             out_shape(&fg_shape[0], &fg_shape[0] + fg_shape.rank());
    int               cdim = out_shape.size();
    out_shape[cdim - 1]    = outChannels;

    Tensor output = Tensor::Create(nvcv::TensorShape(out_shape.data(), out_shape.size(), foreground.layout()),
                                   foreground.dtype());

    return CompositeInto(output, foreground, background, fgMask, pstream);
}

ImageBatchVarShape CompositeVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &foreground,
                                         ImageBatchVarShape &background, ImageBatchVarShape &fgMask,
                                         std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto composite = CreateOperator<cvcuda::Composite>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {foreground, background, fgMask});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*composite});

    composite->submit(pstream->cudaHandle(), foreground, background, fgMask, output);

    return output;
}

ImageBatchVarShape CompositeVarShape(ImageBatchVarShape &foreground, ImageBatchVarShape &background,
                                     ImageBatchVarShape &fgMask, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(foreground.numImages());

    nvcv::ImageFormat format = foreground.uniqueFormat();

    for (auto img = foreground.begin(); img != foreground.end(); ++img)
    {
        auto newimg = Image::Create(img->size(), format);
        output.pushBack(newimg);
    }

    return CompositeVarShapeInto(output, foreground, background, fgMask, pstream);
}

} // namespace

void ExportOpComposite(py::module &m)
{
    using namespace pybind11::literals;

    m.def("composite", &Composite, "foreground"_a, "background"_a, "fgmask"_a, "outchannels"_a, py::kw_only(),
          "stream"_a = nullptr);
    m.def("composite_into", &CompositeInto, "dst"_a, "foreground"_a, "background"_a, "fgmask"_a, py::kw_only(),
          "stream"_a = nullptr);
    m.def("composite", &CompositeVarShape, "foreground"_a, "background"_a, "fgmask"_a, py::kw_only(),
          "stream"_a = nullptr);
    m.def("composite_into", &CompositeVarShapeInto, "dst"_a, "foreground"_a, "background"_a, "fgmask"_a, py::kw_only(),
          "stream"_a = nullptr);
}

} // namespace cvcudapy
