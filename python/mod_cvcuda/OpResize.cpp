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
#include <cvcuda/OpResize.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor ResizeInto(Tensor &output, Tensor &input, NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto resize = CreateOperator<cvcuda::Resize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*resize});

    resize->submit(pstream->cudaHandle(), input, output, interp);

    return std::move(output);
}

Tensor Resize(Tensor &input, const Shape &out_shape, NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    Tensor output
        = Tensor::Create(nvcv::TensorShape(out_shape.data(), out_shape.size(), input.shape().layout()), input.dtype());

    return ResizeInto(output, input, interp, pstream);
}

ImageBatchVarShape ResizeVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                      NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto resize = CreateOperator<cvcuda::Resize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*resize});

    resize->submit(pstream->cudaHandle(), input, output, interp);

    return output;
}

ImageBatchVarShape ResizeVarShape(ImageBatchVarShape &input, const std::vector<std::tuple<int, int>> &out_size,
                                  NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (input.numImages() != (int)out_size.size())
    {
        throw std::runtime_error("Number of input images must be equal to the number of elements in output size list ");
    }

    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        auto              size   = out_size[i];
        auto              image  = Image::Create({std::get<0>(size), std::get<1>(size)}, format);
        output.pushBack(image);
    }

    return ResizeVarShapeInto(output, input, interp, pstream);
}

} // namespace

void ExportOpResize(py::module &m)
{
    using namespace pybind11::literals;

    m.def("resize", &Resize, "src"_a, "shape"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr);
    m.def("resize_into", &ResizeInto, "dst"_a, "src"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
          "stream"_a = nullptr);

    m.def("resize", &ResizeVarShape, "src"_a, "sizes"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
          "stream"_a = nullptr);
    m.def("resize_into", &ResizeVarShapeInto, "dst"_a, "src"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
          "stream"_a = nullptr);
}

} // namespace cvcudapy
