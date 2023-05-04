/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpPadAndStack.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor PadAndStackInto(Tensor &output, ImageBatchVarShape &input, Tensor &top, Tensor &left, NVCVBorderType border,
                       float borderValue, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto padstack = CreateOperator<cvcuda::PadAndStack>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, top, left});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*padstack});

    padstack->submit(pstream->cudaHandle(), input, output, top, left, border, borderValue);

    return std::move(output);
}

Tensor PadAndStack(ImageBatchVarShape &input, Tensor &top, Tensor &left, NVCVBorderType border, float borderValue,
                   std::optional<Stream> pstream)
{
    nvcv::ImageFormat fmt = input.uniqueFormat();
    if (fmt == nvcv::FMT_NONE)
    {
        throw std::runtime_error("All images in the input must have the same format");
    }

    Tensor output = Tensor::CreateForImageBatch(input.numImages(), input.maxSize(), fmt);

    return PadAndStackInto(output, input, top, left, border, borderValue, pstream);
}

} // namespace

void ExportOpPadAndStack(py::module &m)
{
    using namespace pybind11::literals;

    m.def("padandstack", &PadAndStack, "src"_a, "top"_a, "left"_a, "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Pad and Stack operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pad and Stack operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): input image batch containing one or more images.
            top (Tensor): Top tensor to store amount of top padding per batch input image.
            left (Tensor): Left tensor to store amount of left padding per batch input image.
            border (Border): Border mode to be used when accessing elements outside input image.
            bvalue (float): Border value to be used for constant border mode.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("padandstack_into", &PadAndStackInto, "dst"_a, "src"_a, "top"_a, "left"_a, "border"_a = NVCV_BORDER_CONSTANT,
          "bvalue"_a = 0, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Pad and Stack operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Pad and Stack operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (ImageBatchVarShape): input image batch containing one or more images.
            top (Tensor): Top tensor to store amount of top padding per batch input image.
            left (Tensor): Left tensor to store amount of left padding per batch input image.
            border (Border): Border mode to be used when accessing elements outside input image.
            bvalue (float): Border value to be used for constant border mode.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
