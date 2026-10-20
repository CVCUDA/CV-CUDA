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

#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpPadAndStack.hpp>

#include <stdexcept>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

class PadAndStackError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor &PadAndStackInto(Tensor &output, ImageBatchVarShape &input, Tensor &top, Tensor &left, NVCVBorderType border,
                        float borderValue, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto padstack = CreateOperator<cvcuda::PadAndStack>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, top, left});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*padstack});

    guard.run(
        [&padstack, &stream, &input, &output, &top, &left, &border, &borderValue]() {
            padstack->submit(stream.handle(), input.impl(), output.impl(), top.impl(), left.impl(), border,
                             borderValue);
        });

    return output;
}

std::shared_ptr<Tensor> PadAndStack(ImageBatchVarShape &input, Tensor &top, Tensor &left, NVCVBorderType border,
                                    float borderValue, std::optional<Stream *> pstream)
{
    nvcv::ImageFormat fmt = input.impl().uniqueFormat();
    if (fmt == nvcv::FMT_NONE)
    {
        throw PadAndStackError("All images in the input must have the same format");
    }

    auto output = native::CreateTensorForImageBatch(input.numImages(), input.impl().maxSize(), fmt);

    PadAndStackInto(*output, input, top, left, border, borderValue, pstream);
    return output;
}

} // namespace

void ExportOpPadAndStack(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("padandstack", NvtxTrace("cvcuda.padandstack", &PadAndStack), "src"_a, "top"_a, "left"_a,
          "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Pad and Stack operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): input image batch containing one or more images.
            top (cvcuda.Tensor): Top tensor to store amount of top padding per batch input image.
            left (cvcuda.Tensor): Left tensor to store amount of left padding per batch input image.
            border (cvcuda.Border): Border mode to be used when accessing elements outside input image.
            bvalue (float): Border value to be used for constant border mode.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("padandstack_into", NvtxTrace("cvcuda.padandstack_into", &PadAndStackInto), "dst"_a, "src"_a, "top"_a,
          "left"_a, "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the Pad and Stack operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): input image batch containing one or more images.
            top (cvcuda.Tensor): Top tensor to store amount of top padding per batch input image.
            left (cvcuda.Tensor): Left tensor to store amount of left padding per batch input image.
            border (cvcuda.Border): Border mode to be used when accessing elements outside input image.
            bvalue (float): Border value to be used for constant border mode.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
