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
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpConv2D.hpp>
#include <cvcuda/Types.h>

#include <memory>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

ImageBatchVarShape &Conv2DVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                       ImageBatchVarShape &kernel, Tensor &kernel_anchor, NVCVBorderType border,
                                       std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto conv2D = CreateOperator<cvcuda::Conv2D>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, kernel, kernel_anchor});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*conv2D});

    guard.run(
        [&conv2D, &stream, &input, &output, &kernel, &kernel_anchor, &border]()
        { conv2D->submit(stream.handle(), input.impl(), output.impl(), kernel.impl(), kernel_anchor.impl(), border); });

    return output;
}

std::shared_ptr<ImageBatchVarShape> Conv2DVarShape(ImageBatchVarShape &input, ImageBatchVarShape &kernel,
                                                   Tensor &kernel_anchor, NVCVBorderType border,
                                                   std::optional<Stream *> pstream)
{
    auto output = CreateSameShapeImageBatch(input);

    Conv2DVarShapeInto(*output, input, kernel, kernel_anchor, border, pstream);
    return output;
}

} // namespace

void ExportOpConv2D(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("conv2d", NvtxTrace("cvcuda.conv2d", &Conv2DVarShape), "src"_a, "kernel"_a, "kernel_anchor"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Convolve 2D operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            kernel (cvcuda.Tensor): Convolution kernels (one for each batch image) to be used. Each image width and height
                            correspond to the kernel width and height. (must be float)
            kernel_anchor (cvcuda.Tensor): 1D Tensor with the anchor of each kernel (one for each batch image).  The anchor (x, y)
                            indicates the relative position of a filtered point within the kernel.
                            (-1, -1) means that the anchor is at the kernel center.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("conv2d_into", NvtxTrace("cvcuda.conv2d_into", &Conv2DVarShapeInto), "dst"_a, "src"_a, "kernel"_a,
          "kernel_anchor"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the Convolve 2D operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            kernel (cvcuda.Tensor): Convolution kernels (one for each batch image) to be used. Each image width and height
                            correspond to the kernel width and height. (must be float)
            kernel_anchor (cvcuda.Tensor): 1D Tensor with the anchor of each kernel (one for each batch image).  The anchor (x, y)
                            indicates the relative position of a filtered point within the kernel.
                            (-1, -1) means that the anchor is at the kernel center.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
