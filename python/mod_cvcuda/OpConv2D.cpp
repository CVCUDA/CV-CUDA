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
#include <common/String.hpp>
#include <cvcuda/OpConv2D.hpp>
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

ImageBatchVarShape Conv2DVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, ImageBatchVarShape &kernel,
                                      Tensor &kernel_anchor, NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto conv2D = CreateOperator<cvcuda::Conv2D>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, kernel, kernel_anchor});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*conv2D});

    conv2D->submit(pstream->cudaHandle(), input, output, kernel, kernel_anchor, border);

    return output;
}

ImageBatchVarShape Conv2DVarShape(ImageBatchVarShape &input, ImageBatchVarShape &kernel, Tensor &kernel_anchor,
                                  NVCVBorderType border, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return Conv2DVarShapeInto(output, input, kernel, kernel_anchor, border, pstream);
}

} // namespace

void ExportOpConv2D(py::module &m)
{
    using namespace pybind11::literals;

    m.def("conv2d", &Conv2DVarShape, "src"_a, "kernel"_a, "kernel_anchor"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Convolve 2D operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Convolve 2D operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            kernel(Tensor): Convolution kernels (one for each batch image) to be used. Each image width and height
                            correspond to the kernel width and height. (must be float)
            kernel_anchor(Tensor): 1D Tensor with the anchor of each kernel (one for each batch image).  The anchor (x, y)
                            indicates the relative position of a filtered point within the kernel.
                            (-1, -1) means that the anchor is at the kernel center.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("conv2d_into", &Conv2DVarShapeInto, "dst"_a, "src"_a, "kernel"_a, "kernel_anchor"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Convolve 2D operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Convolve 2D operator
            for more details and usage examples.

        Args:
            dst (ImageBatchVarShape): Output image batch to store the result of the operation.
            src (ImageBatchVarShape): Input image batch containing one or more images.
            kernel(Tensor): Convolution kernels (one for each batch image) to be used. Each image width and height
                            correspond to the kernel width and height. (must be float)
            kernel_anchor(Tensor): 1D Tensor with the anchor of each kernel (one for each batch image).  The anchor (x, y)
                            indicates the relative position of a filtered point within the kernel.
                            (-1, -1) means that the anchor is at the kernel center.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
