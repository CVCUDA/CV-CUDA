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
#include <cvcuda/OpAverageBlur.hpp>
#include <cvcuda/Types.h>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor AverageBlurInto(Tensor &output, Tensor &input, const std::tuple<int, int> &kernel_size,
                       const std::tuple<int, int> &kernel_anchor, NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    nvcv::Size2D kernelSizeArg{std::get<0>(kernel_size), std::get<1>(kernel_size)};
    int2         kernelAnchorArg{std::get<0>(kernel_anchor), std::get<1>(kernel_anchor)};

    auto averageBlur = CreateOperator<cvcuda::AverageBlur>(kernelSizeArg, 0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*averageBlur});

    averageBlur->submit(pstream->cudaHandle(), input, output, kernelSizeArg, kernelAnchorArg, border);

    return output;
}

Tensor AverageBlur(Tensor &input, const std::tuple<int, int> &kernel_size, const std::tuple<int, int> &kernel_anchor,
                   NVCVBorderType border, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return AverageBlurInto(output, input, kernel_size, kernel_anchor, border, pstream);
}

ImageBatchVarShape AverageBlurVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                           const std::tuple<int, int> &max_kernel_size, Tensor &kernel_size,
                                           Tensor &kernel_anchor, NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    nvcv::Size2D maxKernelSizeArg{std::get<0>(max_kernel_size), std::get<1>(max_kernel_size)};

    auto averageBlur = CreateOperator<cvcuda::AverageBlur>(maxKernelSizeArg, input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, kernel_size, kernel_anchor});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*averageBlur});

    averageBlur->submit(pstream->cudaHandle(), input, output, kernel_size, kernel_anchor, border);

    return output;
}

ImageBatchVarShape AverageBlurVarShape(ImageBatchVarShape &input, const std::tuple<int, int> &max_kernel_size,
                                       Tensor &kernel_size, Tensor &kernel_anchor, NVCVBorderType border,
                                       std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return AverageBlurVarShapeInto(output, input, max_kernel_size, kernel_size, kernel_anchor, border, pstream);
}

} // namespace

void ExportOpAverageBlur(py::module &m)
{
    using namespace pybind11::literals;

    const std::tuple<int, int> def_anchor{-1, -1};

    m.def("averageblur", &AverageBlur, "src"_a, "kernel_size"_a, "kernel_anchor"_a = def_anchor,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the AverageBlur operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the AverageBlur operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            kernel_size (Tuple [int,int]): Specifies the size of the blur kernel.
            kernel_anchor (Tuple [int,int]): Kernel anchor, use (-1,-1) to indicate kernel center.
            border (NVCVBorderType, optional): Border mode to be used when accessing elements outside input image.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        See also:
            Refer to the CV-CUDA C API reference for the AverageBlur operator
            for more details and usage examples.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("averageblur_into", &AverageBlurInto, "dst"_a, "src"_a, "kernel_size"_a, "kernel_anchor"_a = def_anchor,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the AverageBlur operation on the given cuda stream and writes the result into the 'dst' tensor.

        See also:
            Refer to the CV-CUDA C API reference for the AverageBlur operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            kernel_size (Tuple [int,int]): Specifies the size of the blur kernel.
            kernel_anchor (Tuple [int,int]): Kernel anchor, use (-1,-1) to indicate kernel center.
            border (NVCVBorderType, optional): Border mode to be used when accessing elements outside input image.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("averageblur", &AverageBlurVarShape, "src"_a, "max_kernel_size"_a, "kernel_size"_a, "kernel_anchor"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the AverageBlur operation with a variable shape tensors on the given cuda stream.

        Args:
            src (Tensor): Input tensor containing one or more images.
            max_kernel_size (Tuple [int,int]): Specifies the maximum size of the blur kernel.
            kernel_size (Tuple [int,int]): Specifies the size of the blur kernel within the maximum kernel size.
            kernel_anchor (Tuple [int,int]): Kernel anchor, use (-1,-1) to indicate kernel center.
            border (NVCVBorderType, optional): Border mode to be used when accessing elements outside input image.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("averageblur_into", &AverageBlurVarShapeInto, "dst"_a, "src"_a, "max_kernel_size"_a, "kernel_size"_a,
          "kernel_anchor"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the AverageBlur operation with a variable shape tensors on the given cuda stream.

        Args:
            dst (ImageBatchVarShape): Output containing one or more images.
            src (ImageBatchVarShape): Input containing one or more images.
            max_kernel_size (Tuple [int,int]): Specifies the maximum size of the blur kernel.
            kernel_size (Tuple [int,int]): Specifies the size of the blur kernel within the maximum kernel size.
            kernel_anchor (Tuple [int,int]): Kernel anchor, use (-1,-1) to indicate kernel center.
            border (NVCVBorderType, optional): Border mode to be used when accessing elements outside input image.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
