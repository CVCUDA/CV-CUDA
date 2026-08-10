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

#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpAverageBlur.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
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
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_WRITE, {*averageBlur});

    guard.run([&averageBlur, &pstream, &input, &output, &kernelSizeArg, &kernelAnchorArg, &border]()
              { averageBlur->submit(pstream->cudaHandle(), input, output, kernelSizeArg, kernelAnchorArg, border); });

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
    guard.add(LockMode::LOCK_MODE_READ, {input, kernel_size, kernel_anchor});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*averageBlur});

    guard.run([&averageBlur, &pstream, &input, &output, &kernel_size, &kernel_anchor, &border]()
              { averageBlur->submit(pstream->cudaHandle(), input, output, kernel_size, kernel_anchor, border); });

    return output;
}

ImageBatchVarShape AverageBlurVarShape(ImageBatchVarShape &input, const std::tuple<int, int> &max_kernel_size,
                                       Tensor &kernel_size, Tensor &kernel_anchor, NVCVBorderType border,
                                       std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return AverageBlurVarShapeInto(output, input, max_kernel_size, kernel_size, kernel_anchor, border, pstream);
}

} // namespace

void ExportOpAverageBlur(py::module &m)
{
    using namespace pybind11::literals;

    const std::tuple<int, int> def_anchor{-1, -1};

    m.def("averageblur", NvtxTrace("cvcuda.averageblur", &AverageBlur), "src"_a, "kernel_size"_a,
          "kernel_anchor"_a = def_anchor, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the AverageBlur operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            kernel_size (Tuple[int, int]): Specifies the size of the blur kernel.
            kernel_anchor (Tuple[int, int]): Kernel anchor, use (-1,-1) to indicate kernel center.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.


        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("averageblur_into", NvtxTrace("cvcuda.averageblur_into", &AverageBlurInto), "dst"_a, "src"_a, "kernel_size"_a,
          "kernel_anchor"_a = def_anchor, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the AverageBlur operation on the given cuda stream and writes the result into the 'dst' tensor.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            kernel_size (Tuple[int, int]): Specifies the size of the blur kernel.
            kernel_anchor (Tuple[int, int]): Kernel anchor, use (-1,-1) to indicate kernel center.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("averageblur", NvtxTrace("cvcuda.averageblur", &AverageBlurVarShape), "src"_a, "max_kernel_size"_a,
          "kernel_size"_a, "kernel_anchor"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the AverageBlur operation with a variable shape tensors on the given cuda stream.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            max_kernel_size (Tuple[int, int]): Specifies the maximum size of the blur kernel.
            kernel_size (Tuple[int, int]): Specifies the size of the blur kernel within the maximum kernel size.
            kernel_anchor (Tuple[int, int]): Kernel anchor, use (-1,-1) to indicate kernel center.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("averageblur_into", NvtxTrace("cvcuda.averageblur_into", &AverageBlurVarShapeInto), "dst"_a, "src"_a,
          "max_kernel_size"_a, "kernel_size"_a, "kernel_anchor"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the AverageBlur operation with a variable shape tensors on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output containing one or more images.
            src (cvcuda.ImageBatchVarShape): Input containing one or more images.
            max_kernel_size (Tuple[int, int]): Specifies the maximum size of the blur kernel.
            kernel_size (Tuple[int, int]): Specifies the size of the blur kernel within the maximum kernel size.
            kernel_anchor (Tuple[int, int]): Kernel anchor, use (-1,-1) to indicate kernel center.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
