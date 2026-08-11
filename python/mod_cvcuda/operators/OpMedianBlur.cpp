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
#include <cvcuda/OpMedianBlur.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor MedianBlurInto(Tensor &output, Tensor &input, const std::tuple<int, int> &ksize, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto median_blur = CreateOperator<cvcuda::MedianBlur>(0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*median_blur});

    nvcv::Size2D ksizeArg{std::get<0>(ksize), std::get<1>(ksize)};

    guard.run([&median_blur, &pstream, &input, &output, &ksizeArg]()
              { median_blur->submit(pstream->cudaHandle(), input, output, ksizeArg); });

    return output;
}

Tensor MedianBlur(Tensor &input, const std::tuple<int, int> &ksize, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return MedianBlurInto(output, input, ksize, pstream);
}

ImageBatchVarShape VarShapeMedianBlurInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &ksize,
                                          std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto median_blur = CreateOperator<cvcuda::MedianBlur>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, ksize});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*median_blur});

    guard.run([&median_blur, &pstream, &input, &output, &ksize]()
              { median_blur->submit(pstream->cudaHandle(), input, output, ksize); });

    return output;
}

ImageBatchVarShape VarShapeMedianBlur(ImageBatchVarShape &input, Tensor &ksize, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return VarShapeMedianBlurInto(output, input, ksize, pstream);
}

} // namespace

void ExportOpMedianBlur(py::module &m)
{
    using namespace pybind11::literals;

    m.def("median_blur", NvtxTrace("cvcuda.median_blur", &MedianBlur), "src"_a, "ksize"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Median Blur operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            ksize (Tuple[int, int]): Width and Height of the kernel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("median_blur_into", NvtxTrace("cvcuda.median_blur_into", &MedianBlurInto), "dst"_a, "src"_a, "ksize"_a,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Median Blur operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            ksize (Tuple[int, int]): Width and Height of the kernel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("median_blur", NvtxTrace("cvcuda.median_blur", &VarShapeMedianBlur), "src"_a, "ksize"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Median Blur operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            ksize (cvcuda.Tensor): Width and Height of the kernel for each image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("median_blur_into", NvtxTrace("cvcuda.median_blur_into", &VarShapeMedianBlurInto), "dst"_a, "src"_a,
          "ksize"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Median Blur operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            ksize (cvcuda.Tensor): Width and Height of the kernel for each image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
