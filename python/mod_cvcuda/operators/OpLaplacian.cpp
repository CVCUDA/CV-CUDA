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
#include <cvcuda/OpLaplacian.hpp>
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
Tensor LaplacianInto(Tensor &output, Tensor &input, const int &ksize, const float &scale, NVCVBorderType border,
                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto laplacian = CreateOperator<cvcuda::Laplacian>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*laplacian});

    guard.run([&laplacian, &pstream, &input, &output, &ksize, &scale, &border]()
              { laplacian->submit(pstream->cudaHandle(), input, output, ksize, scale, border); });

    return output;
}

Tensor Laplacian(Tensor &input, const int &ksize, const float &scale, NVCVBorderType border,
                 std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return LaplacianInto(output, input, ksize, scale, border, pstream);
}

ImageBatchVarShape LaplacianVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &ksize,
                                         Tensor &scale, NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto laplacian = CreateOperator<cvcuda::Laplacian>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, ksize, scale});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*laplacian});

    guard.run([&laplacian, &pstream, &input, &output, &ksize, &scale, &border]()
              { laplacian->submit(pstream->cudaHandle(), input, output, ksize, scale, border); });

    return output;
}

ImageBatchVarShape LaplacianVarShape(ImageBatchVarShape &input, Tensor &ksize, Tensor &scale, NVCVBorderType border,
                                     std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input, input.numImages());

    return LaplacianVarShapeInto(output, input, ksize, scale, border, pstream);
}

} // namespace

void ExportOpLaplacian(py::module &m)
{
    using namespace pybind11::literals;

    m.def("laplacian", NvtxTrace("cvcuda.laplacian", &Laplacian), "src"_a, "ksize"_a, "scale"_a = 1.f,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Laplacian operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            ksize (int): Aperture size used to compute the second-derivative filters, it can be 1 or 3.
            scale (float): Scale factor for the Laplacian values (use 1 for no scale).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("laplacian_into", NvtxTrace("cvcuda.laplacian_into", &LaplacianInto), "dst"_a, "src"_a, "ksize"_a,
          "scale"_a = 1.f, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Laplacian operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            ksize (int): Aperture size used to compute the second-derivative filters, it can be 1 or 3.
            scale (float): Scale factor for the Laplacian values (use 1 for no scale).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("laplacian", NvtxTrace("cvcuda.laplacian", &LaplacianVarShape), "src"_a, "ksize"_a, "scale"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Laplacian operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            ksize (cvcuda.Tensor): Aperture size used to compute the second-derivative filters, it can be 1 or 3 for each image.
            scale (cvcuda.Tensor): Scale factor for the Laplacian values (use 1 for no scale) for each image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("laplacian_into", NvtxTrace("cvcuda.laplacian_into", &LaplacianVarShapeInto), "dst"_a, "src"_a, "ksize"_a,
          "scale"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Laplacian operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            ksize (cvcuda.Tensor): Aperture size used to compute the second-derivative filters, it can be 1 or 3 for each image.
            scale (cvcuda.Tensor): Scale factor for the Laplacian values (use 1 for no scale) for each image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
