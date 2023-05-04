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
#include <cvcuda/OpLaplacian.hpp>
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
Tensor LaplacianInto(Tensor &output, Tensor &input, const int &ksize, const float &scale, NVCVBorderType border,
                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto laplacian = CreateOperator<cvcuda::Laplacian>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*laplacian});

    laplacian->submit(pstream->cudaHandle(), input, output, ksize, scale, border);

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
    guard.add(LockMode::LOCK_READ, {input, ksize, scale});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*laplacian});

    laplacian->submit(pstream->cudaHandle(), input, output, ksize, scale, border);

    return output;
}

ImageBatchVarShape LaplacianVarShape(ImageBatchVarShape &input, Tensor &ksize, Tensor &scale, NVCVBorderType border,
                                     std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return LaplacianVarShapeInto(output, input, ksize, scale, border, pstream);
}

} // namespace

void ExportOpLaplacian(py::module &m)
{
    using namespace pybind11::literals;

    m.def("laplacian", &Laplacian, "src"_a, "ksize"_a, "scale"_a = 1.f,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Laplacian operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Laplacian operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            ksize (int): Aperture size used to compute the second-derivative filters, it can be 1 or 3.
            scale (float): Scale factor for the Laplacian values (use 1 for no scale).
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("laplacian_into", &LaplacianInto, "dst"_a, "src"_a, "ksize"_a, "scale"_a = 1.f,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Laplacian operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Laplacian operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            ksize (int): Aperture size used to compute the second-derivative filters, it can be 1 or 3.
            scale (float): Scale factor for the Laplacian values (use 1 for no scale).
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("laplacian", &LaplacianVarShape, "src"_a, "ksize"_a, "scale"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Laplacian operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Laplacian operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            ksize (Tensor): Aperture size used to compute the second-derivative filters, it can be 1 or 3 for each image.
            scale (Tensor): Scale factor for the Laplacian values (use 1 for no scale) for each image.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("laplacian_into", &LaplacianVarShapeInto, "dst"_a, "src"_a, "ksize"_a, "scale"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Laplacian operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Laplacian operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            dst (ImageBatchVarShape): Output image batch containing the result of the operation.
            ksize (Tensor): Aperture size used to compute the second-derivative filters, it can be 1 or 3 for each image.
            scale (Tensor): Scale factor for the Laplacian values (use 1 for no scale) for each image.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
