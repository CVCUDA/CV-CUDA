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
#include <cvcuda/OpThreshold.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor ThresholdInto(Tensor &output, Tensor &input, Tensor &thresh, Tensor &maxval, uint32_t type,
                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    nvcv::TensorShape shape     = input.shape();
    auto              threshold = CreateOperator<cvcuda::Threshold>(type, (int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, thresh, maxval});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*threshold});

    threshold->submit(pstream->cudaHandle(), input, output, thresh, maxval);

    return output;
}

Tensor Threshold(Tensor &input, Tensor &thresh, Tensor &maxval, uint32_t type, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return ThresholdInto(output, input, thresh, maxval, type, pstream);
}

ImageBatchVarShape ThresholdVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &thresh,
                                         Tensor &maxval, uint32_t type, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto threshold = CreateOperator<cvcuda::Threshold>(type, input.numImages());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, thresh, maxval});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*threshold});

    threshold->submit(pstream->cudaHandle(), input, output, thresh, maxval);

    return output;
}

ImageBatchVarShape ThresholdVarShape(ImageBatchVarShape &input, Tensor &thresh, Tensor &maxval, uint32_t type,
                                     std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.uniqueFormat();
    if (!format)
    {
        throw std::runtime_error("All images in input must have the same format.");
    }

    for (auto img = input.begin(); img != input.end(); ++img)
    {
        auto newimg = Image::Create(img->size(), format);
        output.pushBack(newimg);
    }

    return ThresholdVarShapeInto(output, input, thresh, maxval, type, pstream);
}

} // namespace

void ExportOpThreshold(py::module &m)
{
    using namespace pybind11::literals;

    m.def("threshold", &Threshold, "src"_a, "thresh"_a, "maxval"_a, "type"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the Threshold operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Threshold operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            thresh (Tensor): An array of size batch that gives the threshold value of each image.
            maxval (Tensor): An array of size batch that gives the maxval value of each image,
                             using with the NVCV_THRESH_BINARY and NVCV_THRESH_BINARY_INV threshold types.
            type (ThresholdType): Thresholding type.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("threshold_into", &ThresholdInto, "dst"_a, "src"_a, "thresh"_a, "maxval"_a, "type"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Threshold operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Threshold operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            thresh (Tensor): An array of size batch that gives the threshold value of each image.
            maxval (Tensor): An array of size batch that gives the maxval value of each image,
                             using with the NVCV_THRESH_BINARY and NVCV_THRESH_BINARY_INV threshold types.
            type (ThresholdType): Thresholding type.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("threshold", &ThresholdVarShape, "src"_a, "thresh"_a, "maxval"_a, "type"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Threshold operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Threshold operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            thresh (Tensor): An array of size batch that gives the threshold value of each image.
            maxval (Tensor): An array of size batch that gives the maxval value of each image,
                             using with the NVCV_THRESH_BINARY and NVCV_THRESH_BINARY_INV threshold types.
            type (ThresholdType): Thresholding type.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("threshold_into", &ThresholdVarShapeInto, "dst"_a, "src"_a, "thresh"_a, "maxval"_a, "type"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Threshold operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Threshold operator
            for more details and usage examples.

        Args:
            dst (ImageBatchVarShape): Output image batch containing the result of the operation.
            src (ImageBatchVarShape): Input image batch containing one or more images.
            thresh (Tensor): An array of size batch that gives the threshold value of each image.
            maxval (Tensor): An array of size batch that gives the maxval value of each image,
                             using with the NVCV_THRESH_BINARY and NVCV_THRESH_BINARY_INV threshold types.
            type (ThresholdType): Thresholding type.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
