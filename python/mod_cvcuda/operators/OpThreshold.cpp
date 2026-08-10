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
#include <cvcuda/OpThreshold.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class ThresholdError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor ThresholdInto(Tensor &output, Tensor &input, Tensor &thresh, Tensor &maxval, uint32_t type,
                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // HWC inputs (rank 3) have no N dim, so shape[0] is H — fall back to 1.
    int  batchSize = (input.shape().size() == 4) ? (int)input.shape()[0] : 1;
    auto threshold = CreateOperator<cvcuda::Threshold>(type, batchSize);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, thresh, maxval});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*threshold});

    guard.run([&thresh, &threshold, &pstream, &input, &output, &maxval]()
              { threshold->submit(pstream->cudaHandle(), input, output, thresh, maxval); });

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
    guard.add(LockMode::LOCK_MODE_READ, {input, thresh, maxval});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*threshold});

    guard.run([&thresh, &threshold, &pstream, &input, &output, &maxval]()
              { threshold->submit(pstream->cudaHandle(), input, output, thresh, maxval); });

    return output;
}

ImageBatchVarShape ThresholdVarShape(ImageBatchVarShape &input, Tensor &thresh, Tensor &maxval, uint32_t type,
                                     std::optional<Stream> pstream)
{
    auto format = input.uniqueFormat();
    if (!format)
    {
        throw ThresholdError("All images in input must have the same format.");
    }

    ImageBatchVarShape output = CreateSameShapeImageBatch(input, format, input.numImages());

    return ThresholdVarShapeInto(output, input, thresh, maxval, type, pstream);
}

} // namespace

void ExportOpThreshold(py::module &m)
{
    using namespace pybind11::literals;

    m.def("threshold", NvtxTrace("cvcuda.threshold", &Threshold), "src"_a, "thresh"_a, "maxval"_a, "type"_a,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Threshold operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            thresh (cvcuda.Tensor): An array of size batch that gives the threshold value of each image.
            maxval (cvcuda.Tensor): An array of size batch that gives the maxval value of each image,
                             using with the cvcuda.ThresholdType.BINARY or cvcuda.ThresholdType.BINARY_INV
                             threshold types.
            type (cvcuda.ThresholdType): Thresholding type.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("threshold_into", NvtxTrace("cvcuda.threshold_into", &ThresholdInto), "dst"_a, "src"_a, "thresh"_a,
          "maxval"_a, "type"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Threshold operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            thresh (cvcuda.Tensor): An array of size batch that gives the threshold value of each image.
            maxval (cvcuda.Tensor): An array of size batch that gives the maxval value of each image,
                             using with the cvcuda.ThresholdType.BINARY or cvcuda.ThresholdType.BINARY_INV
                             threshold types.
            type (cvcuda.ThresholdType): Thresholding type.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("threshold", NvtxTrace("cvcuda.threshold", &ThresholdVarShape), "src"_a, "thresh"_a, "maxval"_a, "type"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Threshold operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            thresh (cvcuda.Tensor): An array of size batch that gives the threshold value of each image.
            maxval (cvcuda.Tensor): An array of size batch that gives the maxval value of each image,
                             using with the cvcuda.ThresholdType.BINARY or cvcuda.ThresholdType.BINARY_INV
                             threshold types.
            type (cvcuda.ThresholdType): Thresholding type.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("threshold_into", NvtxTrace("cvcuda.threshold_into", &ThresholdVarShapeInto), "dst"_a, "src"_a, "thresh"_a,
          "maxval"_a, "type"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Threshold operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            thresh (cvcuda.Tensor): An array of size batch that gives the threshold value of each image.
            maxval (cvcuda.Tensor): An array of size batch that gives the maxval value of each image,
                             using with the cvcuda.ThresholdType.BINARY or cvcuda.ThresholdType.BINARY_INV
                             threshold types.
            type (cvcuda.ThresholdType): Thresholding type.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
