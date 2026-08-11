/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpHistogramEq.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <stdexcept>

namespace cvcudapy {

namespace {

class HistogramEqError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor HistogramEqInto(Tensor &output, Tensor &input, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }
    // HWC inputs (rank 3) have no N dim, so shape[0] is H — fall back to 1.
    uint32_t batchSize = (input.shape().size() == 4) ? (uint32_t)input.shape()[0] : 1u;
    auto     op        = CreateOperator<cvcuda::HistogramEq>(batchSize);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run([&op, &pstream, &input, &output]() { op->submit(pstream->cudaHandle(), input, output); });

    return std::move(output);
}

Tensor HistogramEq(Tensor &input, nvcv::DataType dtype, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), dtype);

    return HistogramEqInto(output, input, pstream);
}

ImageBatchVarShape HistogramEqVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                           std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::HistogramEq>((uint32_t)input.numImages());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run([&op, &pstream, &input, &output]() { op->submit(pstream->cudaHandle(), input, output); });

    return output;
}

ImageBatchVarShape HistogramEqVarShape(ImageBatchVarShape &input, std::optional<Stream> pstream)
{
    auto format = input.uniqueFormat();
    if (!format)
    {
        throw HistogramEqError("All images in input must have the same format.");
    }

    ImageBatchVarShape output = CreateSameShapeImageBatch(input, format, input.numImages());

    return HistogramEqVarShapeInto(output, input, pstream);
}

} // namespace

void ExportOpHistogramEq(py::module &m)
{
    using namespace pybind11::literals;

    m.def("histogrameq", NvtxTrace("cvcuda.histogrameq", &HistogramEq), "src"_a, "dtype"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the histogram equalization operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input image batch containing one or more images.
            dtype (numpy.dtype): The data type of the output.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output image batch.

    )pbdoc");

    m.def("histogrameq_into", NvtxTrace("cvcuda.histogrameq_into", &HistogramEqInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the histogram equalization operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output image batch containing the result of the operation.
            src (cvcuda.Tensor): Input image batch containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("histogrameq", NvtxTrace("cvcuda.histogrameq", &HistogramEqVarShape), "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the histogram equalization operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("histogrameq_into", NvtxTrace("cvcuda.histogrameq_into", &HistogramEqVarShapeInto), "dst"_a, "src"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the histogram equalization operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
