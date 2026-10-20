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

#include "NativeBinding.hpp"
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpHistogramEq.hpp>
#include <cvcuda/Types.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

class HistogramEqError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor &HistogramEqInto(Tensor &output, Tensor &input, NVCVHistogramEqMethod method, std::optional<Stream *> pstream)
{
    Stream                  &stream = native::GetStream(pstream);
    // HWC inputs (rank 3) have no N dim, so shape[0] is H — fall back to 1.
    const nvcv::TensorShape &inputShape = input.impl().shape();
    uint32_t                 batchSize  = inputShape.rank() == 4 ? static_cast<uint32_t>(inputShape[0]) : 1u;
    auto                     op         = CreateOperator<cvcuda::HistogramEq>(batchSize);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run([&op, &stream, &input, &output, method]()
              { op->submit(stream.handle(), input.impl(), output.impl(), method); });

    return output;
}

std::shared_ptr<Tensor> HistogramEq(Tensor &input, nvcv::DataType dtype, NVCVHistogramEqMethod method,
                                    std::optional<Stream *> pstream)
{
    auto output = native::CreateTensor(input.impl().shape(), dtype);

    HistogramEqInto(*output, input, method, pstream);
    return output;
}

ImageBatchVarShape &HistogramEqVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                            NVCVHistogramEqMethod method, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::HistogramEq>((uint32_t)input.numImages());

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run([&op, &stream, &input, &output, method]()
              { op->submit(stream.handle(), input.impl(), output.impl(), method); });

    return output;
}

std::shared_ptr<ImageBatchVarShape> HistogramEqVarShape(ImageBatchVarShape &input, NVCVHistogramEqMethod method,
                                                        std::optional<Stream *> pstream)
{
    auto format = GetUniqueFormat(input);
    if (!format)
    {
        throw HistogramEqError("All images in input must have the same format.");
    }

    auto output = CreateSameShapeImageBatch(input, *format, input.numImages());

    HistogramEqVarShapeInto(*output, input, method, pstream);
    return output;
}

} // namespace

void ExportOpHistogramEq(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("histogrameq", NvtxTrace("cvcuda.histogrameq", &HistogramEq), "src"_a, "dtype"_a, py::kw_only(),
          "method"_a = NVCV_HISTOGRAM_EQ_OPENCV, "stream"_a = nullptr, R"pbdoc(
        Executes the histogram equalization operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input image batch containing one or more images.
            dtype (numpy.dtype): The data type of the output.
            method (cvcuda.HistogramEq, optional): Lookup-table construction. Defaults to
                cvcuda.HistogramEq.OPENCV; use cvcuda.HistogramEq.PILLOW to match Pillow and
                torchvision.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output image batch.

    )pbdoc");

    m.def("histogrameq_into", NvtxTrace("cvcuda.histogrameq_into", &HistogramEqInto), "dst"_a, "src"_a, py::kw_only(),
          "method"_a = NVCV_HISTOGRAM_EQ_OPENCV, "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the histogram equalization operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output image batch containing the result of the operation.
            src (cvcuda.Tensor): Input image batch containing one or more images.
            method (cvcuda.HistogramEq, optional): Lookup-table construction. Defaults to
                cvcuda.HistogramEq.OPENCV; use cvcuda.HistogramEq.PILLOW to match Pillow and
                torchvision.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("histogrameq", NvtxTrace("cvcuda.histogrameq", &HistogramEqVarShape), "src"_a, py::kw_only(),
          "method"_a = NVCV_HISTOGRAM_EQ_OPENCV, "stream"_a = nullptr, R"pbdoc(
        Executes the histogram equalization operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            method (cvcuda.HistogramEq, optional): Lookup-table construction. Defaults to
                cvcuda.HistogramEq.OPENCV; use cvcuda.HistogramEq.PILLOW to match Pillow and
                torchvision.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("histogrameq_into", NvtxTrace("cvcuda.histogrameq_into", &HistogramEqVarShapeInto), "dst"_a, "src"_a,
          py::kw_only(), "method"_a = NVCV_HISTOGRAM_EQ_OPENCV, "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the histogram equalization operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            method (cvcuda.HistogramEq, optional): Lookup-table construction. Defaults to
                cvcuda.HistogramEq.OPENCV; use cvcuda.HistogramEq.PILLOW to match Pillow and
                torchvision.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
