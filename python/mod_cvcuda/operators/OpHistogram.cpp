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
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpHistogram.hpp>

#include <array>

namespace cvcudapy {

namespace {
using native::Stream;
using native::Tensor;

Tensor &HistogramInto(Tensor &histogram, Tensor &input, std::optional<Tensor *> mask, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    if (mask && *mask && (*mask)->impl().shape() != input.impl().shape())
    {
        throw std::invalid_argument("Mask must have the same shape as input");
    }

    auto op = CreateOperator<cvcuda::Histogram>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {histogram});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    if (mask && *mask)
    {
        guard.add(LockMode::LOCK_MODE_READ, {**mask});
        guard.run(
            [&op, &stream, &input, &mask, &histogram]() {
                op->submit(stream.handle(), input.impl(), nvcv::OptionalTensorConstRef{(*mask)->impl()},
                           histogram.impl());
            });
    }
    else
    {
        guard.run(
            [&op, &stream, &input, &histogram]() {
                op->submit(stream.handle(), input.impl(), nvcv::OptionalTensorConstRef{nvcv::NullOpt},
                           histogram.impl());
            });
    }
    return histogram;
}

std::shared_ptr<Tensor> Histogram(Tensor &input, std::optional<Tensor *> mask, std::optional<Stream *> pstream)
{
    std::array<ssize_t, 3> shape;
    // check for non batched tensors
    if (const nvcv::TensorShape &inputShape = input.impl().shape(); inputShape.rank() == 3)
    {
        shape[0] = 1;
        shape[1] = 256;
        shape[2] = 1;
    }
    else if (inputShape.rank() == 4)
    {
        shape[0] = inputShape[0];
        shape[1] = 256;
        shape[2] = 1;
    }
    else
    {
        throw std::invalid_argument("Input tensor must be HWC, NHWC, CHW, or NCHW");
    }

    auto histogram
        = native::CreateTensor(nvcv::TensorShape(shape.data(), shape.size(), nvcv::TENSOR_HWC), nvcv::TYPE_S32);
    HistogramInto(*histogram, input, mask, pstream);
    return histogram;
}

} // namespace

void ExportOpHistogram(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("histogram", NvtxTrace("cvcuda.histogram", &Histogram), "src"_a, "mask"_a = nullptr, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes an histogram operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images, input tensor must be (N)HWC or (N)CHW, currently only grayscale uint8 is supported.
            mask (cvcuda.Tensor, optional): Input tensor containing the mask of the pixels to be considered for the histogram, must be the same shape as src.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor containing the histogram. The tensor is formatted as HWC with W = 256 and H = number of input tensors, and C = 1.

    )pbdoc");

    m.def("histogram_into", NvtxTrace("cvcuda.histogram_into", &HistogramInto), "histogram"_a, "src"_a,
          "mask"_a = nullptr, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes an histogram operation on the given cuda stream.


        Args:
            histogram (cvcuda.Tensor): Output tensor containing the histogram. The tensor is formatted as HWC with W = 256 and H = number of input tensors, and C = 1.
            src (cvcuda.Tensor): Input tensor containing one or more images, input tensor must be (N)HWC or (N)CHW, currently only grayscale uint8 is supported.
            mask (cvcuda.Tensor, optional): Input tensor containing the bit mask of the pixels to be considered for the histogram, must be the same shape as src.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as histogram).

    )pbdoc");
}

} // namespace cvcudapy
