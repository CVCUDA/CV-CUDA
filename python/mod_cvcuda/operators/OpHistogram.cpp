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

#include <common/PyUtil.hpp>
#include <cvcuda/OpHistogram.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <array>

namespace cvcudapy {

namespace {

Tensor HistogramInto(Tensor &histogram, Tensor &input, std::optional<Tensor> mask, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (mask && mask->shape() != input.shape())
    {
        throw std::invalid_argument("Mask must have the same shape as input");
    }

    auto op = CreateOperator<cvcuda::Histogram>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {histogram});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    if (mask)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*mask});
        guard.run([&op, &pstream, &input, &mask, &histogram]()
                  { op->submit(pstream->cudaHandle(), input, nvcv::OptionalTensorConstRef{*mask}, histogram); });
    }
    else
    {
        guard.run(
            [&op, &pstream, &input, &histogram]()
            { op->submit(pstream->cudaHandle(), input, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, histogram); });
    }
    return std::move(histogram);
}

Tensor Histogram(Tensor &input, const std::optional<Tensor> &mask, std::optional<Stream> pstream)
{
    std::array<ssize_t, 3> shape;
    // check for non batched tensors
    if (input.shape().size() == 3)
    {
        shape[0] = 1;
        shape[1] = 256;
        shape[2] = 1;
    }
    else if (input.shape().size() == 4)
    {
        shape[0] = input.shape()[0];
        shape[1] = 256;
        shape[2] = 1;
    }
    else
    {
        throw std::invalid_argument("Input tensor must be HWC, NHWC, CHW, or NCHW");
    }

    Tensor histogram = Tensor::Create(nvcv::TensorShape(shape.data(), shape.size(), nvcv::TENSOR_HWC), nvcv::TYPE_S32);
    return HistogramInto(histogram, input, mask, pstream);
}

} // namespace

void ExportOpHistogram(py::module &m)
{
    using namespace pybind11::literals;

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
          "mask"_a = nullptr, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
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
