/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace cvcudapy {

namespace {

Tensor HistogramInto(Tensor &histogram, Tensor &input, std::optional<Tensor> mask, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (mask)
    {
        if (mask->shape() != input.shape())
        {
            throw std::invalid_argument("Mask must have the same shape as input");
        }
    }

    auto op = CreateOperator<cvcuda::Histogram>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {histogram});
    guard.add(LockMode::LOCK_NONE, {*op});

    if (mask)
    {
        guard.add(LockMode::LOCK_READ, {*mask});
        op->submit(pstream->cudaHandle(), input, *mask, histogram);
    }
    else
    {
        op->submit(pstream->cudaHandle(), input, nvcv::NullOpt, histogram);
    }
    return std::move(histogram);
}

Tensor Histogram(Tensor &input, std::optional<Tensor> mask, std::optional<Stream> pstream)
{
    ssize_t shape[3];
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
        throw std::invalid_argument("Input tensor must be HWC or NHWC");
    }

    Tensor histogram = Tensor::Create(nvcv::TensorShape(shape, 3, nvcv::TENSOR_HWC), nvcv::TYPE_S32);
    return HistogramInto(histogram, input, mask, pstream);
}

} // namespace

void ExportOpHistogram(py::module &m)
{
    using namespace pybind11::literals;

    m.def("histogram", &Histogram, "src"_a, "mask"_a = nullptr, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes an histogram operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Histogram operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images, input tensor must be (N)HWC, currently only grayscale uint8 is supported.
            mask (Tensor, optional): Input tensor containing the mask of the pixels to be considered for the histogram, must be the same shape as src.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor containing the histogram. The tensor is formatted as HWC with W = 256 and H = number of input tensors, and C = 1.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("histogram_into", &HistogramInto, "histogram"_a, "src"_a, "mask"_a = nullptr, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes an histogram operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Histogram operator
            for more details and usage examples.

        Args:
            histogram (Tensor): Output tensor containing the histogram. The tensor is formatted as HWC with W = 256 and H = number of input tensors, and C = 1.
            src (Tensor): Input tensor containing one or more images, input tensor must be (N)HWC, currently only grayscale uint8 is supported.
            mask (Tensor, optional): Input tensor containing the bit mask of the pixels to be considered for the histogram, must be the same shape as src.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
