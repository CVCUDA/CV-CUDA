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
#include <cvcuda/OpChannelReorder.hpp>
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

Tensor ChannelReorderTensorInto(Tensor &output, Tensor &input, const std::vector<int32_t> &order,
                                std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto          channelReorder = CreateOperator<cvcuda::ChannelReorder>();
    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*channelReorder});
    guard.run(
        [&channelReorder, &pstream, &input, &output, &order] {
            channelReorder->submit(pstream->cudaHandle(), input, output, order.data(),
                                   static_cast<int32_t>(order.size()));
        });
    return output;
}

Tensor ChannelReorderTensor(Tensor &input, const std::vector<int32_t> &order, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());
    return ChannelReorderTensorInto(output, input, order, pstream);
}

ImageBatchVarShape ChannelReorderVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &orders,
                                              std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto chReorder = CreateOperator<cvcuda::ChannelReorder>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, orders});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*chReorder});

    guard.run([&chReorder, &pstream, &input, &output, &orders]()
              { chReorder->submit(pstream->cudaHandle(), input, output, orders); });

    return output;
}

ImageBatchVarShape ChannelReorderVarShape(ImageBatchVarShape &input, Tensor &orders,
                                          std::optional<nvcv::ImageFormat> fmt, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = fmt ? CreateSameShapeImageBatch(input, *fmt) : CreateSameShapeImageBatch(input);

    return ChannelReorderVarShapeInto(output, input, orders, pstream);
}

} // namespace

void ExportOpChannelReorder(py::module &m)
{
    using namespace pybind11::literals;

    m.def("channelreorder", NvtxTrace("cvcuda.channelreorder", &ChannelReorderTensor), "src"_a, "order"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Reorders the channels of a tensor using a host sequence.

        Each output channel ``c`` receives input channel ``order[c]``. Negative entries write zero;
        repeated non-negative entries are allowed. The output has the same shape, layout, and data
        type as the input.

        Args:
            src (cvcuda.Tensor): Input tensor in HWC, NHWC, CHW, or NCHW layout.
            order (Sequence[int]): One source-channel index per output channel.
            stream (cvcuda.Stream, optional): CUDA stream on which to submit the operation.

        Returns:
            cvcuda.Tensor: Reordered output tensor.
    )pbdoc");

    m.def("channelreorder_into", NvtxTrace("cvcuda.channelreorder_into", &ChannelReorderTensorInto), "dst"_a, "src"_a,
          "order"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Reorders tensor channels into a caller-provided output tensor.

        Args:
            dst (cvcuda.Tensor): Output tensor with metadata identical to ``src``.
            src (cvcuda.Tensor): Input tensor.
            order (Sequence[int]): One source-channel index per output channel; negatives write zero.
            stream (cvcuda.Stream, optional): CUDA stream on which to submit the operation.

        Returns:
            cvcuda.Tensor: ``dst``.
    )pbdoc");

    m.def("channelreorder", NvtxTrace("cvcuda.channelreorder", &ChannelReorderVarShape), "src"_a, "order"_a,
          py::kw_only(), "format"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the Channel Reorder operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input tensor containing one or more images.
            order (cvcuda.Tensor): 2D tensor with layout "NC" which specifies, for each output image sample in the batch,
                           the index of the input channel to copy to the output channel.
            format (cvcuda.Format): Format of the destination image.

            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("channelreorder_into", NvtxTrace("cvcuda.channelreorder_into", &ChannelReorderVarShapeInto), "dst"_a, "src"_a,
          "orders"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Channel Reorder operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output tensor to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input tensor containing one or more images.
            order (cvcuda.Tensor): 2D tensor with layout "NC" which specifies, for each output image sample in the batch,
                           the index of the input channel to copy to the output channel.
            format (cvcuda.Format): Format of the destination image.

            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
