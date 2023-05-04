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
#include <cvcuda/OpChannelReorder.hpp>
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

ImageBatchVarShape ChannelReorderVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &orders,
                                              std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto chReorder = CreateOperator<cvcuda::ChannelReorder>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, orders});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*chReorder});

    chReorder->submit(pstream->cudaHandle(), input, output, orders);

    return output;
}

ImageBatchVarShape ChannelReorderVarShape(ImageBatchVarShape &input, Tensor &orders,
                                          std::optional<nvcv::ImageFormat> fmt, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = fmt ? *fmt : input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return ChannelReorderVarShapeInto(output, input, orders, pstream);
}

} // namespace

void ExportOpChannelReorder(py::module &m)
{
    using namespace pybind11::literals;

    m.def("channelreorder", &ChannelReorderVarShape, "src"_a, "order"_a, py::kw_only(), "format"_a = nullptr,
          "stream"_a = nullptr, R"pbdoc(

        Executes the Channel Reorder operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Channel Reorder operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input tensor containing one or more images.
            order(Tensor): 2D tensor with layout "NC" which specifies, for each output image sample in the batch,
                           the index of the input channel to copy to the output channel.
            format(ImageFormat): Format of the destination image.

            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("channelreorder_into", &ChannelReorderVarShapeInto, "dst"_a, "src"_a, "orders"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Channel Reorder operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Channel Reorder operator
            for more details and usage examples.

        Args:
            dst (ImageBatchVarShape): Output tensor to store the result of the operation.
            src (ImageBatchVarShape): Input tensor containing one or more images.
            order(Tensor): 2D tensor with layout "NC" which specifies, for each output image sample in the batch,
                           the index of the input channel to copy to the output channel.
            format(ImageFormat): Format of the destination image.

            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
