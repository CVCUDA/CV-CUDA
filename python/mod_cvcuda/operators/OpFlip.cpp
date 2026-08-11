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
#include <cvcuda/OpFlip.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor FlipInto(Tensor &output, Tensor &input, int32_t flipCode, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto Flip = CreateOperator<cvcuda::Flip>(0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*Flip});

    guard.run([&Flip, &pstream, &input, &output, &flipCode]()
              { Flip->submit(pstream->cudaHandle(), input, output, flipCode); });

    return output;
}

Tensor Flip(Tensor &input, int32_t flipCode, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return FlipInto(output, input, flipCode, pstream);
}

ImageBatchVarShape FlipVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &flipCode,
                                    std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto flip = CreateOperator<cvcuda::Flip>(0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, flipCode});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*flip});

    guard.run([&flip, &pstream, &input, &output, &flipCode]()
              { flip->submit(pstream->cudaHandle(), input, output, flipCode); });

    return output;
}

ImageBatchVarShape FlipVarShape(ImageBatchVarShape &input, Tensor &flipCode, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return FlipVarShapeInto(output, input, flipCode, pstream);
}

} // namespace

void ExportOpFlip(py::module &m)
{
    using namespace pybind11::literals;

    m.def("flip", NvtxTrace("cvcuda.flip", &Flip), "src"_a, "flipCode"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Flip operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            flipCode (int): Flag to specify how to flip the array; 0 means flipping
                            around the x-axis and positive value (for example, 1) means flipping
                            around y-axis. Negative value (for example, -1) means flipping around
                            both axes.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("flip_into", NvtxTrace("cvcuda.flip_into", &FlipInto), "dst"_a, "src"_a, "flipCode"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Flip operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            flipCode (int): Flag to specify how to flip the array; 0 means flipping
                            around the x-axis and positive value (for example, 1) means flipping
                            around y-axis. Negative value (for example, -1) means flipping around
                            both axes.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("flip", NvtxTrace("cvcuda.flip", &FlipVarShape), "src"_a, "flipCode"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Flip operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            flipCode (cvcuda.Tensor): Flag to specify how to flip the array; 0 means flipping
                            around the x-axis and positive value (for example, 1) means flipping
                            around y-axis. Negative value (for example, -1) means flipping around
                            both axes. Specified for all images in batch.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("flip_into", NvtxTrace("cvcuda.flip_into", &FlipVarShapeInto), "dst"_a, "src"_a, "flipCode"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Flip operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            flipCode (cvcuda.Tensor): Flag to specify how to flip the array; 0 means flipping
                            around the x-axis and positive value (for example, 1) means flipping
                            around y-axis. Negative value (for example, -1) means flipping around
                            both axes. Specified for all images in batch.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
