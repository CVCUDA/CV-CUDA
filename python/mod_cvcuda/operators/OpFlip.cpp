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

#include "../nvcv/ImageBatch.hpp"
#include "../nvcv/Tensor.hpp"
#include "NativeBinding.hpp"
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/Binding.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpFlip.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

namespace cvcudapy {

namespace {
nvcvpy::priv::Tensor &FlipInto(nvcvpy::priv::Tensor &output, nvcvpy::priv::Tensor &input, int32_t flipCode,
                               std::optional<nvcvpy::priv::Stream *> pstream)
{
    nvcvpy::priv::Stream &stream = pstream && *pstream ? **pstream : nvcvpy::priv::Stream::Current();

    auto flip = CreateOperator<cvcuda::Flip>(0);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*flip});

    guard.run([&flip, &stream, &input, &output, &flipCode]()
              { flip->submit(stream.handle(), input.impl(), output.impl(), flipCode); });

    return output;
}

std::shared_ptr<nvcvpy::priv::Tensor> Flip(nvcvpy::priv::Tensor &input, int32_t flipCode,
                                           std::optional<nvcvpy::priv::Stream *> pstream)
{
    auto stream = native::GetStream(pstream).sharedStream();
    auto output = nvcvpy::priv::Tensor::CreateFromReqs(
        nvcv::Tensor::CalcRequirements(input.impl().shape(), input.dtype()), stream.get());

    FlipInto(*output, input, flipCode, stream.get());
    return output;
}

nvcvpy::priv::ImageBatchVarShape &FlipVarShapeInto(nvcvpy::priv::ImageBatchVarShape     &output,
                                                   nvcvpy::priv::ImageBatchVarShape     &input,
                                                   nvcvpy::priv::Tensor                 &flipCode,
                                                   std::optional<nvcvpy::priv::Stream *> pstream)
{
    nvcvpy::priv::Stream &stream = pstream && *pstream ? **pstream : nvcvpy::priv::Stream::Current();

    auto flip = CreateOperator<cvcuda::Flip>(0);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, flipCode});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*flip});

    guard.run([&flip, &stream, &input, &output, &flipCode]()
              { flip->submit(stream.handle(), input.impl(), output.impl(), flipCode.impl()); });

    return output;
}

std::shared_ptr<nvcvpy::priv::ImageBatchVarShape> FlipVarShape(nvcvpy::priv::ImageBatchVarShape     &input,
                                                               nvcvpy::priv::Tensor                 &flipCode,
                                                               std::optional<nvcvpy::priv::Stream *> pstream)
{
    auto stream = native::GetStream(pstream).sharedStream();
    auto output = CreateSameShapeImageBatch(input, *stream);

    FlipVarShapeInto(*output, input, flipCode, stream.get());
    return output;
}

} // namespace

void ExportOpFlip(py::module_ &m)
{
    using namespace nanobind::literals;

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
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
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
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
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
