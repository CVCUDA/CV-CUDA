/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    Flip->submit(pstream->cudaHandle(), input, output, flipCode);

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

    flip->submit(pstream->cudaHandle(), input, output, flipCode);

    return output;
}

ImageBatchVarShape FlipVarShape(ImageBatchVarShape &input, Tensor &flipCode, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return FlipVarShapeInto(output, input, flipCode, pstream);
}

} // namespace

void ExportOpFlip(py::module &m)
{
    using namespace pybind11::literals;
    py::options options;
    options.disable_function_signatures();

    m.def("flip", &Flip, "src"_a, "flipCode"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.flip(src: nvcv.Tensor, flipCode: int, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

        Executes the Flip operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Flip operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more images.
            flipCode (int): Flag to specify how to flip the array; 0 means flipping
                            around the x-axis and positive value (for example, 1) means flipping
                            around y-axis. Negative value (for example, -1) means flipping around
                            both axes.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("flip_into", &FlipInto, "dst"_a, "src"_a, "flipCode"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.flip_into(dst: nvcv.Tensor, src: nvcv.Tensor, flipCode: int, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the Flip operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Flip operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output tensor to store the result of the operation.
            src (nvcv.Tensor): Input tensor containing one or more images.
            flipCode (int): Flag to specify how to flip the array; 0 means flipping
                            around the x-axis and positive value (for example, 1) means flipping
                            around y-axis. Negative value (for example, -1) means flipping around
                            both axes.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("flip", &FlipVarShape, "src"_a, "flipCode"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.flip(src: nvcv.ImageBatchVarShape, flipCode: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

	Executes the Flip operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Flip operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            flipCode (nvcv.Tensor): Flag to specify how to flip the array; 0 means flipping
                            around the x-axis and positive value (for example, 1) means flipping
                            around y-axis. Negative value (for example, -1) means flipping around
                            both axes. Specified for all images in batch.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("flip_into", &FlipVarShapeInto, "dst"_a, "src"_a, "flipCode"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.flip_into(dst:nvcv.ImageBatchVarShape, src: nvcv.ImageBatchVarShape, flipCode: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None)

        Executes the Flip operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Flip operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            dst (nvcv.ImageBatchVarShape): Output image batch containing the result of the operation.
            flipCode (nvcv.Tensor): Flag to specify how to flip the array; 0 means flipping
                            around the x-axis and positive value (for example, 1) means flipping
                            around y-axis. Negative value (for example, -1) means flipping around
                            both axes. Specified for all images in batch.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
