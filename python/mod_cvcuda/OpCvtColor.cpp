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

#include "CvtColorUtil.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <map>

namespace cvcudapy {

namespace {

Tensor CvtColorInto(Tensor &output, Tensor &input, NVCVColorConversionCode code, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto cvtColor = CreateOperator<cvcuda::CvtColor>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READWRITE, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*cvtColor});

    cvtColor->submit(pstream->cudaHandle(), input, output, code);

    return output;
}

Tensor CvtColor(Tensor &input, NVCVColorConversionCode code, std::optional<Stream> pstream)
{
    nvcv::ImageFormat outputFormat = GetOutputFormat(input.dtype(), code);
    nvcv::TensorShape outputShape  = GetOutputTensorShape(input.shape(), outputFormat, code);
    nvcv::DataType    outputDType  = outputFormat.planeDataType(0).channelType(0);

#ifndef NDEBUG
    assert(outputFormat.numPlanes() == 1);
    nvcv::DataType channelDType = outputFormat.planeDataType(0).channelType(0);
    for (int c = 1; c < outputFormat.planeDataType(0).numChannels(); ++c)
    {
        assert(channelDType == outputFormat.planeDataType(0).channelType(c));
    }
#endif

    Tensor output = Tensor::Create(outputShape, outputDType);

    return CvtColorInto(output, input, code, pstream);
}

ImageBatchVarShape CvtColorVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                        NVCVColorConversionCode code, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto cvtColor = CreateOperator<cvcuda::CvtColor>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READWRITE, {input});
    guard.add(LockMode::LOCK_MODE_READWRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*cvtColor});

    cvtColor->submit(pstream->cudaHandle(), input, output, code);

    return output;
}

ImageBatchVarShape CvtColorVarShape(ImageBatchVarShape &input, NVCVColorConversionCode code,
                                    std::optional<Stream> pstream)
{
    auto inFormat = input.uniqueFormat();
    if (!inFormat || inFormat.numPlanes() != 1)
    {
        throw std::runtime_error("All images in input must have the same single-plane format");
    }
    auto outFormat = GetOutputFormat(inFormat.planeDataType(0), code);

    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::Size2D size = input[i].size();

        auto img = Image::Create(size, outFormat);
        output.pushBack(img);
    }

    return CvtColorVarShapeInto(output, input, code, pstream);
}

} // namespace

void ExportOpCvtColor(py::module &m)
{
    using namespace pybind11::literals;
    py::options options;
    options.disable_function_signatures();

    m.def("cvtcolor", &CvtColor, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.cvtcolor(src: nvcv.Tensor, code: cvcuda.ColorConversion, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

	Executes the CVT Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the CVT Color operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more images.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("cvtcolor_into", &CvtColorInto, "dst"_a, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.cvtcolor_into(dst: nvcv.Tensor, src: nvcv.Tensor, code: cvcuda.ColorConversion, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the CVT Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the CVT Color operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output tensor to store the result of the operation.
            src (nvcv.Tensor): Input tensor containing one or more images.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("cvtcolor", &CvtColorVarShape, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.cvtcolor(src: nvcv.ImageBatchVarShape, code: cvcuda.ColorConversion, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

	Executes the CVT Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the CVT Color operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("cvtcolor_into", &CvtColorVarShapeInto, "dst"_a, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        cvcuda.cvtcolor_into(dst: nvcv.ImageBatchVarShape, src: nvcv.ImageBatchVarShape, code: cvcuda.ColorConversion, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the CVT Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the CVT Color operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            dst (nvcv.ImageBatchVarShape): Output image batch containing the result of the operation.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
