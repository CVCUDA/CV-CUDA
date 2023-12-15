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

#include "CvtColorUtil.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/Types.h>
#include <nvcv/cuda/TypeTraits.hpp>
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
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*cvtColor});

    cvtColor->submit(pstream->cudaHandle(), input, output, code);

    return output;
}

Tensor CvtColor(Tensor &input, NVCVColorConversionCode code, std::optional<Stream> pstream)
{
    int  ndim      = input.shape().size();
    auto layout    = input.layout();
    auto outFormat = GetOutputFormat(input.dtype(), code);
    auto out_dtype = outFormat.planeDataType(0).channelType(0);
    if (ndim < 3)
    {
        throw std::runtime_error("Invalid input tensor shape");
    }

    std::array<int64_t, NVCV_TENSOR_MAX_RANK> shape_data;
    for (int d = 0; d < ndim; d++)
    {
        if (layout[d] == 'C')
            shape_data[d] = outFormat.numChannels();
        else
            shape_data[d] = input.shape()[d];
    }
    nvcv::TensorShape out_shape(shape_data.data(), ndim, layout);
    Tensor            output = Tensor::Create(out_shape, out_dtype);
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
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*cvtColor});

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

        cvcuda.cvtcolor(src: nvcv.Tensor, code : NVCVColorConversionCode, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

	Executes the CVT Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the CVT Color operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            code (NVCVColorConversionCode): Code describing the desired color conversion.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("cvtcolor_into", &CvtColorInto, "dst"_a, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.cvtcolor_into(ds : nvcv.Tensor, src: nvcv.Tensor, code : NVCVColorConversionCode, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the CVT Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the CVT Color operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            code (NVCVColorConversionCode): Code describing the desired color conversion.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("cvtcolor", &CvtColorVarShape, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.cvtcolor(src: nvcv.ImageBatchVarShape, code : NVCVColorConversionCode, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

	Executes the CVT Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the CVT Color operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            code (NVCVColorConversionCode): Code describing the desired color conversion.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("cvtcolor_into", &CvtColorVarShapeInto, "dst"_a, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        cvcuda.cvtcolor_into(dst : nvcv.ImageBatchVarShape , src: nvcv.ImageBatchVarShape, code : NVCVColorConversionCode, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the CVT Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the CVT Color operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            dst (ImageBatchVarShape): Output image batch containing the result of the operation.
            code (NVCVColorConversionCode): Code describing the desired color conversion.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
