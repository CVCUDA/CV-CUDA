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

#include "../CvtColorUtil.hpp"
#include "Operators.hpp"
#include "VarShapeUtils.hpp"

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
#include <stdexcept>

namespace cvcudapy {

namespace {

class CvtColorOpError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor CvtColorInto(Tensor &output, Tensor &input, NVCVColorConversionCode code, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto cvtColor = CreateOperator<cvcuda::CvtColor>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*cvtColor});

    guard.run([&cvtColor, &pstream, &input, &output, &code]()
              { cvtColor->submit(pstream->cudaHandle(), input, output, code); });

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

bool IsPlanarVarShapeFormat(nvcv::ImageFormat format)
{
    return format.numPlanes() > 1 && format.numPlanes() == format.numChannels();
}

nvcv::ImageFormat PlanarRGBOutputFormat(nvcv::ImageFormat outputFormat)
{
    if (nvcv::DataType channelType = outputFormat.planeDataType(0).channelType(0); channelType != nvcv::TYPE_U8)
    {
        throw CvtColorOpError{"Unsupported planar var-shape CvtColor output data type"};
    }

    switch (outputFormat.swizzle())
    {
    case nvcv::Swizzle::S_XYZ1:
    case nvcv::Swizzle::S_XYZ0:
        return nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB8p};
    case nvcv::Swizzle::S_ZYX1:
    case nvcv::Swizzle::S_ZYX0:
        return nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8p};
    case nvcv::Swizzle::S_XYZW:
        return nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA8p};
    case nvcv::Swizzle::S_ZYXW:
        return nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGRA8p};
    default:
        return outputFormat;
    }
}

nvcv::ImageFormat PreservePlanarVarShapeOutput(nvcv::ImageFormat inputFormat, nvcv::ImageFormat outputFormat)
{
    if (!IsPlanarVarShapeFormat(inputFormat))
    {
        return outputFormat;
    }

    if (outputFormat.numPlanes() != 1)
    {
        return outputFormat;
    }

    if (outputFormat.colorModel() == nvcv::ColorModel::RGB)
    {
        return PlanarRGBOutputFormat(outputFormat);
    }

    if (outputFormat.colorModel() == nvcv::ColorModel::YCbCr
        && outputFormat.chromaSubsampling() == nvcv::ChromaSubsampling::NONE && outputFormat.numChannels() == 3)
    {
        if (nvcv::DataType channelType = outputFormat.planeDataType(0).channelType(0); channelType == nvcv::TYPE_U8)
        {
            return nvcv::ImageFormat{NVCV_IMAGE_FORMAT_YUV8p};
        }
        throw CvtColorOpError{"Unsupported planar var-shape CvtColor YUV output data type"};
    }

    return outputFormat;
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
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*cvtColor});

    guard.run([&cvtColor, &pstream, &input, &output, &code]()
              { cvtColor->submit(pstream->cudaHandle(), input, output, code); });

    return output;
}

ImageBatchVarShape CvtColorVarShape(ImageBatchVarShape &input, NVCVColorConversionCode code,
                                    std::optional<Stream> pstream)
{
    auto inFormat = input.uniqueFormat();
    if (!inFormat)
    {
        throw CvtColorOpError("All images in input must have the same format");
    }
    auto outFormat = GetOutputFormat(inFormat.planeDataType(0), code);
    outFormat      = PreservePlanarVarShapeOutput(inFormat, outFormat);

    ImageBatchVarShape output = CreateSameShapeImageBatch(input, outFormat);

    return CvtColorVarShapeInto(output, input, code, pstream);
}

} // namespace

void ExportOpCvtColor(py::module &m)
{
    using namespace pybind11::literals;

    m.def("cvtcolor", NvtxTrace("cvcuda.cvtcolor", &CvtColor), "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the CVT Color operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("cvtcolor_into", NvtxTrace("cvcuda.cvtcolor_into", &CvtColorInto), "dst"_a, "src"_a, "code"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the CVT Color operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("cvtcolor", NvtxTrace("cvcuda.cvtcolor", &CvtColorVarShape), "src"_a, "code"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the CVT Color operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("cvtcolor_into", NvtxTrace("cvcuda.cvtcolor_into", &CvtColorVarShapeInto), "dst"_a, "src"_a, "code"_a,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the CVT Color operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
