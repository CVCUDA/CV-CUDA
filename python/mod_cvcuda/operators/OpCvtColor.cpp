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
#include "NativeBinding.hpp"
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/Binding.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <map>
#include <stdexcept>

namespace cvcudapy {

namespace {

using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

class CvtColorOpError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

template<class Container>
Container &SubmitCvtColor(Container &output, Container &input, NVCVColorConversionCode code,
                          std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto cvtColor = CreateOperator<cvcuda::CvtColor>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*cvtColor});

    guard.run([&cvtColor, &stream, &input, &output, &code]()
              { cvtColor->submit(stream.handle(), input.impl(), output.impl(), code); });

    return output;
}

Tensor &CvtColorInto(Tensor &output, Tensor &input, NVCVColorConversionCode code, std::optional<Stream *> pstream)
{
    return SubmitCvtColor(output, input, code, pstream);
}

std::shared_ptr<Tensor> CvtColor(Tensor &input, NVCVColorConversionCode code, std::optional<Stream *> pstream)
{
    nvcv::ImageFormat outputFormat = GetOutputFormat(input.dtype(), code);
    nvcv::TensorShape outputShape  = GetOutputTensorShape(input.impl().shape(), outputFormat, code);
    nvcv::DataType    outputDType  = outputFormat.planeDataType(0).channelType(0);

#ifndef NDEBUG
    assert(outputFormat.numPlanes() == 1);
    nvcv::DataType channelDType = outputFormat.planeDataType(0).channelType(0);
    for (int c = 1; c < outputFormat.planeDataType(0).numChannels(); ++c)
    {
        assert(channelDType == outputFormat.planeDataType(0).channelType(c));
    }
#endif

    auto output = native::CreateTensor(outputShape, outputDType);

    CvtColorInto(*output, input, code, pstream);
    return output;
}

bool IsPlanarVarShapeFormat(nvcv::ImageFormat format)
{
    return format.numPlanes() > 1 && format.numPlanes() == format.numChannels();
}

nvcv::ImageFormat PlanarRGBOutputFormat(nvcv::ImageFormat outputFormat)
{
    const nvcv::DataType channelType = outputFormat.planeDataType(0).channelType(0);

    switch (outputFormat.swizzle())
    {
    case nvcv::Swizzle::S_XYZ1:
    case nvcv::Swizzle::S_XYZ0:
        if (channelType == nvcv::TYPE_U8)
            return nvcv::FMT_RGB8p;
        if (channelType == nvcv::TYPE_F16)
            return nvcv::FMT_RGBf16p;
        if (channelType == nvcv::TYPE_F32)
            return nvcv::FMT_RGBf32p;
        break;
    case nvcv::Swizzle::S_ZYX1:
    case nvcv::Swizzle::S_ZYX0:
        if (channelType == nvcv::TYPE_U8)
            return nvcv::FMT_BGR8p;
        if (channelType == nvcv::TYPE_F16)
            return nvcv::FMT_BGRf16p;
        if (channelType == nvcv::TYPE_F32)
            return nvcv::FMT_BGRf32p;
        break;
    case nvcv::Swizzle::S_XYZW:
        if (channelType == nvcv::TYPE_U8)
            return nvcv::FMT_RGBA8p;
        if (channelType == nvcv::TYPE_F16)
            return nvcv::FMT_RGBAf16p;
        if (channelType == nvcv::TYPE_F32)
            return nvcv::FMT_RGBAf32p;
        break;
    case nvcv::Swizzle::S_ZYXW:
        if (channelType == nvcv::TYPE_U8)
            return nvcv::FMT_BGRA8p;
        if (channelType == nvcv::TYPE_F16)
            return nvcv::FMT_BGRAf16p;
        if (channelType == nvcv::TYPE_F32)
            return nvcv::FMT_BGRAf32p;
        break;
    default:
        return outputFormat;
    }

    throw CvtColorOpError{"Unsupported planar var-shape CvtColor RGB output data type"};
}

nvcv::ImageFormat PlanarLabOutputFormat(nvcv::ImageFormat outputFormat)
{
    const nvcv::DataType channelType = outputFormat.planeDataType(0).channelType(0);
    if (channelType == nvcv::TYPE_U8)
        return nvcv::FMT_LAB8p;
    if (channelType == nvcv::TYPE_F16)
        return nvcv::FMT_LABf16p;
    if (channelType == nvcv::TYPE_F32)
        return nvcv::FMT_LABf32p;

    throw CvtColorOpError{"Unsupported planar var-shape CvtColor Lab output data type"};
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

    if (outputFormat.colorModel() == nvcv::ColorModel::LAB)
    {
        return PlanarLabOutputFormat(outputFormat);
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

ImageBatchVarShape &CvtColorVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                         NVCVColorConversionCode code, std::optional<Stream *> pstream)
{
    return SubmitCvtColor(output, input, code, pstream);
}

std::shared_ptr<ImageBatchVarShape> CvtColorVarShape(ImageBatchVarShape &input, NVCVColorConversionCode code,
                                                     std::optional<Stream *> pstream)
{
    auto inFormat = GetUniqueFormat(input);
    if (!inFormat)
    {
        throw CvtColorOpError("All images in input must have the same format");
    }
    auto outFormat = GetOutputFormat(inFormat->planeDataType(0), code);
    outFormat      = PreservePlanarVarShapeOutput(*inFormat, outFormat);

    auto output = CreateSameShapeImageBatch(input, outFormat);

    CvtColorVarShapeInto(*output, input, code, pstream);
    return output;
}

} // namespace

void ExportOpCvtColor(py::module_ &m)
{
    using namespace nanobind::literals;

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
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
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
          py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
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
