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

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpCopyMakeBorder.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/TensorLayoutInfo.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class CopyMakeBorderError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

float4 GetBorderValue(const std::vector<float> &borderValue)
{
    size_t bValueDims = borderValue.size();
    if (bValueDims > 4)
    {
        throw CopyMakeBorderError(
            util::ConcatString("Channels of borderValue should <= 4, current is '", bValueDims, "'"));
    }

    float4 bValue;
    for (int i = 0; i < 4; i++)
    {
        const auto valueIdx               = static_cast<size_t>(i);
        nvcv::cuda::GetElement(bValue, i) = bValueDims > valueIdx ? borderValue[valueIdx] : 0.f;
    }
    return bValue;
}

Tensor CopyMakeBorderInto(Tensor &output, Tensor &input, NVCVBorderType borderMode,
                          const std::vector<float> &borderValue, int top, int left, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    float4 bValue = GetBorderValue(borderValue);

    auto copyMakeBorder = CreateOperator<cvcuda::CopyMakeBorder>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*copyMakeBorder});

    guard.run([&copyMakeBorder, &pstream, &input, &output, &top, &left, &borderMode, &bValue]()
              { copyMakeBorder->submit(pstream->cudaHandle(), input, output, top, left, borderMode, bValue); });

    return output;
}

Tensor CopyMakeBorder(Tensor &input, NVCVBorderType borderMode, const std::vector<float> &borderValue, int top,
                      int bottom, int left, int right, std::optional<Stream> pstream)
{
    auto info = nvcv::TensorLayoutInfoImage::Create(input.layout());
    if (!info)
    {
        throw CopyMakeBorderError("Non-supported tensor layout");
    }

    Shape out_shape              = CreateShape(input.shape());
    out_shape[info->idxHeight()] = out_shape[info->idxHeight()].cast<int64_t>() + top + bottom;
    out_shape[info->idxWidth()]  = out_shape[info->idxWidth()].cast<int64_t>() + left + right;

    Tensor output = Tensor::Create(out_shape, input.dtype(), input.layout());

    return CopyMakeBorderInto(output, input, borderMode, borderValue, top, left, pstream);
}

template<class Output>
Output &VarShapeCopyMakeBorderSubmit(Output &output, ImageBatchVarShape &input, NVCVBorderType borderMode,
                                     const std::vector<float> &borderValue, Tensor &top, Tensor &left,
                                     std::optional<Stream> pstream);

Tensor VarShapeCopyMakeBorderStackInto(Tensor &output, ImageBatchVarShape &input, NVCVBorderType borderMode,
                                       const std::vector<float> &borderValue, Tensor &top, Tensor &left,
                                       std::optional<Stream> pstream)
{
    return VarShapeCopyMakeBorderSubmit(output, input, borderMode, borderValue, top, left, pstream);
}

Tensor VarShapeCopyMakeBorderStack(ImageBatchVarShape &input, NVCVBorderType borderMode,
                                   const std::vector<float> &borderValue, Tensor &top, Tensor &left, int out_height,
                                   int out_width, std::optional<Stream> pstream)
{
    auto format = input.uniqueFormat();
    if (!format)
    {
        throw CopyMakeBorderError("All images in input must have the same format.");
    }

    Tensor output = Tensor::CreateForImageBatch(input.numImages(), {out_width, out_height}, format);

    return VarShapeCopyMakeBorderStackInto(output, input, borderMode, borderValue, top, left, pstream);
}

ImageBatchVarShape VarShapeCopyMakeBorderInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                              NVCVBorderType borderMode, const std::vector<float> &borderValue,
                                              Tensor &top, Tensor &left, std::optional<Stream> pstream)
{
    return VarShapeCopyMakeBorderSubmit(output, input, borderMode, borderValue, top, left, pstream);
}

template<class Output>
Output &VarShapeCopyMakeBorderSubmit(Output &output, ImageBatchVarShape &input, NVCVBorderType borderMode,
                                     const std::vector<float> &borderValue, Tensor &top, Tensor &left,
                                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    float4 bValue = GetBorderValue(borderValue);

    auto copyMakeBorder = CreateOperator<cvcuda::CopyMakeBorder>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, top, left});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*copyMakeBorder});

    guard.run([&copyMakeBorder, &pstream, &input, &output, &top, &left, &borderMode, &bValue]()
              { copyMakeBorder->submit(pstream->cudaHandle(), input, output, top, left, borderMode, bValue); });

    return output;
}

ImageBatchVarShape VarShapeCopyMakeBorder(ImageBatchVarShape &input, NVCVBorderType borderMode,
                                          const std::vector<float> &borderValue, Tensor &top, Tensor &left,
                                          std::vector<int> &out_heights, std::vector<int> &out_widths,
                                          std::optional<Stream> pstream)
{
    if (int(out_heights.size()) != input.numImages())
    {
        throw CopyMakeBorderError(util::ConcatString("out_heights.size() != input.numImages, ", out_heights.size(),
                                                     " != ", input.numImages()));
    }

    if (int(out_widths.size()) != input.numImages())
    {
        throw CopyMakeBorderError(
            util::ConcatString("out_widths.size() != input.numImages, ", out_widths.size(), " != ", input.numImages()));
    }
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.uniqueFormat();
    if (!format)
    {
        throw CopyMakeBorderError("All images in input must have the same format.");
    }

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::Size2D size = {out_widths[i], out_heights[i]};
        auto         img  = Image::Create(size, format);
        output.pushBackImage(img);
    }
    return VarShapeCopyMakeBorderInto(output, input, borderMode, borderValue, top, left, pstream);
}

} // namespace

void ExportOpCopyMakeBorder(py::module &m)
{
    using namespace pybind11::literals;

    m.def("copymakeborder", NvtxTrace("cvcuda.copymakeborder", &CopyMakeBorder), "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "bottom"_a, "left"_a, "right"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Copy Make Border operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            border_mode (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            border_value (List[float], optional): Border value to be used for constant border mode, each element of the array corresponds to the
                                       image color channel must be a size <= 4 and dim of 1, where the values specify the border color for each color channel.
            top (int): The top pixel position.
            left (int): The left pixel position.
            bottom (int): The bottom pixel position.
            right (int): The right pixel position.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("copymakeborder_into", NvtxTrace("cvcuda.copymakeborder_into", &CopyMakeBorderInto), "dst"_a, "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "left"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Copy Make Border operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            border_mode (cvcuda.Border): Border mode to be used when accessing elements outside input image.
            border_value (List[float], optional): Border value to be used for constant border mode, each element of the array corresponds to the
                                       image color channel must be a size <= 4 and dim of 1, where the values specify the border color for each color channel.
            top (int): The top pixel position.
            left (int): The left pixel position.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("copymakeborderstack", NvtxTrace("cvcuda.copymakeborderstack", &VarShapeCopyMakeBorderStack), "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "left"_a, "out_height"_a, "out_width"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Copy Make Border Stack operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            border_mode (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            border_value (List[float], optional): Border value to be used for constant border mode, each element of the array corresponds to the
                                       image color channel must be a size <= 4 and dim of 1, where the values specify the border color for each color channel.
            top (cvcuda.Tensor): The top pixel position for each image.
            left (cvcuda.Tensor): The left pixel position for each image.
            out_height (int): The height of the output.
            out_width (int): The width of the output.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output images.

    )pbdoc");

    m.def("copymakeborderstack_into", NvtxTrace("cvcuda.copymakeborderstack_into", &VarShapeCopyMakeBorderStackInto),
          "dst"_a, "src"_a, "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "left"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Copy Make Border Stack operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            border_mode (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            border_value (List[float], optional): Border value to be used for constant border mode, each element of the array corresponds to the
                                       image color channel must be a size <= 4 and dim of 1, where the values specify the border color for each color channel.
            top (cvcuda.Tensor): The top pixel position for each image.
            left (cvcuda.Tensor): The left pixel position for each image.
            out_height (int): The height of the output.
            out_width (int): The width of the output.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("copymakeborder", NvtxTrace("cvcuda.copymakeborder", &VarShapeCopyMakeBorder), "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "left"_a, "out_heights"_a, "out_widths"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Copy Make Border operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            border_mode (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            border_value (List[float], optional): Border value to be used for constant border mode, each element of the array corresponds to the
                                       image color channel must be a size <= 4 and dim of 1, where the values specify the border color for each color channel.
            top (cvcuda.Tensor): The top pixel position for each image.
            left (cvcuda.Tensor): The left pixel position for each image.
            out_heights (cvcuda.Tensor): The heights of each output image.
            out_widths (cvcuda.Tensor):  The widths of each output image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("copymakeborder_into", NvtxTrace("cvcuda.copymakeborder_into", &VarShapeCopyMakeBorderInto), "dst"_a, "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "left"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Copy Make Border operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            border_mode (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            border_value (List[float], optional): Border value to be used for constant border mode, each element of the array corresponds to the
                                       image color channel must be a size <= 4 and dim of 1, where the values specify the border color for each color channel.
            top (cvcuda.Tensor): The top pixel position for each image.
            left (cvcuda.Tensor): The left pixel position for each image.
            out_heights (cvcuda.Tensor): The heights of each output image.
            out_widths (cvcuda.Tensor):  The widths of each output image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
