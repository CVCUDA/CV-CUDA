/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/DataType.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor CopyMakeBorderInto(Tensor &output, Tensor &input, NVCVBorderType borderMode,
                          const std::vector<float> &borderValue, int top, int left, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    size_t bValueDims = borderValue.size();
    if (bValueDims > 4)
    {
        throw std::runtime_error(
            util::FormatString("Channels of borderValue should <= 4, current is '%lu'", bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++)
    {
        nvcv::cuda::GetElement(bValue, i) = bValueDims > i ? borderValue[i] : 0.f;
    }

    auto copyMakeBorder = CreateOperator<cvcuda::CopyMakeBorder>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*copyMakeBorder});

    copyMakeBorder->submit(pstream->cudaHandle(), input, output, top, left, borderMode, bValue);

    return output;
}

Tensor CopyMakeBorder(Tensor &input, NVCVBorderType borderMode, const std::vector<float> &borderValue, int top,
                      int bottom, int left, int right, std::optional<Stream> pstream)
{
    nvcv::TensorShape in_shape = input.shape();
    Shape             out_shape(&in_shape[0], &in_shape[0] + in_shape.size());
    int               cdim = out_shape.size() - 1;
    out_shape[cdim - 2] += top + bottom;
    out_shape[cdim - 1] += left + right;

    Tensor output
        = Tensor::Create(nvcv::TensorShape(out_shape.data(), out_shape.size(), input.layout()), input.dtype());

    return CopyMakeBorderInto(output, input, borderMode, borderValue, top, left, pstream);
}

Tensor VarShapeCopyMakeBorderStackInto(Tensor &output, ImageBatchVarShape &input, NVCVBorderType borderMode,
                                       const std::vector<float> &borderValue, Tensor &top, Tensor &left,
                                       std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    size_t bValueDims = borderValue.size();
    if (bValueDims > 4)
    {
        throw std::runtime_error(
            util::FormatString("Channels of borderValue should <= 4, current is '%lu'", bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++)
    {
        nvcv::cuda::GetElement(bValue, i) = bValueDims > i ? borderValue[i] : 0.f;
    }

    auto copyMakeBorder = CreateOperator<cvcuda::CopyMakeBorder>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, top, left});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*copyMakeBorder});

    copyMakeBorder->submit(pstream->cudaHandle(), input, output, top, left, borderMode, bValue);

    return output;
}

Tensor VarShapeCopyMakeBorderStack(ImageBatchVarShape &input, NVCVBorderType borderMode,
                                   const std::vector<float> &borderValue, Tensor &top, Tensor &left, int out_height,
                                   int out_width, std::optional<Stream> pstream)
{
    auto format = input.uniqueFormat();
    if (!format)
    {
        throw std::runtime_error("All images in input must have the same format.");
    }

    Tensor output = Tensor::CreateForImageBatch(input.numImages(), {out_width, out_height}, format);

    return VarShapeCopyMakeBorderStackInto(output, input, borderMode, borderValue, top, left, pstream);
}

ImageBatchVarShape VarShapeCopyMakeBorderInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                              NVCVBorderType borderMode, const std::vector<float> &borderValue,
                                              Tensor &top, Tensor &left, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    size_t bValueDims = borderValue.size();
    if (bValueDims > 4)
    {
        throw std::runtime_error(
            util::FormatString("Channels of borderValue should <= 4, current is '%lu'", bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++)
    {
        nvcv::cuda::GetElement(bValue, i) = bValueDims > i ? borderValue[i] : 0.f;
    }

    auto copyMakeBorder = CreateOperator<cvcuda::CopyMakeBorder>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, top, left});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*copyMakeBorder});

    copyMakeBorder->submit(pstream->cudaHandle(), input, output, top, left, borderMode, bValue);

    return output;
}

ImageBatchVarShape VarShapeCopyMakeBorder(ImageBatchVarShape &input, NVCVBorderType borderMode,
                                          const std::vector<float> &borderValue, Tensor &top, Tensor &left,
                                          std::vector<int> &out_heights, std::vector<int> &out_widths,
                                          std::optional<Stream> pstream)
{
    if (int(out_heights.size()) != input.numImages())
    {
        throw std::runtime_error(util::FormatString("out_heights.size() != input.numImages, %lu != %d",
                                                    out_heights.size(), input.numImages()));
    }

    if (int(out_widths.size()) != input.numImages())
    {
        throw std::runtime_error(util::FormatString("out_widths.size() != input.numImages, %lu != %d",
                                                    out_heights.size(), input.numImages()));
    }
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.uniqueFormat();
    if (!format)
    {
        throw std::runtime_error("All images in input must have the same format.");
    }

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::Size2D size = {out_widths[i], out_heights[i]};
        auto         img  = Image::Create(size, format);
        output.pushBack(img);
    }
    return VarShapeCopyMakeBorderInto(output, input, borderMode, borderValue, top, left, pstream);
}

} // namespace

void ExportOpCopyMakeBorder(py::module &m)
{
    using namespace pybind11::literals;

    m.def("copymakeborder", &CopyMakeBorder, "src"_a, "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "bottom"_a, "left"_a, "right"_a,
          "stream"_a       = nullptr);
    m.def("copymakeborder_into", &CopyMakeBorderInto, "dst"_a, "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "left"_a, "stream"_a = nullptr);
    m.def("copymakeborderstack", &VarShapeCopyMakeBorderStack, "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "left"_a, "out_height"_a, "out_width"_a, "stream"_a = nullptr);
    m.def("copymakeborderstack_into", &VarShapeCopyMakeBorderStackInto, "dst"_a, "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "left"_a, "stream"_a = nullptr);
    m.def("copymakeborder", &VarShapeCopyMakeBorder, "src"_a, "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "left"_a, "out_heights"_a, "out_widths"_a,
          "stream"_a       = nullptr);
    m.def("copymakeborder_into", &VarShapeCopyMakeBorderInto, "dst"_a, "src"_a,
          "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a = std::vector<float>(),
          py::kw_only(), "top"_a, "left"_a, "stream"_a = nullptr);
}

} // namespace cvcudapy
